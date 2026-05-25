const std = @import("std");
const vector = @import("vector.zig");

/// Compact, mmap-friendly model format for the bucketed KD-tree exact NN index.
///
/// Layout (all little-endian; the contest target and dev host are both x86_64,
/// so sections are cast directly out of the backing buffer with no byte-swapping):
///
///   [Header]                              64 bytes, 8-aligned
///   [BucketEntry × bucket_count]          stage-1 directory (one per 4-bit key)
///   [KDNode × node_count]                 stage-2 per-bucket KD-trees, bucket-major
///   [vectors: count × 16 × i16]           64-aligned; bucket-major, leaf-contiguous
///   [labels: ceil(count/8) bytes]         1 bit per vector (1 = fraud), same order
///
/// Vectors are reordered so each bucket is contiguous and, within a bucket, each
/// KD-tree leaf is a contiguous vector range. Every node carries the axis-aligned
/// bounding box of its subtree over all stored dims; the squared distance from a
/// query to that box is an exact lower bound used to prune whole subtrees.
/// Stable format identifier — never changes. The layout version lives in the
/// `version` header field so it can bump independently; `parse` rejects any model
/// whose version it doesn't understand.
pub const magic = "ZORDONDB".*;
/// v3: per-bucket KD-tree (replaces the flat 2-D cell grid of v2) — bounds the
/// worst-case (outlier) scan, which a flat grid could not.
pub const format_version: u32 = 3;
pub const bucket_count = 16;
/// Default KD-tree leaf size: small enough that an outlier prunes down to a few
/// leaves, large enough to amortize traversal over a contiguous SIMD scan.
pub const default_leaf_size: u16 = 64;

pub const Header = extern struct {
    magic: [8]u8,
    version: u32,
    count: u32,
    dims: u16,
    stored_dims: u16,
    scale: u16,
    leaf_size: u16,
    bucket_count: u16,
    _pad0: u16 = 0,
    node_count: u32,
    buckets_off: u64,
    nodes_off: u64,
    vectors_off: u64,
    labels_off: u64,
};

pub const BucketEntry = extern struct {
    /// Index (in vectors, not bytes) of this bucket's first vector.
    vec_start: u32,
    vec_count: u32,
    /// Index into the node array of this bucket's KD-tree root (valid iff vec_count>0).
    root: u32,
    /// Number of KD-tree nodes in this bucket (for stats / verification).
    node_count: u32,
};

/// A KD-tree node. `count > 0` marks a leaf holding `count` contiguous vectors
/// starting at vector index `lo`. `count == 0` marks an internal node whose two
/// children are node indices `lo` (left) and `hi` (right). Either way `box_lo`/
/// `box_hi` is the tight bounding box of every vector in the subtree.
pub const KDNode = extern struct {
    box_lo: [vector.stored_dims]i16,
    box_hi: [vector.stored_dims]i16,
    lo: u32,
    hi: u32,
    count: u32,
};

comptime {
    std.debug.assert(@sizeOf(Header) == 64);
    std.debug.assert(@sizeOf(BucketEntry) == 16);
    std.debug.assert(@sizeOf(KDNode) == 76);
}

pub const buffer_align: std.mem.Alignment = .fromByteUnits(64);

pub const Model = struct {
    /// The backing allocation. All slices below point into it (zero-copy).
    buffer: []align(64) const u8,
    count: u32,
    leaf_size: u16,
    buckets: []const BucketEntry,
    nodes: []const KDNode,
    /// count * stored_dims i16 values.
    vectors: []const i16,
    labels: []const u8,

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        allocator.free(self.buffer);
        self.* = undefined;
    }

    pub fn labelIsFraud(self: Model, index: usize) bool {
        return (self.labels[index / 8] & (@as(u8, 1) << @intCast(index % 8))) != 0;
    }

    /// The 16 i16 lanes of vector `index`.
    pub fn vectorAt(self: Model, index: usize) *const [vector.stored_dims]i16 {
        return self.vectors[index * vector.stored_dims ..][0..vector.stored_dims];
    }
};

pub fn load(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !Model {
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(
        io,
        path,
        allocator,
        .limited(1_000_000_000),
        buffer_align,
        null,
    );
    errdefer allocator.free(bytes);
    return parse(bytes);
}

/// Build a `Model` view over an already-loaded, 64-aligned buffer. The returned
/// model takes ownership of `bytes` (freed by `deinit`).
pub fn parse(bytes: []align(64) const u8) !Model {
    if (bytes.len < @sizeOf(Header)) return error.InvalidModel;
    const header: *const Header = @ptrCast(bytes.ptr);
    if (!std.mem.eql(u8, &header.magic, &magic)) return error.InvalidModel;
    if (header.version != format_version) return error.UnsupportedModelVersion;
    if (header.dims != vector.dims or
        header.stored_dims != vector.stored_dims or
        header.scale != vector.scale or
        header.bucket_count != bucket_count) return error.InvalidModel;

    const count = header.count;
    const node_count = header.node_count;

    const vectors_len = @as(usize, count) * vector.stored_dims;
    const labels_len = (@as(usize, count) + 7) / 8;

    // Bounds-check every section against the buffer.
    if (header.buckets_off + bucket_count * @sizeOf(BucketEntry) > bytes.len) return error.InvalidModel;
    if (header.nodes_off + @as(usize, node_count) * @sizeOf(KDNode) > bytes.len) return error.InvalidModel;
    if (header.vectors_off + vectors_len * @sizeOf(i16) > bytes.len) return error.InvalidModel;
    if (header.labels_off + labels_len > bytes.len) return error.InvalidModel;

    const buckets_ptr: [*]const BucketEntry = @ptrCast(@alignCast(bytes.ptr + header.buckets_off));
    const nodes_ptr: [*]const KDNode = @ptrCast(@alignCast(bytes.ptr + header.nodes_off));
    const vectors_ptr: [*]const i16 = @ptrCast(@alignCast(bytes.ptr + header.vectors_off));

    return .{
        .buffer = bytes,
        .count = count,
        .leaf_size = header.leaf_size,
        .buckets = buckets_ptr[0..bucket_count],
        .nodes = nodes_ptr[0..node_count],
        .vectors = vectors_ptr[0..vectors_len],
        .labels = bytes[header.labels_off..][0..labels_len],
    };
}

pub const Item = struct {
    vec: [vector.stored_dims]i16,
    fraud: bool,
};

/// Build the model byte image from quantized items. Reorders the items into
/// bucket-major, leaf-contiguous layout and builds a KD-tree per bucket. The caller
/// owns the returned buffer (write it to disk, or hand it to `parse`).
pub fn build(allocator: std.mem.Allocator, items: []const Item, leaf_size_arg: u16) ![]align(64) u8 {
    const count = items.len;
    if (count > std.math.maxInt(u32)) return error.TooManyItems;
    const leaf_size: u32 = if (leaf_size_arg == 0) default_leaf_size else leaf_size_arg;

    // --- stage 1: group item indices by bucket key (counting sort) ---
    const keys = try allocator.alloc(u4, count);
    defer allocator.free(keys);
    var bucket_sizes = [_]u32{0} ** bucket_count;
    for (items, 0..) |item, i| {
        const k = vector.bucketKey(item.vec);
        keys[i] = k;
        bucket_sizes[k] += 1;
    }
    var bucket_start = [_]u32{0} ** (bucket_count + 1);
    for (0..bucket_count) |k| bucket_start[k + 1] = bucket_start[k] + bucket_sizes[k];

    const by_bucket = try allocator.alloc(u32, count);
    defer allocator.free(by_bucket);
    {
        var cursor = bucket_start;
        for (items, 0..) |_, i| {
            const k = keys[i];
            by_bucket[cursor[k]] = @intCast(i);
            cursor[k] += 1;
        }
    }

    // --- stage 2: per bucket, build a KD-tree over its members ---
    const order = try allocator.alloc(u32, count); // final vector permutation
    defer allocator.free(order);
    var buckets: [bucket_count]BucketEntry = undefined;
    var nodes: std.ArrayList(KDNode) = .empty;
    defer nodes.deinit(allocator);

    var out_pos: u32 = 0;
    for (0..bucket_count) |k| {
        // `by_bucket` is partitioned in place by the recursive build, so this slice
        // is the bucket's mutable working set of item indices.
        const members = by_bucket[bucket_start[k]..bucket_start[k + 1]];
        const node_start: u32 = @intCast(nodes.items.len);
        buckets[k] = .{ .vec_start = out_pos, .vec_count = @intCast(members.len), .root = 0, .node_count = 0 };
        if (members.len == 0) continue;
        buckets[k].root = try buildKdNode(allocator, items, members, leaf_size, &nodes, order, &out_pos);
        buckets[k].node_count = @as(u32, @intCast(nodes.items.len)) - node_start;
    }
    std.debug.assert(out_pos == count);

    // --- lay out the byte image ---
    const node_total = nodes.items.len;
    const buckets_off: usize = @sizeOf(Header);
    const nodes_off: usize = buckets_off + bucket_count * @sizeOf(BucketEntry);
    const vectors_off: usize = std.mem.alignForward(usize, nodes_off + node_total * @sizeOf(KDNode), 64);
    const vectors_len = count * vector.stored_dims;
    const labels_off: usize = vectors_off + vectors_len * @sizeOf(i16);
    const labels_len = (count + 7) / 8;
    const total = labels_off + labels_len;

    const buf = try allocator.alignedAlloc(u8, buffer_align, total);
    errdefer allocator.free(buf);
    @memset(buf, 0);

    const header: *Header = @ptrCast(buf.ptr);
    header.* = .{
        .magic = magic,
        .version = format_version,
        .count = @intCast(count),
        .dims = vector.dims,
        .stored_dims = vector.stored_dims,
        .scale = @intCast(vector.scale),
        .leaf_size = @intCast(leaf_size),
        .bucket_count = bucket_count,
        .node_count = @intCast(node_total),
        .buckets_off = buckets_off,
        .nodes_off = nodes_off,
        .vectors_off = vectors_off,
        .labels_off = labels_off,
    };

    const out_buckets: [*]BucketEntry = @ptrCast(@alignCast(buf.ptr + buckets_off));
    @memcpy(out_buckets[0..bucket_count], &buckets);
    const out_nodes: [*]KDNode = @ptrCast(@alignCast(buf.ptr + nodes_off));
    @memcpy(out_nodes[0..node_total], nodes.items);

    const out_vectors: [*]i16 = @ptrCast(@alignCast(buf.ptr + vectors_off));
    const out_labels = buf[labels_off..][0..labels_len];
    for (order[0..count], 0..) |item_idx, j| {
        @memcpy(out_vectors[j * vector.stored_dims ..][0..vector.stored_dims], &items[item_idx].vec);
        if (items[item_idx].fraud) out_labels[j / 8] |= @as(u8, 1) << @intCast(j % 8);
    }

    return buf;
}

const SortCtx = struct {
    items: []const Item,
    dim: usize,
    fn less(ctx: SortCtx, a: u32, b: u32) bool {
        return ctx.items[a].vec[ctx.dim] < ctx.items[b].vec[ctx.dim];
    }
};

/// Recursively build a balanced KD-tree over `members` (an item-index slice that is
/// partitioned in place). Leaves of <= `leaf_size` members are emitted as contiguous
/// vector ranges appended to `order`. Internal nodes split at the median of the
/// widest-extent dimension. Returns the index of the subtree root in `nodes`
/// (children are appended before their parent).
fn buildKdNode(
    allocator: std.mem.Allocator,
    items: []const Item,
    members: []u32,
    leaf_size: u32,
    nodes: *std.ArrayList(KDNode),
    order: []u32,
    out_pos: *u32,
) error{OutOfMemory}!u32 {
    var box_lo: [vector.stored_dims]i16 = .{std.math.maxInt(i16)} ** vector.stored_dims;
    var box_hi: [vector.stored_dims]i16 = .{std.math.minInt(i16)} ** vector.stored_dims;
    for (members) |idx| {
        const v = items[idx].vec;
        for (0..vector.stored_dims) |d| {
            box_lo[d] = @min(box_lo[d], v[d]);
            box_hi[d] = @max(box_hi[d], v[d]);
        }
    }

    // widest dimension (0 only when every member is identical on all dims).
    var split_dim: usize = 0;
    var widest: i32 = -1;
    for (0..vector.stored_dims) |d| {
        const e = @as(i32, box_hi[d]) - box_lo[d];
        if (e > widest) {
            widest = e;
            split_dim = d;
        }
    }

    if (members.len <= leaf_size or widest == 0) {
        const vec_start = out_pos.*;
        for (members) |idx| {
            order[out_pos.*] = idx;
            out_pos.* += 1;
        }
        const idx: u32 = @intCast(nodes.items.len);
        try nodes.append(allocator, .{ .box_lo = box_lo, .box_hi = box_hi, .lo = vec_start, .hi = 0, .count = @intCast(members.len) });
        return idx;
    }

    std.mem.sort(u32, members, SortCtx{ .items = items, .dim = split_dim }, SortCtx.less);
    const mid = members.len / 2;
    const left = try buildKdNode(allocator, items, members[0..mid], leaf_size, nodes, order, out_pos);
    const right = try buildKdNode(allocator, items, members[mid..], leaf_size, nodes, order, out_pos);
    const idx: u32 = @intCast(nodes.items.len);
    try nodes.append(allocator, .{ .box_lo = box_lo, .box_hi = box_hi, .lo = left, .hi = right, .count = 0 });
    return idx;
}

test "build then parse round-trips a tiny model" {
    const allocator = std.testing.allocator;
    var items = [_]Item{
        .{ .vec = q(.{ 0.0, 0.1, 0.2, 0.3, 0.4, -1, -1, 0.1, 0.1, 0, 1, 0, 0.15, 0.01 }), .fraud = false },
        .{ .vec = q(.{ 0.9, 0.9, 1.0, 0.2, 0.8, -1, -1, 0.95, 1.0, 0, 1, 1, 0.75, 0.005 }), .fraud = true },
        .{ .vec = q(.{ 0.05, 0.1, 0.2, 0.35, 0.5, -1, -1, 0.12, 0.1, 0, 1, 0, 0.15, 0.01 }), .fraud = false },
    };
    const bytes = try build(allocator, &items, 8);
    var model = try parse(bytes);
    defer model.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 3), model.count);
    try std.testing.expectEqual(@as(u16, 16), @as(u16, bucket_count));
    // every vector is present exactly once, labels follow the reorder
    var fraud_seen: u32 = 0;
    for (0..model.count) |i| {
        if (model.labelIsFraud(i)) fraud_seen += 1;
    }
    try std.testing.expectEqual(@as(u32, 1), fraud_seen);

    // bucket counts sum to total
    var total: u32 = 0;
    for (model.buckets) |b| total += b.vec_count;
    try std.testing.expectEqual(@as(u32, 3), total);
}

test "parse rejects an unknown format version" {
    const allocator = std.testing.allocator;
    var items = [_]Item{
        .{ .vec = q(.{ 0.0, 0.1, 0.2, 0.3, 0.4, -1, -1, 0.1, 0.1, 0, 1, 0, 0.15, 0.01 }), .fraud = false },
    };
    const bytes = try build(allocator, &items, 8);
    defer allocator.free(bytes); // parse returns an error, so it never takes ownership
    const header: *Header = @ptrCast(bytes.ptr);
    header.version +%= 1;
    try std.testing.expectError(error.UnsupportedModelVersion, parse(bytes));
}

fn q(v: vector.Vector) [vector.stored_dims]i16 {
    return vector.quantize(v);
}
