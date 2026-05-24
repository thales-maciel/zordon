const std = @import("std");
const vector = @import("vector.zig");

/// Compact, mmap-friendly model format for the two-stage exact NN index.
///
/// Layout (all little-endian; the contest target and dev host are both x86_64,
/// so sections are cast directly out of the backing buffer with no byte-swapping):
///
///   [Header]                              64 bytes, 8-aligned
///   [BucketEntry × bucket_count]          stage-1 directory (one per 4-bit key)
///   [CellEntry × cell_count]              stage-2 grid cells, grouped per bucket
///   [vectors: count × 16 × i16]           64-aligned; bucket-major, cell-sorted
///   [labels: ceil(count/8) bytes]         1 bit per vector (1 = fraud), same order
///
/// Vectors are reordered so each bucket is contiguous and, within a bucket, each
/// grid cell is contiguous. A CellEntry therefore names a contiguous vector range
/// plus the tight [lo,hi] extent of its two grid dimensions (for the lower bound).
/// Stable format identifier — never changes. The layout version lives in the
/// `version` header field so it can bump independently; `parse` rejects any model
/// whose version it doesn't understand.
pub const magic = "ZORDONDB".*;
pub const format_version: u32 = 1;
pub const bucket_count = 16;
pub const default_bins = 48;

pub const Header = extern struct {
    magic: [8]u8,
    version: u32,
    count: u32,
    dims: u16,
    stored_dims: u16,
    scale: u16,
    bins: u16,
    bucket_count: u16,
    _pad0: u16 = 0,
    cell_count: u32,
    buckets_off: u64,
    cells_off: u64,
    vectors_off: u64,
    labels_off: u64,
};

pub const BucketEntry = extern struct {
    /// Index (in vectors, not bytes) of this bucket's first vector.
    vec_start: u32,
    vec_count: u32,
    /// Index into the cell array of this bucket's first cell.
    cell_start: u32,
    cell_count: u32,
    /// The two stored dimensions this bucket's grid partitions on.
    dim_a: u8,
    dim_b: u8,
    _pad: u16 = 0,
};

pub const CellEntry = extern struct {
    /// Tight extent of dim_a / dim_b over the cell's members (quantized units).
    /// Used as a lower bound: any member's value lies within [lo, hi].
    lo_a: i32,
    hi_a: i32,
    lo_b: i32,
    hi_b: i32,
    vec_start: u32,
    vec_count: u32,
};

comptime {
    std.debug.assert(@sizeOf(Header) == 64);
    std.debug.assert(@sizeOf(BucketEntry) == 20);
    std.debug.assert(@sizeOf(CellEntry) == 24);
}

pub const buffer_align: std.mem.Alignment = .fromByteUnits(64);

pub const Model = struct {
    /// The backing allocation. All slices below point into it (zero-copy).
    buffer: []align(64) const u8,
    count: u32,
    bins: u16,
    buckets: []const BucketEntry,
    cells: []const CellEntry,
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
    const bins = header.bins;
    const cell_count = header.cell_count;

    const vectors_len = @as(usize, count) * vector.stored_dims;
    const labels_len = (@as(usize, count) + 7) / 8;

    // Bounds-check every section against the buffer.
    if (header.buckets_off + bucket_count * @sizeOf(BucketEntry) > bytes.len) return error.InvalidModel;
    if (header.cells_off + @as(usize, cell_count) * @sizeOf(CellEntry) > bytes.len) return error.InvalidModel;
    if (header.vectors_off + vectors_len * @sizeOf(i16) > bytes.len) return error.InvalidModel;
    if (header.labels_off + labels_len > bytes.len) return error.InvalidModel;

    const buckets_ptr: [*]const BucketEntry = @ptrCast(@alignCast(bytes.ptr + header.buckets_off));
    const cells_ptr: [*]const CellEntry = @ptrCast(@alignCast(bytes.ptr + header.cells_off));
    const vectors_ptr: [*]const i16 = @ptrCast(@alignCast(bytes.ptr + header.vectors_off));

    return .{
        .buffer = bytes,
        .count = count,
        .bins = bins,
        .buckets = buckets_ptr[0..bucket_count],
        .cells = cells_ptr[0..cell_count],
        .vectors = vectors_ptr[0..vectors_len],
        .labels = bytes[header.labels_off..][0..labels_len],
    };
}

pub const Item = struct {
    vec: [vector.stored_dims]i16,
    fraud: bool,
};

/// Build the model byte image from quantized items. Reorders the items into
/// bucket-major, cell-sorted layout and computes the per-bucket grid. The caller
/// owns the returned buffer (write it to disk, or hand it to `parse`).
pub fn build(allocator: std.mem.Allocator, items: []const Item, bins: u16) ![]align(64) u8 {
    const count = items.len;
    if (count > std.math.maxInt(u32)) return error.TooManyItems;
    std.debug.assert(bins >= 1);

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

    // --- stage 2: per bucket, pick grid dims and cell-sort members ---
    const order = try allocator.alloc(u32, count); // final vector permutation
    defer allocator.free(order);
    var buckets: [bucket_count]BucketEntry = undefined;
    var cells: std.ArrayList(CellEntry) = .empty;
    defer cells.deinit(allocator);

    const ncells: usize = @as(usize, bins) * bins;
    const cell_hist = try allocator.alloc(u32, ncells + 1);
    defer allocator.free(cell_hist);
    const member_cell = try allocator.alloc(u32, count); // scratch, reused per bucket
    defer allocator.free(member_cell);

    var out_pos: u32 = 0;
    for (0..bucket_count) |k| {
        const members = by_bucket[bucket_start[k]..bucket_start[k + 1]];
        buckets[k] = .{
            .vec_start = out_pos,
            .vec_count = @intCast(members.len),
            .cell_start = @intCast(cells.items.len),
            .cell_count = 0,
            .dim_a = 0,
            .dim_b = 1,
        };
        if (members.len == 0) continue;

        const da, const db = pickGridDims(items, members);
        buckets[k].dim_a = @intCast(da);
        buckets[k].dim_b = @intCast(db);

        // grid extent of the two dims over this bucket
        const ext_a = extent(items, members, da);
        const ext_b = extent(items, members, db);

        // counting-sort members by cell id
        @memset(cell_hist, 0);
        for (members, 0..) |item_idx, m| {
            const ca = cellOf(items[item_idx].vec[da], ext_a, bins);
            const cb = cellOf(items[item_idx].vec[db], ext_b, bins);
            const cid = ca * bins + cb;
            member_cell[m] = cid;
            cell_hist[cid + 1] += 1;
        }
        for (0..ncells) |c| cell_hist[c + 1] += cell_hist[c];
        // cell_hist[c] is now the start offset of cell c within `members`
        const cell_off = try allocator.alloc(u32, ncells);
        defer allocator.free(cell_off);
        @memcpy(cell_off, cell_hist[0..ncells]);

        const sorted = try allocator.alloc(u32, members.len);
        defer allocator.free(sorted);
        for (members, 0..) |item_idx, m| {
            const cid = member_cell[m];
            sorted[cell_off[cid]] = item_idx;
            cell_off[cid] += 1;
        }

        // emit non-empty cells with tight lo/hi, append members to `order`
        for (0..ncells) |c| {
            const s = cell_hist[c];
            const e = cell_hist[c + 1];
            if (s == e) continue;
            var lo_a: i32 = std.math.maxInt(i32);
            var hi_a: i32 = std.math.minInt(i32);
            var lo_b: i32 = std.math.maxInt(i32);
            var hi_b: i32 = std.math.minInt(i32);
            const cell_vec_start = out_pos;
            for (sorted[s..e]) |item_idx| {
                const va: i32 = items[item_idx].vec[da];
                const vb: i32 = items[item_idx].vec[db];
                lo_a = @min(lo_a, va);
                hi_a = @max(hi_a, va);
                lo_b = @min(lo_b, vb);
                hi_b = @max(hi_b, vb);
                order[out_pos] = item_idx;
                out_pos += 1;
            }
            try cells.append(allocator, .{
                .lo_a = lo_a,
                .hi_a = hi_a,
                .lo_b = lo_b,
                .hi_b = hi_b,
                .vec_start = cell_vec_start,
                .vec_count = e - s,
            });
        }
        buckets[k].cell_count = @as(u32, @intCast(cells.items.len)) - buckets[k].cell_start;
    }
    std.debug.assert(out_pos == count);

    // --- lay out the byte image ---
    const cell_total = cells.items.len;
    const buckets_off: usize = @sizeOf(Header);
    const cells_off: usize = buckets_off + bucket_count * @sizeOf(BucketEntry);
    const vectors_off: usize = std.mem.alignForward(usize, cells_off + cell_total * @sizeOf(CellEntry), 64);
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
        .bins = bins,
        .bucket_count = bucket_count,
        .cell_count = @intCast(cell_total),
        .buckets_off = buckets_off,
        .cells_off = cells_off,
        .vectors_off = vectors_off,
        .labels_off = labels_off,
    };

    const out_buckets: [*]BucketEntry = @ptrCast(@alignCast(buf.ptr + buckets_off));
    @memcpy(out_buckets[0..bucket_count], &buckets);
    const out_cells: [*]CellEntry = @ptrCast(@alignCast(buf.ptr + cells_off));
    @memcpy(out_cells[0..cell_total], cells.items);

    const out_vectors: [*]i16 = @ptrCast(@alignCast(buf.ptr + vectors_off));
    const out_labels = buf[labels_off..][0..labels_len];
    for (order[0..count], 0..) |item_idx, j| {
        @memcpy(out_vectors[j * vector.stored_dims ..][0..vector.stored_dims], &items[item_idx].vec);
        if (items[item_idx].fraud) out_labels[j / 8] |= @as(u8, 1) << @intCast(j % 8);
    }

    return buf;
}

const Extent = struct { min: i32, range: i32 }; // range = max + 1 - min, >= 1

fn extent(items: []const Item, members: []const u32, dim: usize) Extent {
    var lo: i32 = std.math.maxInt(i32);
    var hi: i32 = std.math.minInt(i32);
    for (members) |idx| {
        const v: i32 = items[idx].vec[dim];
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    return .{ .min = lo, .range = hi + 1 - lo };
}

/// Map a quantized value to a cell index in [0, bins).
fn cellOf(value: i16, ext: Extent, bins: u16) u32 {
    const rel = @as(i64, value) - ext.min;
    const cell = @divFloor(rel * bins, ext.range);
    return @intCast(std.math.clamp(cell, 0, bins - 1));
}

/// Pick the two highest-variance stored dimensions, skipping the three bucket-bit
/// dimensions (constant within a bucket) so the grid splits on dimensions that vary.
fn pickGridDims(items: []const Item, members: []const u32) struct { usize, usize } {
    var best_a: usize = 0;
    var best_b: usize = 1;
    var var_a: f64 = -1;
    var var_b: f64 = -1;
    const n: f64 = @floatFromInt(members.len);
    for (0..vector.dims) |dim| {
        if (dim == 9 or dim == 10 or dim == 11) continue;
        var sum: i64 = 0;
        var sum_sq: i64 = 0;
        for (members) |idx| {
            const v: i64 = items[idx].vec[dim];
            sum += v;
            sum_sq += v * v;
        }
        const mean = @as(f64, @floatFromInt(sum)) / n;
        const variance = @as(f64, @floatFromInt(sum_sq)) / n - mean * mean;
        if (variance > var_a) {
            var_b = var_a;
            best_b = best_a;
            var_a = variance;
            best_a = dim;
        } else if (variance > var_b) {
            var_b = variance;
            best_b = dim;
        }
    }
    return .{ best_a, best_b };
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
