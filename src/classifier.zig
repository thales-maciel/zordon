const std = @import("std");
const model_mod = @import("model.zig");
const vector = @import("vector.zig");

const dims = vector.stored_dims;
const scale: i64 = vector.scale;
/// Lower bound contributed by a single differing binary flag: (scale - 0)^2.
const flag_penalty: i64 = scale * scale;

pub const Decision = struct {
    approved: bool,
    fraud_score: f32,
    fraud_neighbors: u8,
    neighbors: u8,
};

pub const Classifier = struct {
    model: model_mod.Model,

    pub fn deinit(self: *Classifier, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
        self.* = undefined;
    }

    pub fn decide(self: Classifier, query: vector.Vector) Decision {
        const q = vector.quantize(query);
        const top = self.nearest(q);
        const k = top.k;
        if (k == 0) {
            return .{ .approved = true, .fraud_score = 0.0, .fraud_neighbors = 0, .neighbors = 0 };
        }
        var frauds: u8 = 0;
        for (top.idx[0..k]) |index| {
            if (self.model.labelIsFraud(index)) frauds += 1;
        }
        const score = @as(f32, @floatFromInt(frauds)) / @as(f32, @floatFromInt(k));
        return .{
            .approved = score < 0.6,
            .fraud_score = score,
            .fraud_neighbors = frauds,
            .neighbors = @intCast(k),
        };
    }

    /// Exact k-nearest search (k <= 5) over the bucketed index.
    pub fn nearest(self: Classifier, q: vector.QuantizedVector) TopK {
        const k: usize = @min(5, self.model.count);
        var top = TopK{ .k = k };
        if (k == 0) return top;

        const qv: @Vector(dims, i32) = @intCast(@as(@Vector(dims, i16), q));
        const own = vector.bucketKey(q);

        // Stage 1: search the query's own bucket first, then any other bucket whose
        // 4-bit lower bound is still closer than our current 5th-nearest distance.
        // The branch-and-bound makes this exact; in practice no other bucket qualifies
        // (a differing flag costs scale^2, far beyond any real neighbour distance).
        self.scanBucket(own, qv, &top);
        for (0..model_mod.bucket_count) |kk| {
            if (kk == own) continue;
            if (bucketLowerBound(q, @intCast(kk)) < top.worst()) {
                self.scanBucket(@intCast(kk), qv, &top);
            }
        }
        return top;
    }

    fn scanBucket(self: Classifier, key: u4, qv: @Vector(dims, i32), top: *TopK) void {
        const bucket = self.model.buckets[key];
        const vectors = self.model.vectors;
        var i: u32 = bucket.vec_start;
        const end = bucket.vec_start + bucket.vec_count;
        while (i < end) : (i += 1) {
            const dist = distanceSq(qv, vectors, i);
            top.offer(dist, i);
        }
    }
};

/// Squared Euclidean distance over the 16 stored lanes, widened to i64 so the
/// 16-lane sum (up to ~6.4e9) cannot overflow.
fn distanceSq(qv: @Vector(dims, i32), vectors: []const i16, index: u32) i64 {
    const lanes: [dims]i16 = vectors[@as(usize, index) * dims ..][0..dims].*;
    const cv: @Vector(dims, i32) = @intCast(@as(@Vector(dims, i16), lanes));
    const d = qv - cv;
    const sq_vec: @Vector(dims, i32) = d * d;
    const sq64: @Vector(dims, i64) = @intCast(sq_vec);
    return @reduce(.Add, sq64);
}

/// Exact lower bound on the squared distance from query `q` to any vector in
/// bucket `key`, using only the four bucket-defining dimensions. The three binary
/// flags are exact (each differing flag adds scale^2); the history bit is treated
/// conservatively (adds at least scale^2 when it differs).
fn bucketLowerBound(q: vector.QuantizedVector, key: u4) i64 {
    var lb: i64 = 0;
    const t_online: i64 = if ((key >> 3) & 1 == 1) scale else 0;
    const t_card: i64 = if ((key >> 2) & 1 == 1) scale else 0;
    const t_unk: i64 = if ((key >> 1) & 1 == 1) scale else 0;
    lb += sq(@as(i64, q[9]) - t_online);
    lb += sq(@as(i64, q[10]) - t_card);
    lb += sq(@as(i64, q[11]) - t_unk);
    const q_hist: u4 = @intFromBool(q[5] > -(vector.scale >> 1));
    if (q_hist != (key & 1)) lb += flag_penalty;
    return lb;
}

inline fn sq(x: i64) i64 {
    return x * x;
}

/// Fixed-size top-K (K <= 5) nearest tracker by ascending squared distance.
const TopK = struct {
    dist: [5]i64 = .{std.math.maxInt(i64)} ** 5,
    idx: [5]u32 = .{0} ** 5,
    k: usize,

    fn worst(self: *const TopK) i64 {
        var w = self.dist[0];
        for (self.dist[1..self.k]) |d| w = @max(w, d);
        return w;
    }

    fn offer(self: *TopK, dist: i64, index: u32) void {
        var worst_slot: usize = 0;
        var worst_dist = self.dist[0];
        for (self.dist[1..self.k], 1..) |d, slot| {
            if (d > worst_dist) {
                worst_dist = d;
                worst_slot = slot;
            }
        }
        if (dist < worst_dist) {
            self.dist[worst_slot] = dist;
            self.idx[worst_slot] = index;
        }
    }
};

pub fn initFallback(allocator: std.mem.Allocator) !Classifier {
    const samples = [_]struct { v: vector.Vector, fraud: bool }{
        .{ .v = .{ 0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1.0, -1.0, 0.0292, 0.15, 0, 1, 0, 0.15, 0.0060 }, .fraud = false },
        .{ .v = .{ 0.0109, 0.1667, 0.05, 0.3913, 0.6667, 0.3007, 0.0139, 0.0154, 0.20, 0, 1, 0, 0.15, 0.0282 }, .fraud = false },
        .{ .v = .{ 0.0336, 0.1667, 0.05, 0.4348, 0.6667, 0.1278, 0.0008, 0.0170, 0.10, 0, 1, 0, 0.20, 0.0200 }, .fraud = false },
        .{ .v = .{ 0.9506, 0.8333, 1.00, 0.2174, 0.8333, -1.0, -1.0, 0.9523, 1.00, 0, 1, 1, 0.75, 0.0055 }, .fraud = true },
        .{ .v = .{ 0.5796, 0.9167, 1.00, 0.0435, 0.0000, 0.0056, 0.4394, 0.4598, 0.40, 1, 0, 1, 0.85, 0.0032 }, .fraud = true },
        .{ .v = .{ 0.8000, 1.0000, 1.00, 0.1300, 0.5000, 0.0100, 0.8000, 0.8500, 0.90, 1, 0, 1, 0.80, 0.0100 }, .fraud = true },
    };
    var items: [samples.len]model_mod.Item = undefined;
    for (samples, 0..) |s, i| {
        items[i] = .{ .vec = vector.quantize(s.v), .fraud = s.fraud };
    }
    const bytes = try model_mod.build(allocator, &items, 8);
    return .{ .model = try model_mod.parse(bytes) };
}

test "bucketed search matches brute force over random data" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();

    const n = 500;
    const items = try allocator.alloc(model_mod.Item, n);
    defer allocator.free(items);
    for (items) |*it| {
        var v: vector.Vector = undefined;
        for (0..vector.dims) |d| v[d] = rand.float(f32);
        v[9] = if (rand.boolean()) 1.0 else 0.0;
        v[10] = if (rand.boolean()) 1.0 else 0.0;
        v[11] = if (rand.boolean()) 1.0 else 0.0;
        if (rand.boolean()) {
            v[5] = -1.0;
            v[6] = -1.0;
        }
        it.* = .{ .vec = vector.quantize(v), .fraud = rand.boolean() };
    }

    const bytes = try model_mod.build(allocator, items, 8);
    var model = try model_mod.parse(bytes);
    defer model.deinit(allocator);
    const clf = Classifier{ .model = model };

    for (0..200) |_| {
        var v: vector.Vector = undefined;
        for (0..vector.dims) |d| v[d] = rand.float(f32);
        v[9] = if (rand.boolean()) 1.0 else 0.0;
        v[10] = if (rand.boolean()) 1.0 else 0.0;
        v[11] = if (rand.boolean()) 1.0 else 0.0;
        if (rand.boolean()) {
            v[5] = -1.0;
            v[6] = -1.0;
        }
        const q = vector.quantize(v);
        const qv: @Vector(dims, i32) = @intCast(@as(@Vector(dims, i16), q));

        // brute force: the five smallest squared distances over all vectors
        var brute = TopK{ .k = 5 };
        for (0..model.count) |i| brute.offer(distanceSq(qv, model.vectors, @intCast(i)), @intCast(i));

        const got = clf.nearest(q);
        var bd = brute.dist;
        var gd = got.dist;
        std.mem.sort(i64, &bd, {}, std.sort.asc(i64));
        std.mem.sort(i64, &gd, {}, std.sort.asc(i64));
        try std.testing.expectEqualSlices(i64, &bd, &gd);
    }
}

test "fallback classifier separates obvious high-risk payload" {
    var clf = try initFallback(std.testing.allocator);
    defer clf.deinit(std.testing.allocator);

    const legit = vector.Vector{ 0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1.0, -1.0, 0.0292, 0.15, 0, 1, 0, 0.15, 0.0060 };
    try std.testing.expect(clf.decide(legit).approved);

    const fraud = vector.Vector{ 0.9506, 0.8333, 1.00, 0.2174, 0.8333, -1.0, -1.0, 0.9523, 1.00, 0, 1, 1, 0.75, 0.0055 };
    try std.testing.expect(!clf.decide(fraud).approved);
}
