const std = @import("std");
const zordon = @import("zordon");

/// Microbenchmark for the index search (excludes HTTP + JSON parsing). Queries are
/// reference vectors sampled from the model itself, so the bucket distribution of
/// the workload matches the data. Reports per-query latency percentiles.
const dims = zordon.vector.stored_dims;

/// A query sampled from the model, optionally perturbed on continuous dimensions to
/// model an unseen transaction (the flag dims that pick the bucket stay fixed).
fn makeQuery(model: zordon.model.Model, rand: std.Random, perturb: i32) [dims]i16 {
    const idx = rand.uintLessThan(u32, model.count);
    var q = model.vectorAt(idx).*;
    if (perturb == 0) return q;
    const cont = [_]usize{ 0, 2, 3, 4, 7, 8, 12, 13 };
    for (cont) |d| {
        const noise = rand.intRangeAtMost(i32, -perturb, perturb);
        q[d] = @intCast(std.math.clamp(@as(i32, q[d]) + noise, 0, zordon.vector.scale));
    }
    return q;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    const path = if (args.len > 1) args[1] else "data/model/references.i16.bin";
    const iters: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 200_000;
    // perturb: add noise to a sampled reference vector so the query is "unseen"
    // (larger d_5², more grid cells visited) — models real test queries, not the
    // trivially-easy exact dataset members.
    const perturb: i32 = if (args.len > 3) try std.fmt.parseInt(i32, args[3], 10) else 0;

    var model = try zordon.model.load(io, allocator, path);
    defer model.deinit(allocator);
    const clf = zordon.classifier.Classifier{ .model = model };
    std.debug.print("model: {d} vectors, {d} cells\n", .{ model.count, model.cells.len });

    var prng = std.Random.DefaultPrng.init(0x9E3779B97F4A7C15);
    const rand = prng.random();

    const Clock = std.Io.Clock;
    var checksum: u64 = 0;
    for (0..10_000) |_| {
        checksum +%= clf.nearest(makeQuery(model, rand, perturb)).idx[0];
    }

    // Throughput: one timestamp pair around the whole loop (no per-query overhead).
    const start = Clock.awake.now(io);
    for (0..iters) |_| {
        checksum +%= clf.nearest(makeQuery(model, rand, perturb)).idx[0];
    }
    const elapsed = Clock.awake.now(io);
    const total_ns: f64 = @floatFromInt(start.durationTo(elapsed).nanoseconds);
    const mean_us = total_ns / @as(f64, @floatFromInt(iters)) / 1000.0;

    // Percentiles: per-query timing (slightly inflated by the clock calls themselves).
    const times = try allocator.alloc(u64, iters);
    defer allocator.free(times);
    const scans = try allocator.alloc(u64, iters);
    defer allocator.free(scans);
    for (times, scans) |*slot, *sc| {
        const q = makeQuery(model, rand, perturb);
        const a = Clock.awake.now(io);
        const top = clf.nearest(q);
        const b = Clock.awake.now(io);
        slot.* = @intCast(a.durationTo(b).nanoseconds);
        sc.* = top.scanned;
        checksum +%= top.idx[0];
    }
    std.mem.sort(u64, times, {}, std.sort.asc(u64));
    std.mem.sort(u64, scans, {}, std.sort.asc(u64));
    std.debug.print("  candidates scanned: p50={d} p99={d} max={d}\n", .{
        scans[iters / 2], scans[iters * 99 / 100], scans[iters - 1],
    });

    const p = struct {
        fn at(ts: []const u64, q: f64) f64 {
            const i: usize = @intFromFloat(q * @as(f64, @floatFromInt(ts.len - 1)));
            return @as(f64, @floatFromInt(ts[i])) / 1000.0;
        }
    };
    std.debug.print(
        \\queries={d}  checksum={x}
        \\  mean={d:.3}ms  p50={d:.3}ms  p90={d:.3}ms  p99={d:.3}ms  p999={d:.3}ms  max={d:.3}ms
        \\  throughput (single core) ~ {d:.0} q/s
        \\
    , .{
        iters,                          checksum,
        mean_us / 1000.0,               p.at(times, 0.50) / 1000.0,
        p.at(times, 0.90) / 1000.0,     p.at(times, 0.99) / 1000.0,
        p.at(times, 0.999) / 1000.0,    @as(f64, @floatFromInt(times[iters - 1])) / 1_000_000.0,
        1_000_000.0 / mean_us,
    });
}
