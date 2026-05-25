const std = @import("std");
const zordon = @import("zordon");

/// Offline detection + scan-cost sweep over the full preview/test query set
/// (test-data.json — identical to what the contest engine replays). For each
/// candidate budget it reports the exact detection error (FP/FN, hardware
/// independent) and the distribution of candidates scanned per query (the proxy
/// for per-query CPU cost). This is the fast loop: the only thing it cannot
/// measure is the absolute Mac-Mini p99 (CFS throttling + a ~5x slower core),
/// which a single preview test confirms.
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const model_path = if (args.len > 1) args[1] else "data/model/references.i16.bin";
    const data_path = if (args.len > 2) args[2] else "rinha-de-backend-2026/test/test-data.json";

    var model = try zordon.model.load(io, allocator, model_path);
    defer model.deinit(allocator);
    const clf = zordon.classifier.Classifier{ .model = model };

    const raw = try std.Io.Dir.cwd().readFileAlloc(io, data_path, allocator, .limited(400_000_000));
    defer allocator.free(raw);
    const entries = try extractEntries(allocator, raw);
    defer allocator.free(entries);

    // Pre-vectorize once so the timed sweep measures search only.
    const queries = try allocator.alloc(zordon.vector.QuantizedVector, entries.len);
    defer allocator.free(queries);
    const expect_approved = try allocator.alloc(bool, entries.len);
    defer allocator.free(expect_approved);
    var valid: usize = 0;
    var parse_err: usize = 0;
    for (entries) |e| {
        var scratch: [32 * 1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&scratch);
        const parsed = zordon.payload.parse(fba.allocator(), e.body) catch {
            parse_err += 1;
            continue;
        };
        const v = zordon.vector.fromRequest(parsed.value) catch {
            parse_err += 1;
            continue;
        };
        queries[valid] = zordon.vector.quantize(v);
        expect_approved[valid] = e.approved;
        valid += 1;
    }
    std.debug.print("model: {d} vectors | entries: {d} | valid queries: {d} | parse errors: {d}\n", .{ model.count, entries.len, valid, parse_err });

    const caps = [_]u32{ 200_000, 100_000, 50_000, 30_000, 20_000, 12_000, 8_000, 5_000, 3_000, 2_000, 1_000 };
    const scanned = try allocator.alloc(u32, valid);
    defer allocator.free(scanned);

    std.debug.print("\n{s:>8} {s:>4} {s:>4} {s:>8}  {s:>8} {s:>8} {s:>8} {s:>8} {s:>9}  {s:>9}\n", .{
        "cap", "FP", "FN", "det_sc", "scan_p50", "scan_p90", "scan_p99", "scn_p999", "scan_max", "us/query",
    });
    for (caps) |cap| {
        var fp: usize = 0;
        var fn_count: usize = 0;
        const start = std.Io.Clock.awake.now(io);
        for (0..valid) |i| {
            const top = clf.nearestCap(queries[i], cap);
            var frauds: u32 = 0;
            for (top.idx[0..top.k]) |idx| {
                if (model.labelIsFraud(idx)) frauds += 1;
            }
            const score = @as(f32, @floatFromInt(frauds)) / @as(f32, @floatFromInt(top.k));
            const approved = score < 0.6;
            scanned[i] = top.scanned;
            if (approved != expect_approved[i]) {
                if (expect_approved[i]) fp += 1 else fn_count += 1; // legit declined / fraud approved
            }
        }
        const end = std.Io.Clock.awake.now(io);
        const us_per = @as(f64, @floatFromInt(start.durationTo(end).nanoseconds)) / @as(f64, @floatFromInt(valid)) / 1000.0;

        std.mem.sort(u32, scanned, {}, std.sort.asc(u32));
        // Detection score with the contest formula (offline Err = 0): E = FP + 3*FN.
        const e_weighted: f64 = @floatFromInt(fp + 3 * fn_count);
        const eps = e_weighted / @as(f64, @floatFromInt(valid));
        const rate = 1000.0 * std.math.log10(1.0 / @max(eps, 0.001));
        const penalty = 300.0 * std.math.log10(1.0 + e_weighted);
        const det = rate - penalty;

        std.debug.print("{d:>8} {d:>4} {d:>4} {d:>8.1}  {d:>8} {d:>8} {d:>8} {d:>8} {d:>9}  {d:>9.2}\n", .{
            cap, fp, fn_count, det,
            pct(scanned, 50), pct(scanned, 90), pct(scanned, 99), pct(scanned, 999), scanned[valid - 1],
            us_per,
        });
    }
}

fn pct(sorted: []const u32, permille: usize) u32 {
    // permille uses 0..1000 so we can express p99.9 as 999.
    const idx = (sorted.len * permille) / 1000;
    return sorted[@min(idx, sorted.len - 1)];
}

const Entry = struct { body: []const u8, approved: bool };

/// Extract each entry's `request` object (as raw JSON bytes) and its
/// `expected_approved` flag, by brace-matching — avoids depending on a particular
/// std.json API and reuses the exact production parse path on the body.
fn extractEntries(allocator: std.mem.Allocator, raw: []const u8) ![]Entry {
    var list: std.ArrayList(Entry) = .empty;
    const req_key = "\"request\":";
    const app_key = "\"expected_approved\":";
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, raw, i, req_key)) |ri| {
        var j = ri + req_key.len;
        while (j < raw.len and raw[j] != '{') j += 1;
        const body_start = j;
        var depth: i32 = 0;
        var in_str = false;
        var esc = false;
        while (j < raw.len) : (j += 1) {
            const c = raw[j];
            if (in_str) {
                if (esc) esc = false else if (c == '\\') esc = true else if (c == '"') in_str = false;
                continue;
            }
            switch (c) {
                '"' => in_str = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if (depth == 0) {
                        j += 1;
                        break;
                    }
                },
                else => {},
            }
        }
        const body = raw[body_start..j];
        const ai = std.mem.indexOfPos(u8, raw, j, app_key) orelse return error.MissingApproved;
        var k = ai + app_key.len;
        while (k < raw.len and (raw[k] == ' ' or raw[k] == '\t' or raw[k] == '\n' or raw[k] == '\r')) k += 1;
        try list.append(allocator, .{ .body = body, .approved = raw[k] == 't' });
        i = k;
    }
    return list.toOwnedSlice(allocator);
}
