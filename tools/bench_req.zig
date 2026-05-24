const std = @import("std");
const zordon = @import("zordon");

/// Per-request cost breakdown over realistic payloads (the contest example set),
/// isolating JSON parse vs vectorize vs search. The plain index microbench uses
/// dataset members (tiny d_5²); real unseen queries are heavier, so this measures
/// the actual server-side work per request.
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const model_path = if (args.len > 1) args[1] else "data/model/references.i16.bin";
    const payload_path = if (args.len > 2) args[2] else "rinha-de-backend-2026/resources/example-payloads.json";
    const iters: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 500_000;

    var model = try zordon.model.load(io, allocator, model_path);
    defer model.deinit(allocator);
    const clf = zordon.classifier.Classifier{ .model = model };

    const raw = try std.Io.Dir.cwd().readFileAlloc(io, payload_path, allocator, .limited(10_000_000));
    defer allocator.free(raw);
    const payloads = try splitObjects(allocator, raw);
    defer allocator.free(payloads);
    std.debug.print("model: {d} vectors; corpus: {d} payloads; iters: {d}\n", .{ model.count, payloads.len, iters });

    const Clock = std.Io.Clock;
    var sink: u64 = 0;

    // full path: parse + vectorize + decide
    const t_full = bench(io, iters, payloads, &sink, struct {
        fn run(clf_: zordon.classifier.Classifier, body: []const u8, s: *u64) void {
            var scratch: [32 * 1024]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&scratch);
            const parsed = zordon.payload.parse(fba.allocator(), body) catch return;
            const q = zordon.vector.fromRequest(parsed.value) catch return;
            s.* +%= clf_.decide(q).fraud_neighbors;
        }
    }.run, clf);

    // parse + vectorize only
    const t_vec = bench(io, iters, payloads, &sink, struct {
        fn run(_: zordon.classifier.Classifier, body: []const u8, s: *u64) void {
            var scratch: [32 * 1024]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&scratch);
            const parsed = zordon.payload.parse(fba.allocator(), body) catch return;
            const q = zordon.vector.fromRequest(parsed.value) catch return;
            s.* +%= @intFromBool(q[0] > 0);
        }
    }.run, clf);

    // parse only
    const t_parse = bench(io, iters, payloads, &sink, struct {
        fn run(_: zordon.classifier.Classifier, body: []const u8, s: *u64) void {
            var scratch: [32 * 1024]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&scratch);
            const parsed = zordon.payload.parse(fba.allocator(), body) catch return;
            s.* +%= parsed.value.transaction.installments;
        }
    }.run, clf);

    _ = Clock;
    const per = struct {
        fn us(t: f64, n: usize) f64 {
            return t / @as(f64, @floatFromInt(n)) / 1000.0;
        }
    };
    std.debug.print(
        \\sink={x}
        \\  parse only            : {d:.3} us/req
        \\  parse + vectorize     : {d:.3} us/req   (vectorize = {d:.3})
        \\  parse + vec + search  : {d:.3} us/req   (search = {d:.3})
        \\  => full request ~ {d:.3} us  -> sustainable ~ {d:.0} req/s per 0.475 CPU core
        \\
    , .{
        sink,
        per.us(t_parse, iters),
        per.us(t_vec, iters),   per.us(t_vec - t_parse, iters),
        per.us(t_full, iters),  per.us(t_full - t_vec, iters),
        per.us(t_full, iters),  0.475 * 1_000_000.0 / per.us(t_full, iters),
    });
}

fn bench(
    io: std.Io,
    iters: usize,
    payloads: []const []const u8,
    sink: *u64,
    comptime run: fn (zordon.classifier.Classifier, []const u8, *u64) void,
    clf: zordon.classifier.Classifier,
) f64 {
    // warmup
    for (0..@min(payloads.len * 100, iters)) |i| run(clf, payloads[i % payloads.len], sink);
    const start = std.Io.Clock.awake.now(io);
    for (0..iters) |i| run(clf, payloads[i % payloads.len], sink);
    const end = std.Io.Clock.awake.now(io);
    return @floatFromInt(start.durationTo(end).nanoseconds);
}

fn splitObjects(allocator: std.mem.Allocator, input: []const u8) ![][]const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var depth: i32 = 0;
    var start: usize = 0;
    var in_str = false;
    var esc = false;
    for (input, 0..) |c, i| {
        if (in_str) {
            if (esc) esc = false else if (c == '\\') esc = true else if (c == '"') in_str = false;
            continue;
        }
        switch (c) {
            '"' => in_str = true,
            '{' => {
                if (depth == 0) start = i;
                depth += 1;
            },
            '}' => {
                depth -= 1;
                if (depth == 0) try list.append(allocator, input[start .. i + 1]);
            },
            else => {},
        }
    }
    return list.toOwnedSlice(allocator);
}
