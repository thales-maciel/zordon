const std = @import("std");
const zordon = @import("zordon");

const vector = zordon.vector;
const model = zordon.model;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    if (args.len < 3 or args.len > 4) {
        std.debug.print("usage: zig build preprocess -- <references.json> <references.bin> [bins]\n", .{});
        return error.InvalidArguments;
    }

    const input_path = args[1];
    const output_path = args[2];
    const bins: u16 = if (args.len == 4) try std.fmt.parseInt(u16, args[3], 10) else model.default_bins;

    const input = try std.Io.Dir.cwd().readFileAlloc(io, input_path, allocator, .limited(1_000_000_000));
    defer allocator.free(input);

    const count = countReferences(input);
    if (count == 0) return error.EmptyReferences;

    const items = try allocator.alloc(model.Item, count);
    defer allocator.free(items);
    try parseReferences(input, items);

    const bytes = try model.build(allocator, items, bins);
    defer allocator.free(bytes);

    var out_file = try std.Io.Dir.cwd().createFile(io, output_path, .{});
    defer out_file.close(io);
    var write_buffer: [1 << 20]u8 = undefined;
    var writer = out_file.writer(io, &write_buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();

    std.debug.print("wrote {d} references ({d} bytes, {d} bins) to {s}\n", .{ count, bytes.len, bins, output_path });
}

fn countReferences(input: []const u8) usize {
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, input, pos, "\"vector\"")) |idx| {
        count += 1;
        pos = idx + 8;
    }
    return count;
}

fn parseReferences(input: []const u8, items: []model.Item) !void {
    var pos: usize = 0;
    var row: usize = 0;
    while (std.mem.indexOfPos(u8, input, pos, "\"vector\"")) |vector_key| {
        if (row >= items.len) return error.TooManyRows;
        var vec: [vector.stored_dims]i16 = @splat(0);

        const open = std.mem.indexOfScalarPos(u8, input, vector_key, '[') orelse return error.InvalidReferences;
        var cursor = open + 1;
        for (0..vector.dims) |dim| {
            skipWhitespace(input, &cursor);
            const value_start = cursor;
            while (cursor < input.len and input[cursor] != ',' and input[cursor] != ']') : (cursor += 1) {}
            if (cursor >= input.len) return error.InvalidReferences;
            const token = std.mem.trim(u8, input[value_start..cursor], " \n\r\t");
            const value = try std.fmt.parseFloat(f64, token);
            vec[dim] = vector.quantizeValue(value);
            if (dim + 1 < vector.dims) {
                if (input[cursor] != ',') return error.InvalidReferences;
                cursor += 1;
            }
        }
        if (input[cursor] != ']') return error.InvalidReferences;

        const label_key = std.mem.indexOfPos(u8, input, cursor, "\"label\"") orelse return error.InvalidReferences;
        const label_colon = std.mem.indexOfScalarPos(u8, input, label_key, ':') orelse return error.InvalidReferences;
        const quote_a = std.mem.indexOfScalarPos(u8, input, label_colon, '"') orelse return error.InvalidReferences;
        const quote_b = std.mem.indexOfScalarPos(u8, input, quote_a + 1, '"') orelse return error.InvalidReferences;
        const label = input[quote_a + 1 .. quote_b];
        const fraud = if (std.mem.eql(u8, label, "fraud"))
            true
        else if (std.mem.eql(u8, label, "legit"))
            false
        else
            return error.InvalidReferences;

        items[row] = .{ .vec = vec, .fraud = fraud };
        row += 1;
        pos = quote_b + 1;
    }
    if (row != items.len) return error.InvalidReferences;
}

fn skipWhitespace(input: []const u8, cursor: *usize) void {
    while (cursor.* < input.len) : (cursor.* += 1) {
        switch (input[cursor.*]) {
            ' ', '\n', '\r', '\t' => {},
            else => return,
        }
    }
}
