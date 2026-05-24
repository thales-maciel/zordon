const std = @import("std");
const zordon = @import("zordon");
const server_mod = @import("server.zig");

const default_model_path = "data/model/references.i16.bin";

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const port = parsePort(init.environ_map.get("PORT") orelse "8080") catch 8080;
    const workers = parseWorkers(init.environ_map.get("ZORDON_WORKERS") orelse "1") catch 1;
    const model_path = init.environ_map.get("ZORDON_MODEL_PATH") orelse default_model_path;
    // When set, receive client fds from the load balancer over this Unix socket
    // (SCM_RIGHTS) instead of accepting TCP directly. fd mode runs a single worker.
    const unix_path = init.environ_map.get("ZORDON_UNIX_PATH");

    var classifier = loadClassifier(init.io, allocator, model_path) catch |err| blk: {
        std.log.warn("could not load model at '{s}' ({t}); using fallback development model", .{ model_path, err });
        break :blk try zordon.classifier.initFallback(allocator);
    };
    defer classifier.deinit(allocator);

    const server = server_mod.Server{
        .classifier = &classifier,
        .allocator = allocator,
        .unix_path = unix_path,
    };
    try server.listen(port, if (unix_path != null) 1 else workers);
}

fn loadClassifier(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !zordon.classifier.Classifier {
    return .{ .model = try zordon.model.load(io, allocator, path) };
}

fn parsePort(value: []const u8) !u16 {
    return std.fmt.parseInt(u16, value, 10);
}

fn parseWorkers(value: []const u8) !u16 {
    return std.fmt.parseInt(u16, value, 10);
}
