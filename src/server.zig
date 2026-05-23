const std = @import("std");
const c = std.c;
const zordon = @import("zordon");

const max_request_bytes = 64 * 1024;
const default_response =
    "HTTP/1.1 200 OK\r\n" ++
    "Content-Type: application/json\r\n" ++
    "Connection: close\r\n" ++
    "Content-Length: 37\r\n" ++
    "\r\n" ++
    "{\"approved\":true,\"fraud_score\":0.0}";

pub const Server = struct {
    classifier: *const zordon.classifier.Classifier,
    allocator: std.mem.Allocator,

    pub fn listen(self: *const Server, port: u16) !void {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        defer _ = c.close(fd);

        var yes: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, &yes, @sizeOf(c_int));

        var addr = std.os.linux.sockaddr.in{
            .family = c.AF.INET,
            .port = std.mem.nativeToBig(u16, port),
            .addr = 0,
            .zero = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
        };

        if (c.bind(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr))) != 0) return error.BindFailed;
        if (c.listen(fd, 1024) != 0) return error.ListenFailed;

        std.log.info("listening on 0.0.0.0:{d}", .{port});
        while (true) {
            const client = c.accept(fd, null, null);
            if (client < 0) continue;

            const thread = std.Thread.spawn(.{ .stack_size = 128 * 1024 }, handleClient, .{ self, client }) catch {
                _ = c.close(client);
                continue;
            };
            thread.detach();
        }
    }
};

const Request = struct {
    method: []const u8,
    path: []const u8,
    body: []const u8,
};

fn handleClient(server: *const Server, fd: c.fd_t) void {
    defer _ = c.close(fd);

    var buffer: [max_request_bytes]u8 = undefined;
    const request = readRequest(fd, &buffer) catch {
        sendAll(fd, "HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n") catch {};
        return;
    };

    if (std.mem.eql(u8, request.method, "GET") and std.mem.eql(u8, request.path, "/ready")) {
        sendText(fd, "200 OK", "text/plain", "ok\n") catch {};
        return;
    }

    if (std.mem.eql(u8, request.method, "POST") and std.mem.eql(u8, request.path, "/fraud-score")) {
        var parsed = zordon.payload.parse(server.allocator, request.body) catch {
            sendAll(fd, default_response) catch {};
            return;
        };
        defer parsed.deinit();

        const query = zordon.vector.fromRequest(parsed.value) catch {
            sendAll(fd, default_response) catch {};
            return;
        };
        const decision = server.classifier.decide(query);
        sendDecision(fd, decision) catch {};
        return;
    }

    sendText(fd, "404 Not Found", "text/plain", "not found\n") catch {};
}

fn readRequest(fd: c.fd_t, buffer: []u8) !Request {
    var len: usize = 0;
    var header_end: ?usize = null;
    var content_length: usize = 0;

    while (true) {
        if (len == buffer.len) return error.RequestTooLarge;
        const n = c.recv(fd, buffer.ptr + len, buffer.len - len, 0);
        if (n <= 0) return error.ConnectionClosed;
        len += @intCast(n);

        if (header_end == null) {
            if (std.mem.indexOf(u8, buffer[0..len], "\r\n\r\n")) |idx| {
                header_end = idx + 4;
                content_length = try parseContentLength(buffer[0..idx]);
            }
        }

        if (header_end) |end| {
            if (len >= end + content_length) {
                const head = buffer[0 .. end - 4];
                var lines = std.mem.splitSequence(u8, head, "\r\n");
                const request_line = lines.next() orelse return error.InvalidRequest;
                var parts = std.mem.splitScalar(u8, request_line, ' ');
                const method = parts.next() orelse return error.InvalidRequest;
                const path = parts.next() orelse return error.InvalidRequest;
                return .{
                    .method = method,
                    .path = path,
                    .body = buffer[end .. end + content_length],
                };
            }
        }
    }
}

fn parseContentLength(head: []const u8) !usize {
    var lines = std.mem.splitSequence(u8, head, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon];
        if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            return std.fmt.parseInt(usize, std.mem.trim(u8, line[colon + 1 ..], " \t"), 10);
        }
    }
    return 0;
}

fn sendDecision(fd: c.fd_t, decision: zordon.classifier.Decision) !void {
    var body_buf: [64]u8 = undefined;
    const body = try std.fmt.bufPrint(&body_buf, "{{\"approved\":{},\"fraud_score\":{d:.1}}}", .{
        decision.approved,
        decision.fraud_score,
    });
    return sendText(fd, "200 OK", "application/json", body);
}

fn sendText(fd: c.fd_t, status: []const u8, content_type: []const u8, body: []const u8) !void {
    var header_buf: [256]u8 = undefined;
    const header = try std.fmt.bufPrint(
        &header_buf,
        "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nConnection: close\r\nContent-Length: {d}\r\n\r\n",
        .{ status, content_type, body.len },
    );
    try sendAll(fd, header);
    try sendAll(fd, body);
}

fn sendAll(fd: c.fd_t, bytes: []const u8) !void {
    var sent: usize = 0;
    while (sent < bytes.len) {
        const n = c.send(fd, bytes.ptr + sent, bytes.len - sent, 0);
        if (n <= 0) return error.SendFailed;
        sent += @intCast(n);
    }
}
