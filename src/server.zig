const std = @import("std");
const zordon = @import("zordon");
const net = zordon.net;
const linux = net.linux;
const EPOLL = net.EPOLL;
const SOCK = net.SOCK;
const MSG = net.MSG;

const max_events = 256;
/// Per-connection read buffer. A request is request-line + headers + a ~400 byte
/// JSON body, comfortably under 8 KiB; anything larger is treated as malformed.
const conn_buf_size = 8 * 1024;
/// Connection slots per worker. The LB keeps ~connections-worth alive; this leaves
/// generous headroom.
const max_conns = 1024;
const max_workers = 16;

/// Scratch for parsing one JSON payload. Stack-local per request, so worker threads
/// never contend on a shared allocator.
const json_scratch = 32 * 1024;

const default_body = "{\"approved\":true,\"fraud_score\":0.0}";

// epoll_event.data.ptr sentinels (real *Conn pointers are heap addresses > 1).
const tag_listener: usize = 0;
const tag_control: usize = 1;

pub const Server = struct {
    classifier: *const zordon.classifier.Classifier,
    allocator: std.mem.Allocator,
    /// When set, the server receives client fds from the load balancer over this
    /// Unix socket (SCM_RIGHTS) instead of accepting TCP connections directly.
    unix_path: ?[]const u8 = null,

    pub fn listen(self: *const Server, port: u16, workers: u16) !void {
        const n = std.math.clamp(workers, 1, max_workers);
        if (self.unix_path) |p| {
            std.log.info("receiving fds on {s} with {d} worker(s)", .{ p, n });
        } else {
            std.log.info("listening on 0.0.0.0:{d} with {d} worker(s)", .{ port, n });
        }

        var threads: [max_workers]?std.Thread = .{null} ** max_workers;
        var i: u16 = 1;
        while (i < n) : (i += 1) {
            threads[i] = std.Thread.spawn(.{ .stack_size = 1 << 20 }, runWorker, .{ self, port }) catch null;
        }
        // Run one worker on the calling thread; it never returns under normal operation.
        runWorker(self, port);
        for (threads[1..n]) |t| if (t) |thread| thread.join();
    }
};

fn runWorker(self: *const Server, port: u16) void {
    var worker = Worker.init(self, port) catch |err| {
        std.log.err("worker init failed: {t}", .{err});
        return;
    };
    worker.run();
}

const Conn = struct {
    fd: i32,
    len: usize,
    buf: [conn_buf_size]u8,
};

const Worker = struct {
    server: *const Server,
    /// TCP listener (direct mode) or Unix listener that the LB connects to (fd mode).
    listen_fd: i32,
    /// In fd mode, the single accepted LB control connection delivering passed fds.
    control_fd: i32,
    unix_mode: bool,
    epfd: i32,
    conns: []Conn,
    free: []u32,
    free_top: usize,

    fn init(server: *const Server, port: u16) !Worker {
        const unix_mode = server.unix_path != null;
        const listen_fd = if (server.unix_path) |p|
            try net.createUnixListener(p)
        else
            try net.createTcpListener(port);
        errdefer _ = linux.close(listen_fd);

        const eprc = linux.epoll_create1(EPOLL.CLOEXEC);
        if (net.sysret(eprc) != .SUCCESS) return error.EpollCreateFailed;
        const epfd: i32 = @intCast(eprc);
        errdefer _ = linux.close(epfd);

        var ev = linux.epoll_event{ .events = EPOLL.IN, .data = .{ .ptr = tag_listener } };
        if (net.sysret(linux.epoll_ctl(epfd, EPOLL.CTL_ADD, listen_fd, &ev)) != .SUCCESS) {
            return error.EpollAddFailed;
        }

        const allocator = server.allocator;
        const conns = try allocator.alloc(Conn, max_conns);
        errdefer allocator.free(conns);
        const free = try allocator.alloc(u32, max_conns);
        errdefer allocator.free(free);
        for (free, 0..) |*slot, idx| slot.* = @intCast(max_conns - 1 - idx);

        return .{
            .server = server,
            .listen_fd = listen_fd,
            .control_fd = -1,
            .unix_mode = unix_mode,
            .epfd = epfd,
            .conns = conns,
            .free = free,
            .free_top = max_conns,
        };
    }

    fn run(self: *Worker) void {
        var events: [max_events]linux.epoll_event = undefined;
        while (true) {
            const rc = linux.epoll_wait(self.epfd, &events, max_events, -1);
            switch (net.sysret(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                else => return,
            }
            for (events[0..rc]) |ev| switch (ev.data.ptr) {
                tag_listener => if (self.unix_mode) self.acceptControl() else self.acceptClients(),
                tag_control => self.onControl(),
                else => self.onReadable(@ptrFromInt(ev.data.ptr)),
            };
        }
    }

    /// Direct mode: accept TCP clients on the listener.
    fn acceptClients(self: *Worker) void {
        while (true) {
            const rc = linux.accept4(self.listen_fd, null, null, SOCK.NONBLOCK | SOCK.CLOEXEC);
            switch (net.sysret(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                .AGAIN => return,
                else => return,
            }
            const fd: i32 = @intCast(rc);
            net.setNoDelay(fd);
            self.registerClient(fd);
        }
    }

    /// fd mode: accept the load balancer's Unix control connection.
    fn acceptControl(self: *Worker) void {
        while (true) {
            const rc = linux.accept4(self.listen_fd, null, null, SOCK.NONBLOCK | SOCK.CLOEXEC);
            switch (net.sysret(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                .AGAIN => return,
                else => return,
            }
            const fd: i32 = @intCast(rc);
            if (self.control_fd >= 0) _ = linux.close(self.control_fd); // replace a stale LB conn
            self.control_fd = fd;
            var ev = linux.epoll_event{ .events = EPOLL.IN, .data = .{ .ptr = tag_control } };
            if (net.sysret(linux.epoll_ctl(self.epfd, EPOLL.CTL_ADD, fd, &ev)) != .SUCCESS) {
                _ = linux.close(fd);
                self.control_fd = -1;
            }
        }
    }

    /// fd mode: drain passed client fds from the LB and register them as connections.
    fn onControl(self: *Worker) void {
        if (self.control_fd < 0) return;
        while (true) {
            switch (net.recvFd(self.control_fd)) {
                .fd => |client_fd| {
                    net.setNoDelay(client_fd);
                    self.registerClient(client_fd);
                },
                .again => return,
                .closed => {
                    _ = linux.close(self.control_fd);
                    self.control_fd = -1;
                    return;
                },
            }
        }
    }

    fn registerClient(self: *Worker, fd: i32) void {
        const conn = self.acquire() orelse {
            _ = linux.close(fd);
            return;
        };
        conn.fd = fd;
        conn.len = 0;
        var ev = linux.epoll_event{ .events = EPOLL.IN, .data = .{ .ptr = @intFromPtr(conn) } };
        if (net.sysret(linux.epoll_ctl(self.epfd, EPOLL.CTL_ADD, fd, &ev)) != .SUCCESS) {
            _ = linux.close(fd);
            self.release(conn);
        }
    }

    fn onReadable(self: *Worker, conn: *Conn) void {
        var peer_closed = false;
        drain: while (true) {
            if (conn.len == conn_buf_size) {
                // No complete request fit in the buffer: malformed/oversized.
                self.closeConn(conn);
                return;
            }
            const rc = linux.recvfrom(conn.fd, conn.buf[conn.len..].ptr, conn_buf_size - conn.len, 0, null, null);
            switch (net.sysret(rc)) {
                .SUCCESS => {
                    if (rc == 0) {
                        peer_closed = true;
                        break :drain;
                    }
                    conn.len += rc;
                },
                .INTR => continue :drain,
                .AGAIN => break :drain,
                else => {
                    self.closeConn(conn);
                    return;
                },
            }
        }

        self.processBuffer(conn);
        if (peer_closed) self.closeConn(conn);
    }

    fn processBuffer(self: *Worker, conn: *Conn) void {
        var off: usize = 0;
        while (parseRequest(conn.buf[off..conn.len])) |req| {
            self.handleRequest(conn.fd, req);
            off += req.consumed;
        }
        if (off > 0) {
            const remaining = conn.len - off;
            std.mem.copyForwards(u8, conn.buf[0..remaining], conn.buf[off..conn.len]);
            conn.len = remaining;
        }
    }

    fn handleRequest(self: *Worker, fd: i32, req: Request) void {
        if (std.mem.eql(u8, req.method, "GET") and std.mem.eql(u8, req.path, "/ready")) {
            send(fd, "200 OK", "text/plain", "ok\n");
            return;
        }
        if (std.mem.eql(u8, req.method, "POST") and std.mem.eql(u8, req.path, "/fraud-score")) {
            self.handleScore(fd, req.body);
            return;
        }
        send(fd, "404 Not Found", "text/plain", "not found\n");
    }

    fn handleScore(self: *Worker, fd: i32, body: []const u8) void {
        var scratch: [json_scratch]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&scratch);
        const parsed = zordon.payload.parse(fba.allocator(), body) catch {
            send(fd, "200 OK", "application/json", default_body);
            return;
        };
        const query = zordon.vector.fromRequest(parsed.value) catch {
            send(fd, "200 OK", "application/json", default_body);
            return;
        };
        const decision = self.server.classifier.decide(query);
        var body_buf: [64]u8 = undefined;
        const out = std.fmt.bufPrint(&body_buf, "{{\"approved\":{},\"fraud_score\":{d:.1}}}", .{
            decision.approved,
            decision.fraud_score,
        }) catch default_body;
        send(fd, "200 OK", "application/json", out);
    }

    fn acquire(self: *Worker) ?*Conn {
        if (self.free_top == 0) return null;
        self.free_top -= 1;
        return &self.conns[self.free[self.free_top]];
    }

    fn release(self: *Worker, conn: *Conn) void {
        const idx: u32 = @intCast((@intFromPtr(conn) - @intFromPtr(self.conns.ptr)) / @sizeOf(Conn));
        self.free[self.free_top] = idx;
        self.free_top += 1;
    }

    fn closeConn(self: *Worker, conn: *Conn) void {
        // Closing the fd removes it from the epoll set.
        _ = linux.close(conn.fd);
        self.release(conn);
    }
};

const Request = struct {
    method: []const u8,
    path: []const u8,
    body: []const u8,
    consumed: usize,
};

/// Parse one HTTP request from the front of `buf`. Returns null if the buffer does
/// not yet hold a complete request (headers + Content-Length bytes of body).
fn parseRequest(buf: []const u8) ?Request {
    const header_end = std.mem.indexOf(u8, buf, "\r\n\r\n") orelse return null;
    const body_start = header_end + 4;
    const content_length = parseContentLength(buf[0..header_end]);
    if (buf.len < body_start + content_length) return null;

    var lines = std.mem.splitSequence(u8, buf[0..header_end], "\r\n");
    const request_line = lines.next() orelse return null;
    var parts = std.mem.splitScalar(u8, request_line, ' ');
    const method = parts.next() orelse return null;
    const path = parts.next() orelse return null;
    return .{
        .method = method,
        .path = path,
        .body = buf[body_start .. body_start + content_length],
        .consumed = body_start + content_length,
    };
}

fn parseContentLength(header: []const u8) usize {
    var lines = std.mem.splitSequence(u8, header, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        if (std.ascii.eqlIgnoreCase(line[0..colon], "content-length")) {
            return std.fmt.parseInt(usize, std.mem.trim(u8, line[colon + 1 ..], " \t"), 10) catch 0;
        }
    }
    return 0;
}

/// Build and send one keepalive HTTP/1.1 response. No `Connection` header, so the
/// connection stays open (HTTP/1.1 default) and the client reuses it.
fn send(fd: i32, status: []const u8, content_type: []const u8, body: []const u8) void {
    var buf: [512]u8 = undefined;
    const resp = std.fmt.bufPrint(
        &buf,
        "HTTP/1.1 {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\n\r\n{s}",
        .{ status, content_type, body.len, body },
    ) catch return;
    sendAll(fd, resp);
}

fn sendAll(fd: i32, bytes: []const u8) void {
    var sent: usize = 0;
    var retries: u32 = 0;
    while (sent < bytes.len) {
        const rc = linux.sendto(fd, bytes[sent..].ptr, bytes.len - sent, MSG.NOSIGNAL, null, 0);
        switch (net.sysret(rc)) {
            .SUCCESS => {
                sent += rc;
                retries = 0;
            },
            .INTR => {},
            .AGAIN => {
                retries += 1;
                if (retries > 1000) return; // socket buffer stuck; give up
            },
            else => return,
        }
    }
}
