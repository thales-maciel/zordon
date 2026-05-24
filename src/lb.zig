//! Round-robin load balancer that passes accepted client sockets to backend API
//! processes via SCM_RIGHTS over Unix sockets. It copies no request/response bytes
//! and opens no second TCP connection: per client it does accept4 -> sendmsg(fd) ->
//! close. After the hand-off the client talks directly to a backend. No detection
//! logic lives here — it only distributes connections.
const std = @import("std");
const zordon = @import("zordon");
const net = zordon.net;
const linux = net.linux;
const EPOLL = net.EPOLL;
const SOCK = net.SOCK;

const max_events = 256;
const max_backends = 8;
const tag_listener: usize = 0; // backends use data.ptr = index + 1

pub fn main(init: std.process.Init) !void {
    const port = parsePort(init.environ_map.get("LB_PORT") orelse "9999") catch 9999;
    const backends_env = init.environ_map.get("LB_BACKENDS") orelse "/sock/api1.sock,/sock/api2.sock";

    var lb = Lb{ .epfd = -1, .listen_fd = -1, .n = 0, .paths = undefined, .fds = undefined };

    var it = std.mem.splitScalar(u8, backends_env, ',');
    while (it.next()) |raw| {
        const path = std.mem.trim(u8, raw, " \t");
        if (path.len == 0) continue;
        if (lb.n >= max_backends) break;
        lb.paths[lb.n] = path;
        lb.n += 1;
    }
    if (lb.n == 0) return error.NoBackends;

    const eprc = linux.epoll_create1(EPOLL.CLOEXEC);
    if (net.sysret(eprc) != .SUCCESS) return error.EpollCreateFailed;
    lb.epfd = @intCast(eprc);

    // Connect to every backend's Unix socket, retrying until each API has bound it.
    for (0..lb.n) |i| {
        lb.fds[i] = connectBackendRetry(lb.paths[i]);
        lb.registerBackend(i);
        std.log.info("connected to backend {s}", .{lb.paths[i]});
    }

    lb.listen_fd = try net.createTcpListener(port);
    var ev = linux.epoll_event{ .events = EPOLL.IN, .data = .{ .ptr = tag_listener } };
    if (net.sysret(linux.epoll_ctl(lb.epfd, EPOLL.CTL_ADD, lb.listen_fd, &ev)) != .SUCCESS) return error.EpollAddFailed;
    std.log.info("load balancer listening on 0.0.0.0:{d} -> {d} backend(s)", .{ port, lb.n });

    lb.run();
}

const Lb = struct {
    epfd: i32,
    listen_fd: i32,
    n: usize,
    paths: [max_backends][]const u8,
    fds: [max_backends]i32,
    rr: usize = 0,

    fn run(self: *Lb) void {
        var events: [max_events]linux.epoll_event = undefined;
        while (true) {
            const rc = linux.epoll_wait(self.epfd, &events, max_events, -1);
            switch (net.sysret(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                else => return,
            }
            for (events[0..rc]) |e| {
                if (e.data.ptr == tag_listener) {
                    self.acceptLoop();
                } else {
                    // The backend never sends us data, so any event means it closed.
                    self.reconnect(e.data.ptr - 1);
                }
            }
        }
    }

    fn acceptLoop(self: *Lb) void {
        while (true) {
            const rc = linux.accept4(self.listen_fd, null, null, SOCK.NONBLOCK | SOCK.CLOEXEC);
            switch (net.sysret(rc)) {
                .SUCCESS => {},
                .INTR => continue,
                .AGAIN => return,
                else => return,
            }
            const client_fd: i32 = @intCast(rc);
            // Set NODELAY now so the backend serves a Nagle-free socket from the start.
            net.setNoDelay(client_fd);
            self.dispatch(client_fd);
            // The backend holds its own dup of the socket; drop our reference.
            _ = linux.close(client_fd);
        }
    }

    /// Hand `client_fd` to the next backend (round-robin); on a dead backend,
    /// reconnect it and try the next one.
    fn dispatch(self: *Lb, client_fd: i32) void {
        var attempt: usize = 0;
        while (attempt < self.n) : (attempt += 1) {
            const idx = self.rr % self.n;
            self.rr +%= 1;
            net.sendFd(self.fds[idx], 1, client_fd) catch {
                self.reconnect(idx);
                continue;
            };
            return;
        }
        // All backends unreachable: drop the connection (client sees a reset).
    }

    fn registerBackend(self: *Lb, idx: usize) void {
        var ev = linux.epoll_event{ .events = EPOLL.IN | EPOLL.RDHUP, .data = .{ .ptr = idx + 1 } };
        _ = linux.epoll_ctl(self.epfd, EPOLL.CTL_ADD, self.fds[idx], &ev);
    }

    fn reconnect(self: *Lb, idx: usize) void {
        _ = linux.epoll_ctl(self.epfd, EPOLL.CTL_DEL, self.fds[idx], null);
        _ = linux.close(self.fds[idx]);
        self.fds[idx] = connectBackendRetry(self.paths[idx]);
        self.registerBackend(idx);
    }
};

fn connectBackendRetry(path: []const u8) i32 {
    while (true) {
        return net.connectUnix(path) catch {
            sleepMs(100);
            continue;
        };
    }
}

fn sleepMs(ms: u64) void {
    var req = linux.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    _ = linux.nanosleep(&req, null);
}

fn parsePort(value: []const u8) !u16 {
    return std.fmt.parseInt(u16, value, 10);
}
