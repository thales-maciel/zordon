//! Low-level networking helpers shared by the API server (`server.zig`) and the
//! load balancer (`lb.zig`): raw `std.os.linux` syscall scaffolding, TCP/Unix
//! socket setup, and SCM_RIGHTS file-descriptor passing.
const std = @import("std");

pub const linux = std.os.linux;

pub const AF = linux.AF;
pub const SOCK = linux.SOCK;
pub const SOL = linux.SOL;
pub const SO = linux.SO;
pub const EPOLL = linux.EPOLL;
pub const MSG = linux.MSG;
pub const IPPROTO = linux.IPPROTO;
pub const TCP = linux.TCP;
pub const SCM = linux.SCM;

/// Decode a raw Linux syscall return (`-errno` as an unsigned value;
/// `std.posix.errno` only works for libc's `-1`/`_errno` convention).
pub fn sysret(rc: usize) linux.E {
    const s: isize = @bitCast(rc);
    if (s >= -4095 and s <= -1) return @enumFromInt(@as(u16, @intCast(-s)));
    return .SUCCESS;
}

pub fn setNoDelay(fd: i32) void {
    const one: c_int = 1;
    _ = linux.setsockopt(fd, IPPROTO.TCP, TCP.NODELAY, @ptrCast(&one), @sizeOf(c_int));
}

/// Non-blocking TCP listener on 0.0.0.0:port (REUSEADDR + REUSEPORT).
pub fn createTcpListener(port: u16) !i32 {
    const rc = linux.socket(AF.INET, SOCK.STREAM | SOCK.NONBLOCK | SOCK.CLOEXEC, 0);
    if (sysret(rc) != .SUCCESS) return error.SocketFailed;
    const fd: i32 = @intCast(rc);
    errdefer _ = linux.close(fd);

    const one: c_int = 1;
    _ = linux.setsockopt(fd, SOL.SOCKET, SO.REUSEADDR, @ptrCast(&one), @sizeOf(c_int));
    _ = linux.setsockopt(fd, SOL.SOCKET, SO.REUSEPORT, @ptrCast(&one), @sizeOf(c_int));

    var addr = linux.sockaddr.in{
        .family = AF.INET,
        .port = std.mem.nativeToBig(u16, port),
        .addr = 0,
        .zero = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    if (sysret(linux.bind(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr)))) != .SUCCESS) return error.BindFailed;
    if (sysret(linux.listen(fd, 1024)) != .SUCCESS) return error.ListenFailed;
    return fd;
}

fn unixAddr(path: []const u8) !linux.sockaddr.un {
    var addr = linux.sockaddr.un{ .family = AF.UNIX, .path = undefined };
    @memset(&addr.path, 0);
    if (path.len >= addr.path.len) return error.PathTooLong;
    @memcpy(addr.path[0..path.len], path);
    return addr;
}

/// Non-blocking AF_UNIX stream listener bound to a filesystem `path`
/// (any stale socket file is removed first).
pub fn createUnixListener(path: []const u8) !i32 {
    var pathz: [108:0]u8 = undefined;
    if (path.len >= pathz.len) return error.PathTooLong;
    @memcpy(pathz[0..path.len], path);
    pathz[path.len] = 0;
    _ = linux.unlink(&pathz);

    const rc = linux.socket(AF.UNIX, SOCK.STREAM | SOCK.NONBLOCK | SOCK.CLOEXEC, 0);
    if (sysret(rc) != .SUCCESS) return error.SocketFailed;
    const fd: i32 = @intCast(rc);
    errdefer _ = linux.close(fd);

    var addr = try unixAddr(path);
    if (sysret(linux.bind(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr)))) != .SUCCESS) return error.BindFailed;
    if (sysret(linux.listen(fd, 1024)) != .SUCCESS) return error.ListenFailed;
    return fd;
}

/// Blocking AF_UNIX stream connection to `path`.
pub fn connectUnix(path: []const u8) !i32 {
    const rc = linux.socket(AF.UNIX, SOCK.STREAM | SOCK.CLOEXEC, 0);
    if (sysret(rc) != .SUCCESS) return error.SocketFailed;
    const fd: i32 = @intCast(rc);
    errdefer _ = linux.close(fd);

    var addr = try unixAddr(path);
    if (sysret(linux.connect(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr)))) != .SUCCESS) return error.ConnectFailed;
    return fd;
}

// --- SCM_RIGHTS fd passing ---
// No CMSG_* macros exist in std, so the control buffer is hand-rolled. `cmsghdr`
// uses a usize `len` field (kernel layout), so @sizeOf == 16 on x86-64 — always
// use @sizeOf, never a hardcoded constant.
const cmsg_data_len = @sizeOf(i32);
const cmsg_len = @sizeOf(linux.cmsghdr) + cmsg_data_len;
const cmsg_space = std.mem.alignForward(usize, cmsg_len, @sizeOf(usize));

/// Send `fd_to_pass` over the connected Unix socket `unix_fd`, alongside one byte
/// of ordinary data (some kernels require >=1 iov byte to deliver ancillary data).
pub fn sendFd(unix_fd: i32, payload: u8, fd_to_pass: i32) !void {
    var control: [cmsg_space]u8 align(@alignOf(linux.cmsghdr)) = undefined;
    @memset(&control, 0);
    const cmsg: *linux.cmsghdr = @ptrCast(&control);
    cmsg.* = .{ .len = cmsg_len, .level = SOL.SOCKET, .type = SCM.RIGHTS };
    const data: *align(1) i32 = @ptrCast(&control[@sizeOf(linux.cmsghdr)]);
    data.* = fd_to_pass;

    var byte = payload;
    var iov = [_]std.posix.iovec_const{.{ .base = @ptrCast(&byte), .len = 1 }};
    var msg = linux.msghdr_const{
        .name = null,
        .namelen = 0,
        .iov = &iov,
        .iovlen = 1,
        .control = &control,
        .controllen = cmsg_space,
        .flags = 0,
    };
    while (true) {
        switch (sysret(linux.sendmsg(unix_fd, &msg, MSG.NOSIGNAL))) {
            .SUCCESS => return,
            .INTR => continue,
            else => return error.SendFdFailed,
        }
    }
}

pub const RecvFd = union(enum) {
    fd: i32,
    again, // no message ready (EAGAIN) or a message carried no descriptor
    closed, // peer hung up
};

/// Receive one descriptor from `unix_fd`. Assumes a non-blocking socket.
pub fn recvFd(unix_fd: i32) RecvFd {
    var control: [cmsg_space]u8 align(@alignOf(linux.cmsghdr)) = undefined;
    @memset(&control, 0);
    var byte: u8 = 0;
    var iov = [_]std.posix.iovec{.{ .base = @ptrCast(&byte), .len = 1 }};
    var msg = linux.msghdr{
        .name = null,
        .namelen = 0,
        .iov = &iov,
        .iovlen = 1,
        .control = &control,
        .controllen = cmsg_space,
        .flags = 0,
    };
    while (true) {
        const rc = linux.recvmsg(unix_fd, &msg, 0);
        switch (sysret(rc)) {
            .SUCCESS => {
                if (rc == 0) return .closed;
                if (msg.controllen < cmsg_len) return .again;
                const cmsg: *const linux.cmsghdr = @ptrCast(&control);
                if (cmsg.level != SOL.SOCKET or cmsg.type != SCM.RIGHTS) return .again;
                const data: *align(1) const i32 = @ptrCast(&control[@sizeOf(linux.cmsghdr)]);
                return .{ .fd = data.* };
            },
            .INTR => continue,
            .AGAIN => return .again,
            else => return .closed,
        }
    }
}

test "sendFd/recvFd round-trips a descriptor across a socketpair" {
    var pair: [2]i32 = undefined;
    if (sysret(linux.socketpair(AF.UNIX, SOCK.STREAM, 0, &pair)) != .SUCCESS) return error.SkipZigTest;
    defer _ = linux.close(pair[0]);
    defer _ = linux.close(pair[1]);

    var pipefds: [2]i32 = undefined;
    if (sysret(linux.pipe2(&pipefds, .{})) != .SUCCESS) return error.SkipZigTest;
    defer _ = linux.close(pipefds[0]);

    try sendFd(pair[0], 'x', pipefds[1]);
    _ = linux.close(pipefds[1]); // our copy; the passed dup keeps the pipe open

    const got = switch (recvFd(pair[1])) {
        .fd => |f| f,
        else => return error.NoFdReceived,
    };
    defer _ = linux.close(got);

    const message = "hello";
    try std.testing.expectEqual(message.len, linux.write(got, message, message.len));
    var buf: [16]u8 = undefined;
    const n = linux.read(pipefds[0], &buf, buf.len);
    try std.testing.expectEqualStrings(message, buf[0..n]);
}
