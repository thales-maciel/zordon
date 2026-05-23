const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .x86_64,
            .os_tag = .linux,
            .abi = .musl,
        },
    });
    const optimize = b.standardOptimizeOption(.{ .preferred_optimize_mode = .ReleaseFast });

    const zordon_mod = b.addModule("zordon", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "zordon",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zordon", .module = zordon_mod },
            },
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the API server");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    run_step.dependOn(&run_cmd.step);

    const preprocess = b.addExecutable(.{
        .name = "preprocess-references",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/preprocess.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zordon", .module = zordon_mod },
            },
        }),
    });

    const preprocess_step = b.step("preprocess", "Convert references JSON into the compact model file");
    const preprocess_cmd = b.addRunArtifact(preprocess);
    if (b.args) |args| preprocess_cmd.addArgs(args);
    preprocess_step.dependOn(&preprocess_cmd.step);

    const tests = b.addTest(.{
        .root_module = zordon_mod,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
