const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
    const optimize = b.standardOptimizeOption(.{});

    const wasm = b.addExecutable(.{
        .name = "main",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    wasm.rdynamic = true;
    wasm.entry = .disabled;
    b.installArtifact(wasm);
}
