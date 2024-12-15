const std = @import("std");
const wasm = @import("web/wasm.zig");
const gpu = @import("web/gpu.zig");
const la = @import("linear_algebra.zig");
const log = std.log.scoped(.main_wasm);

pub const std_options = std.Options{
    .log_level = .info,
    .logFn = wasm.log,
};

var floor: Floor = undefined;
var enemies: [20]Enemy = undefined;

const Enemy = struct {
    const sprite_data = @embedFile("textures/bug.data");
    const sprite_width = 32;
    const sprite_height = 32;
    const frame_count = 3;

    const shader_billboard_code = @embedFile("shaders/billboard.wgsl");
    var pipeline: gpu.RenderPipeline = undefined;

    var uniform_buffer: gpu.Buffer = undefined;
    var instance_buffer: gpu.Buffer = undefined;
    var bind_group: gpu.BindGroup = undefined;

    const UniformData = struct {
        view: la.mat4,
        projection: la.mat4,
        scale: la.vec2,
        offset: la.vec2,
    };

    const InstanceData = extern struct {
        position: la.vec2,
        frame: u32,
    };

    position: la.vec2,
    alive: bool,

    fn init() void {
        const module = gpu.createShaderModule(.{ .code = shader_billboard_code });
        pipeline = gpu.createRenderPipeline(.{
            .vertex = .{ .module = module },
            .fragment = .{ .module = module },
            .depth_stencil = &.{
                .depth_compare = .greater,
                .format = .depth24plus,
                .depth_write_enabled = true,
            },
        });

        const texture = gpu.createTexture(.{
            .size = .{ .width = sprite_width, .height = sprite_height, .depth = frame_count },
            .format = .rgba8unorm,
            .usage = .{ .texture_binding = true, .copy_dst = true },
        });
        gpu.queueWriteTexture(texture, .{
            .data = sprite_data,
            .bytes_per_row = 4 * sprite_width,
            .rows_per_image = sprite_height,
            .width = sprite_width,
            .height = sprite_height,
            .depth = frame_count,
        });

        const sampler = gpu.createSampler(.{});

        uniform_buffer = gpu.createBuffer(.{
            .size = @sizeOf(UniformData),
            .usage = .{ .uniform = true, .copy_dst = true },
        });
        const sprite_scale: la.vec2 = .{ 0.25, -0.25 };
        const sprite_offset: la.vec2 = .{ 0, 0.25 };
        gpu.queueWriteBuffer(Enemy.uniform_buffer, 2 * @sizeOf(la.mat4), std.mem.asBytes(&sprite_scale));
        gpu.queueWriteBuffer(Enemy.uniform_buffer, 2 * @sizeOf(la.mat4) + @sizeOf(la.vec2), std.mem.asBytes(&sprite_offset));

        instance_buffer = gpu.createBuffer(.{
            .size = enemies.len * @sizeOf(InstanceData),
            .usage = .{ .storage = true, .copy_dst = true },
        });

        bind_group = gpu.createBindGroup(.{
            .layout = pipeline.getBindGroupLayout(0),
            .entries = &.{
                .{ .binding = 0, .resource = uniform_buffer },
                .{ .binding = 1, .resource = sampler },
                .{ .binding = 2, .resource = texture.createView(.{ .array_layer_count = frame_count }) },
                .{ .binding = 3, .resource = instance_buffer },
            },
        });
    }

    fn draw(render_pass: gpu.RenderPass) void {
        var instance_count: u32 = 0;

        for (enemies) |enemy| {
            if (enemy.alive) {
                const instance_data = [_]InstanceData{.{
                    .position = enemy.position,
                    .frame = 0,
                }};
                gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(InstanceData), std.mem.sliceAsBytes(&instance_data));
                instance_count += 1;
            }
        }

        render_pass.setPipeline(pipeline);
        render_pass.setBindGroup(0, bind_group);
        render_pass.draw(.{ .vertex_count = 6, .instance_count = instance_count });
    }
};

const Floor = struct {
    const circuit_data = @embedFile("textures/circuit.data");
    const circuit_width = 64;
    const circuit_height = 64;
    const tile_count = 5;

    const shader_textured_code = @embedFile("shaders/textured.wgsl");
    var pipeline: gpu.RenderPipeline = undefined;

    uniform_buffer: gpu.Buffer,
    bind_group: gpu.BindGroup,
    vertex_buffer: gpu.Buffer,
    index_buffer: gpu.Buffer,
    index_count: u32,

    fn init(self: *Floor, rows: u32, cols: u32) void {
        const module = gpu.createShaderModule(.{ .code = shader_textured_code });
        pipeline = gpu.createRenderPipeline(.{
            .vertex = .{
                .module = module,
                .buffers = &.{
                    .{
                        .array_stride = (3 + 2 + 1) * @sizeOf(f32),
                        .attributes = &.{
                            .{
                                .format = .float32x3,
                                .offset = 0,
                                .shader_location = 0,
                            },
                            .{
                                .format = .float32x2,
                                .offset = 3 * @sizeOf(f32),
                                .shader_location = 1,
                            },
                            .{
                                .format = .float32,
                                .offset = 5 * @sizeOf(f32),
                                .shader_location = 2,
                            },
                        },
                    },
                },
            },
            .fragment = .{
                .module = module,
            },
            .depth_stencil = &.{
                .depth_compare = .greater,
                .depth_write_enabled = true,
                .format = .depth24plus,
            },
        });

        self.uniform_buffer = gpu.createBuffer(.{
            .size = @sizeOf(la.mat4),
            .usage = .{ .uniform = true, .copy_dst = true },
        });

        const texture = gpu.createTexture(.{
            .size = .{ .width = circuit_width, .height = circuit_height, .depth = tile_count },
            .format = .rgba8unorm,
            .usage = .{ .texture_binding = true, .copy_dst = true },
        });
        gpu.queueWriteTexture(texture, .{
            .data = circuit_data,
            .bytes_per_row = 4 * circuit_width,
            .rows_per_image = circuit_height,
            .width = circuit_width,
            .height = circuit_height,
            .depth = tile_count,
        });
        const sampler = gpu.createSampler(.{});

        self.bind_group = gpu.createBindGroup(.{
            .layout = pipeline.getBindGroupLayout(0),
            .entries = &.{
                .{ .binding = 0, .resource = self.uniform_buffer },
                .{ .binding = 1, .resource = sampler },
                .{ .binding = 2, .resource = texture.createView(.{ .array_layer_count = tile_count }) },
            },
        });

        const vertex_size = 6 * @sizeOf(f32);
        self.vertex_buffer = gpu.createBuffer(.{
            .size = rows * cols * 4 * vertex_size,
            .usage = .{ .vertex = true, .copy_dst = true },
        });
        self.index_buffer = gpu.createBuffer(.{
            .size = rows * cols * 6 * @sizeOf(u16),
            .usage = .{ .index = true, .copy_dst = true },
        });
    }

    fn generate(self: *Floor, rows: u32, cols: u32, r: std.Random) void {
        var i: u16 = 0;
        for (0..rows) |row| {
            for (0..cols) |col| {
                defer i += 1;

                const x: f32 = @floatFromInt(col);
                const y: f32 = @floatFromInt(row);
                const tile: f32 = @floatFromInt(r.intRangeLessThan(u32, 0, tile_count));
                const vertex_data = [_]f32{
                    x + 0, y + 0, 0, 0, 0, tile,
                    x + 1, y + 0, 0, 1, 0, tile,
                    x + 1, y + 1, 0, 1, 1, tile,
                    x + 0, y + 1, 0, 0, 1, tile,
                };
                const vertex_size = 6 * @sizeOf(f32);
                gpu.queueWriteBuffer(self.vertex_buffer, i * 4 * vertex_size, std.mem.sliceAsBytes(&vertex_data));

                const offset: u16 = @intCast(i * 4);
                const index_data = [_]u16{
                    offset + 0, offset + 1, offset + 2,
                    offset + 0, offset + 2, offset + 3,
                };
                gpu.queueWriteBuffer(self.index_buffer, i * 6 * @sizeOf(u16), std.mem.sliceAsBytes(&index_data));
            }
        }
        self.index_count = i * 6;
    }

    fn draw(self: Floor, render_pass: gpu.RenderPass) void {
        render_pass.setPipeline(pipeline);
        render_pass.setBindGroup(0, self.bind_group);
        render_pass.setVertexBuffer(0, self.vertex_buffer, .{});
        render_pass.setIndexBuffer(self.index_buffer, .uint16, .{});
        render_pass.drawIndexed(.{ .index_count = self.index_count });
    }
};

pub export fn onInit() void {
    var rng = std.Random.DefaultPrng.init(0);
    const r = rng.random();

    floor.init(16, 16);
    floor.generate(16, 16, r);

    Enemy.init();
    for (&enemies) |*enemy| {
        enemy.position = .{ r.float(f32) * 16, r.float(f32) * 16 };
        enemy.alive = true;
    }
}

const Player = struct {
    theta: f32 = 0,
    phi: f32 = 0,
    x: f32 = 8,
    y: f32 = 0,
};
var player = Player{};

pub export fn onMouseMove(x: f32, y: f32) void {
    player.phi += x;
    player.theta = std.math.clamp(player.theta + y, -90, 90);
}

extern fn isKeyDown(key: u32) bool;

var t_prev: f32 = 0;
pub export fn onDraw() void {
    const t: f32 = @floatCast(wasm.performance.now() / 1000.0);
    defer t_prev = t;
    const dt = t - t_prev;

    var move_x: f32 = 0;
    var move_y: f32 = 0;
    if (isKeyDown(87)) move_y += 1;
    if (isKeyDown(65)) move_x -= 1;
    if (isKeyDown(83)) move_y -= 1;
    if (isKeyDown(68)) move_x += 1;

    const speed = 2;
    const s = @sin(std.math.degreesToRadians(player.phi));
    const c = @cos(std.math.degreesToRadians(player.phi));
    player.x += speed * dt * (c * move_x + s * move_y);
    player.y += speed * dt * (-s * move_x + c * move_y);

    const back_buffer = gpu.getCurrentTexture();
    defer back_buffer.release();

    const width = back_buffer.getWidth();
    const height = back_buffer.getHeight();

    const depth_texture = gpu.createTexture(.{
        .size = .{ .width = width, .height = height },
        .format = .depth24plus,
        .usage = .{ .render_attachment = true },
    });
    defer depth_texture.release();

    const aspect_ratio = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));

    const projection = la.perspective(60, aspect_ratio, 0.01);
    const view = la.mul(
        la.mul(la.rotation(player.theta - 90, .{ 1, 0, 0 }), la.rotation(player.phi, .{ 0, 0, 1 })),
        la.translation(-player.x, -player.y, -0.25),
    );
    var mvp = la.mul(projection, view);
    gpu.queueWriteBuffer(floor.uniform_buffer, 0, std.mem.sliceAsBytes(&mvp));
    gpu.queueWriteBuffer(Enemy.uniform_buffer, 0, std.mem.sliceAsBytes(&view));
    gpu.queueWriteBuffer(Enemy.uniform_buffer, @sizeOf(la.mat4), std.mem.sliceAsBytes(&projection));

    const command_encoder = gpu.createCommandEncoder();
    defer command_encoder.release();

    const back_buffer_view = back_buffer.createView(.{});
    defer back_buffer_view.release();
    const depth_texture_view = depth_texture.createView(.{});
    defer depth_texture_view.release();
    const render_pass = command_encoder.beginRenderPass(.{
        .color_attachments = &.{
            .{
                .view = back_buffer_view,
                .load_op = .clear,
                .store_op = .store,
                .clear_value = .{ .r = 0.2, .g = 0.2, .b = 0.3, .a = 1 },
            },
        },
        .depth_stencil_attachment = &.{
            .view = depth_texture_view,
            .depth_load_op = .clear,
            .depth_store_op = .store,
            .depth_clear_value = 0,
        },
    });
    defer render_pass.release();

    floor.draw(render_pass);
    Enemy.draw(render_pass);
    render_pass.end();

    const command_buffer = command_encoder.finish();
    defer command_buffer.release();
    gpu.queueSubmit(command_buffer);
}
