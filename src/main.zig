const std = @import("std");
const wasm = @import("web/wasm.zig");
const gpu = @import("web/gpu.zig");
const la = @import("linear_algebra.zig");
const log = std.log.scoped(.main_wasm);

pub const std_options = std.Options{
    .log_level = .info,
    .logFn = wasm.log,
};

var depth_texture: gpu.Texture = undefined;

var floor: Floor = undefined;
var enemies: [20]Enemy = undefined;
var shots: [20]Shot = undefined;

const level_width = 16;
const level_height = 16;

const Billboard = struct {
    const shader_billboard_code = @embedFile("shaders/billboard.wgsl");
    var pipeline: gpu.RenderPipeline = undefined;

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
    }
};

const Shot = struct {
    const speed = 8;

    const sprite_data = @embedFile("textures/shot.data");
    const sprite_width = 8;
    const sprite_height = 8;

    var uniform_buffer: gpu.Buffer = undefined;
    var instance_buffer: gpu.Buffer = undefined;
    var bind_group: gpu.BindGroup = undefined;

    position: la.vec2,
    direction: la.vec2,
    active: bool,

    fn init() void {
        const texture = gpu.createTexture(.{
            .size = .{ .width = sprite_width, .height = sprite_height },
            .format = .rgba8unorm,
            .usage = .{ .texture_binding = true, .copy_dst = true },
        });
        gpu.queueWriteTexture(texture, .{
            .data = sprite_data,
            .bytes_per_row = 4 * sprite_width,
            .width = sprite_width,
            .height = sprite_height,
        });

        const sampler = gpu.createSampler(.{});

        uniform_buffer = gpu.createBuffer(.{
            .size = @sizeOf(Billboard.UniformData),
            .usage = .{ .uniform = true, .copy_dst = true },
        });
        const sprite_scale: la.vec2 = .{ 1.0 / 16.0, -1.0 / 16.0 };
        const sprite_offset: la.vec2 = .{ 0, 0.25 };
        gpu.queueWriteBuffer(uniform_buffer, 2 * @sizeOf(la.mat4), std.mem.asBytes(&sprite_scale));
        gpu.queueWriteBuffer(uniform_buffer, 2 * @sizeOf(la.mat4) + @sizeOf(la.vec2), std.mem.asBytes(&sprite_offset));

        instance_buffer = gpu.createBuffer(.{
            .size = shots.len * @sizeOf(Billboard.InstanceData),
            .usage = .{ .storage = true, .copy_dst = true },
        });

        bind_group = gpu.createBindGroup(.{
            .layout = Billboard.pipeline.getBindGroupLayout(0),
            .entries = &.{
                .{ .binding = 0, .resource = uniform_buffer },
                .{ .binding = 1, .resource = sampler },
                .{ .binding = 2, .resource = texture.createView(.{ .dimension = .@"2d_array" }) },
                .{ .binding = 3, .resource = instance_buffer },
            },
        });
    }

    fn updateAll(dt: f32) void {
        for (&shots) |*shot| {
            if (shot.active) {
                shot.position += shot.direction * @as(la.vec2, @splat(dt));

                // check enemies
                for (&enemies) |*enemy| {
                    if (enemy.active) {
                        const diff = shot.position - enemy.position;
                        const dist_sqr = diff[0] * diff[0] + diff[1] * diff[1];
                        if (dist_sqr < 0.3 * 0.3) {
                            enemy.hit();
                            shot.active = false;
                        }
                    }
                }

                if (shot.position[0] < 0 or shot.position[1] < 0 or shot.position[0] > level_width or shot.position[1] > level_height) {
                    shot.active = false;
                }
            }
        }
    }

    fn drawAll(render_pass: gpu.RenderPass) void {
        var instance_count: u32 = 0;

        for (shots) |shot| {
            if (shot.active) {
                const instance_data = [_]Billboard.InstanceData{.{
                    .position = shot.position,
                    .frame = 0,
                }};
                gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(Billboard.InstanceData), std.mem.sliceAsBytes(&instance_data));
                instance_count += 1;
            }
        }

        render_pass.setPipeline(Billboard.pipeline);
        render_pass.setBindGroup(0, bind_group);
        render_pass.draw(.{ .vertex_count = 6, .instance_count = instance_count });
    }
};

const Enemy = struct {
    const sprite_data = @embedFile("textures/bug.data");
    const sprite_width = 32;
    const sprite_height = 32;
    const frame_count = 4;

    var uniform_buffer: gpu.Buffer = undefined;
    var instance_buffer: gpu.Buffer = undefined;
    var bind_group: gpu.BindGroup = undefined;

    position: la.vec2,
    frame: u32,
    idle: f32,
    hurt_cooldown: f32,
    active: bool,

    fn init() void {
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
            .size = @sizeOf(Billboard.UniformData),
            .usage = .{ .uniform = true, .copy_dst = true },
        });
        const sprite_scale: la.vec2 = .{ 0.25, -0.25 };
        const sprite_offset: la.vec2 = .{ 0, 0.25 };
        gpu.queueWriteBuffer(uniform_buffer, 2 * @sizeOf(la.mat4), std.mem.asBytes(&sprite_scale));
        gpu.queueWriteBuffer(uniform_buffer, 2 * @sizeOf(la.mat4) + @sizeOf(la.vec2), std.mem.asBytes(&sprite_offset));

        instance_buffer = gpu.createBuffer(.{
            .size = enemies.len * @sizeOf(Billboard.InstanceData),
            .usage = .{ .storage = true, .copy_dst = true },
        });

        bind_group = gpu.createBindGroup(.{
            .layout = Billboard.pipeline.getBindGroupLayout(0),
            .entries = &.{
                .{ .binding = 0, .resource = uniform_buffer },
                .{ .binding = 1, .resource = sampler },
                .{ .binding = 2, .resource = texture.createView(.{
                    .dimension = .@"2d_array",
                    .array_layer_count = frame_count,
                }) },
                .{ .binding = 3, .resource = instance_buffer },
            },
        });
    }

    fn hit(self: *Enemy) void {
        self.hurt_cooldown = 0.5;
    }

    fn updateAll(dt: f32) void {
        for (&enemies) |*enemy| {
            if (enemy.active) {
                if (enemy.hurt_cooldown > 0) {
                    enemy.hurt_cooldown -= dt;
                    enemy.frame = if (@sin(40 * enemy.hurt_cooldown) > 0) 2 else 3;
                } else {
                    if (enemy.frame > 1) enemy.frame = 0;
                    if (enemy.idle > 0) {
                        enemy.idle -= dt;
                    } else {
                        enemy.idle = 0.25;
                        enemy.frame = if (enemy.frame != 0) 0 else 1;
                    }

                    // walk towards player
                    const dir = la.vec2{ player.x, player.y } - enemy.position;
                    enemy.position += dir * @as(la.vec2, @splat(0.1 * dt));
                }
            }
        }
    }

    fn drawAll(render_pass: gpu.RenderPass) void {
        var instance_count: u32 = 0;

        for (enemies) |enemy| {
            if (enemy.active) {
                const instance_data = [_]Billboard.InstanceData{.{
                    .position = enemy.position,
                    .frame = enemy.frame,
                }};
                gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(Billboard.InstanceData), std.mem.sliceAsBytes(&instance_data));
                instance_count += 1;
            }
        }

        render_pass.setPipeline(Billboard.pipeline);
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
                .{ .binding = 2, .resource = texture.createView(.{
                    .dimension = .@"2d_array",
                    .array_layer_count = tile_count,
                }) },
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
    const back_buffer = gpu.getCurrentTexture();
    depth_texture = gpu.createTexture(.{
        .size = .{ .width = back_buffer.getWidth(), .height = back_buffer.getHeight() },
        .format = .depth24plus,
        .usage = .{ .render_attachment = true },
    });

    var rng = std.Random.DefaultPrng.init(0);
    const r = rng.random();

    floor.init(level_width, level_height);
    floor.generate(level_width, level_height, r);

    Billboard.init();
    Shot.init();
    for (&shots) |*shot| {
        // shot.position = .{ r.float(f32) * level_width, r.float(f32) * level_height };
        // shot.direction = .{ 8 * r.float(f32) - 4, 8 * r.float(f32) - 4 };
        shot.active = false;
    }
    Enemy.init();
    for (&enemies) |*enemy| {
        enemy.position = .{ r.float(f32) * level_width, r.float(f32) * level_height };
        enemy.frame = 0;
        enemy.hurt_cooldown = 0;
        enemy.idle = r.float(f32);
        enemy.active = true;
    }
}

const Player = struct {
    const speed = 2;

    theta: f32 = 0,
    phi: f32 = 0,
    x: f32 = 8,
    y: f32 = 0,
};
var player = Player{};

var shoot = false;
pub export fn onMouseDown() void {
    shoot = true;
}

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

    const s = @sin(std.math.degreesToRadians(player.phi));
    const c = @cos(std.math.degreesToRadians(player.phi));
    const dir_x = (c * move_x + s * move_y);
    const dir_y = (-s * move_x + c * move_y);
    player.x += dir_x * Player.speed * dt;
    player.y += dir_y * Player.speed * dt;

    if (shoot) {
        shoot = false;
        for (&shots) |*shot| {
            if (!shot.active) {
                shot.active = true;
                shot.direction = .{ s, c };
                shot.position = .{ player.x, player.y };
                shot.position += shot.direction * la.vec2{ 0.2, 0.2 }; // prevent screen flash
                shot.direction *= @splat(Shot.speed);
                break;
            }
        }
    }

    Shot.updateAll(dt);
    Enemy.updateAll(dt);

    const back_buffer = gpu.getCurrentTexture();
    defer back_buffer.release();

    const width = back_buffer.getWidth();
    const height = back_buffer.getHeight();

    if (depth_texture.getWidth() != width or depth_texture.getHeight() != height) {
        depth_texture.release();
        depth_texture = gpu.createTexture(.{
            .size = .{ .width = width, .height = height },
            .format = .depth24plus,
            .usage = .{ .render_attachment = true },
        });
    }

    const aspect_ratio = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));

    const projection = la.perspective(60, aspect_ratio, 0.01);
    const view = la.mul(
        la.mul(la.rotation(player.theta - 90, .{ 1, 0, 0 }), la.rotation(player.phi, .{ 0, 0, 1 })),
        la.translation(-player.x, -player.y, -0.25),
    );
    var mvp = la.mul(projection, view);
    gpu.queueWriteBuffer(floor.uniform_buffer, 0, std.mem.sliceAsBytes(&mvp));
    gpu.queueWriteBuffer(Shot.uniform_buffer, 0, std.mem.sliceAsBytes(&view));
    gpu.queueWriteBuffer(Shot.uniform_buffer, @sizeOf(la.mat4), std.mem.sliceAsBytes(&projection));
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
    Shot.drawAll(render_pass);
    Enemy.drawAll(render_pass);
    render_pass.end();

    const command_buffer = command_encoder.finish();
    defer command_buffer.release();
    gpu.queueSubmit(command_buffer);
}
