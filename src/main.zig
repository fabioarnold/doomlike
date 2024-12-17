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

const LightData = struct {
    const Light = struct {
        position: la.vec3,
        color: la.vec3,
    };

    lights: [8]Light,
    active_lights: u32,
};
var light_data: LightData = undefined;
var light_buffer: gpu.Buffer = undefined;

const Level = struct {
    const rows = 16;
    const cols = 16;

    const Tile = enum(u1) {
        empty,
        solid,
    };

    var tilemap: [rows * cols]Tile = undefined;

    const UniformData = struct {
        view: la.mat4,
        projection: la.mat4,
    };
    const InstanceData = struct {
        model: la.mat4,
    };

    const texture_data = @embedFile("textures/brickwall.data");
    const texture_width = 32;
    const texture_height = 32;

    const shader_code = @embedFile("shaders/tile.wgsl");
    var pipeline: gpu.RenderPipeline = undefined;

    var uniform_buffer: gpu.Buffer = undefined;
    var instance_buffer: gpu.Buffer = undefined;
    var bind_group: gpu.BindGroup = undefined;
    var instance_count: u32 = 0;

    fn init() void {
        const module = gpu.createShaderModule(.{ .code = shader_code });
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
            .size = .{ .width = texture_width, .height = texture_height },
            .format = .rgba8unorm,
            .usage = .{ .texture_binding = true, .copy_dst = true },
        });
        gpu.queueWriteTexture(texture, .{
            .data = texture_data,
            .bytes_per_row = 4 * texture_width,
            .width = texture_width,
            .height = texture_height,
        });

        const sampler = gpu.createSampler(.{});

        uniform_buffer = gpu.createBuffer(.{
            .size = @sizeOf(UniformData),
            .usage = .{ .uniform = true, .copy_dst = true },
        });

        instance_buffer = gpu.createBuffer(.{
            .size = tilemap.len * 4 * @sizeOf(InstanceData),
            .usage = .{ .storage = true, .copy_dst = true },
        });

        bind_group = gpu.createBindGroup(.{
            .layout = pipeline.getBindGroupLayout(0),
            .entries = &.{
                .{ .binding = 0, .resource = uniform_buffer },
                .{ .binding = 1, .resource = sampler },
                .{ .binding = 2, .resource = texture.createView(.{ .dimension = .@"2d_array" }) },
                .{ .binding = 3, .resource = instance_buffer },
                .{ .binding = 4, .resource = light_buffer },
            },
        });
    }

    fn generate() void {
        tilemap = .{
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .solid, .solid, .empty, .empty, .solid, .solid, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty, .empty, .empty, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .solid, .solid, .solid, .solid, .solid, .solid, .solid, .solid, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
            .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty, .empty,
        };

        for (0..rows) |row| {
            for (0..cols) |col| {
                const i = row * cols + col;
                const x: f32 = @floatFromInt(col);
                const y: f32 = @floatFromInt(row);
                if (tilemap[i] != .empty) continue;

                // floor
                {
                    const instance_data: InstanceData = .{
                        .model = la.mul(la.translation(x + 0.5, y + 0.5, 0), la.scale(0.5, 0.5, 0.5)),
                    };
                    gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(InstanceData), std.mem.asBytes(&instance_data));
                    instance_count += 1;
                }

                if (col == 0 or tilemap[i - 1] == .solid) {
                    const instance_data: InstanceData = .{
                        .model = la.mul(la.mul(la.mul(la.translation(x, y + 0.5, 0.5), la.rotation(-90, .{ 0, 1, 0 })), la.rotation(90, .{ 0, 0, 1 })), la.scale(0.5, 0.5, 0.5)),
                    };
                    gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(InstanceData), std.mem.asBytes(&instance_data));
                    instance_count += 1;
                }
                if (col + 1 == cols or tilemap[i + 1] == .solid) {
                    const instance_data: InstanceData = .{
                        .model = la.mul(la.mul(la.mul(la.translation(x + 1, y + 0.5, 0.5), la.rotation(90, .{ 0, 1, 0 })), la.rotation(270, .{ 0, 0, 1 })), la.scale(0.5, 0.5, 0.5)),
                    };
                    gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(InstanceData), std.mem.asBytes(&instance_data));
                    instance_count += 1;
                }
                if (row == 0 or tilemap[i - rows] == .solid) {
                    const instance_data: InstanceData = .{
                        .model = la.mul(la.mul(la.mul(la.translation(x + 0.5, y, 0.5), la.rotation(90, .{ 1, 0, 0 })), la.rotation(180, .{ 0, 0, 1 })), la.scale(0.5, 0.5, 0.5)),
                    };
                    gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(InstanceData), std.mem.asBytes(&instance_data));
                    instance_count += 1;
                }
                if (row + 1 == rows or tilemap[i + rows] == .solid) {
                    const instance_data: InstanceData = .{
                        .model = la.mul(la.mul(la.translation(x + 0.5, y + 1, 0.5), la.rotation(-90, .{ 1, 0, 0 })), la.scale(0.5, 0.5, 0.5)),
                    };
                    gpu.queueWriteBuffer(instance_buffer, instance_count * @sizeOf(InstanceData), std.mem.asBytes(&instance_data));
                    instance_count += 1;
                }
            }
        }
    }

    fn distance(p: la.vec2) f32 {
        const hw = 0.5 * cols;
        const hh = 0.5 * rows;
        var min_dist = @min(hw - @abs(hw - p[0]), hh - @abs(hh - p[1]));

        for (0..rows) |row| {
            for (0..cols) |col| {
                const i = row * cols + col;
                if (tilemap[i] == .empty) continue;
                const x: f32 = @floatFromInt(col);
                const y: f32 = @floatFromInt(row);
                const dx = @max(@abs(x + 0.5 - p[0]) - 0.5, 0);
                const dy = @max(@abs(y + 0.5 - p[1]) - 0.5, 0);
                min_dist = @min(min_dist, @sqrt(dx * dx + dy * dy));
            }
        }

        return min_dist;
    }

    fn draw(render_pass: gpu.RenderPass) void {
        if (instance_count == 0) return;

        render_pass.setPipeline(pipeline);
        render_pass.setBindGroup(0, bind_group);
        render_pass.draw(.{ .vertex_count = 6, .instance_count = instance_count });
    }
};

const Billboard = struct {
    const shader_code = @embedFile("shaders/billboard.wgsl");
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
        const module = gpu.createShaderModule(.{ .code = shader_code });
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
    const speed = 16;

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
        const sprite_offset: la.vec2 = .{ 0, 0.5 };
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
                        if (dist_sqr < 0.5 * 0.5) {
                            enemy.hit();
                            shot.active = false;
                        }
                    }
                }

                if (shot.position[0] < 0 or shot.position[1] < 0 or shot.position[0] > Level.cols or shot.position[1] > Level.rows) {
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
        if (instance_count == 0) return;

        render_pass.setPipeline(Billboard.pipeline);
        render_pass.setBindGroup(0, bind_group);
        render_pass.draw(.{ .vertex_count = 6, .instance_count = instance_count });
    }
};

const Enemy = struct {
    const speed = 1;

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
        const sprite_scale: la.vec2 = .{ 0.5, -0.5 };
        const sprite_offset: la.vec2 = .{ 0, 0.5 };
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
                    var dir = la.vec2{ player.x, player.y } - enemy.position;
                    const dir_len_sqr = dir[0] * dir[0] + dir[1] * dir[1];
                    if (dir_len_sqr > 0.5 * 0.5 and dir_len_sqr < 5 * 5) {
                        dir /= @splat(@sqrt(dir_len_sqr));
                        enemy.position += dir * @as(la.vec2, @splat(speed * dt));
                    }
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
        if (instance_count == 0) return;

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

    player.x = 8;
    player.y = 2;

    light_buffer = gpu.createBuffer(.{
        .size = @sizeOf(LightData),
        .usage = .{ .uniform = true, .copy_dst = true },
    });

    // floor.init(Level.cols / 2, Level.rows / 2);
    // floor.generate(Level.cols / 2, Level.rows / 2, r);

    Level.init();
    Level.generate();

    Billboard.init();
    Shot.init();
    for (&shots) |*shot| {
        // shot.position = .{ r.float(f32) * Level.cols, r.float(f32) * Level.rows };
        // shot.direction = .{ 8 * r.float(f32) - 4, 8 * r.float(f32) - 4 };
        shot.active = false;
    }
    Enemy.init();
    for (&enemies) |*enemy| {
        // enemy.position = .{ r.float(f32) * Level.cols, r.float(f32) * Level.rows };
        enemy.position = .{ 14, 6 };
        enemy.frame = 0;
        enemy.hurt_cooldown = 0;
        enemy.idle = r.float(f32);
        enemy.active = true;
        break;
    }
}

const Player = struct {
    const speed = 4;
    const radius = 0.5;

    x: f32 = 0,
    y: f32 = 0,
    phi: f32 = 0,
    theta: f32 = 0,
};
var player = Player{};

const Pointers = struct {
    player: *const Player,
    tilemap: *const @TypeOf(Level.tilemap),
    enemies: *const @TypeOf(enemies),
    shots: *const @TypeOf(shots),
};

pub export fn getPointers() *const Pointers {
    log.info("@sizeOf(Enemy)={}", .{@sizeOf(Enemy)});
    log.info("@offsetOf(Enemy, active)={}", .{@offsetOf(Enemy, "active")});
    log.info("@sizeOf(Shot)={}", .{@sizeOf(Shot)});
    log.info("@offsetOf(Shot, active)={}", .{@offsetOf(Shot, "active")});
    return &.{
        .player = &player,
        .tilemap = &Level.tilemap,
        .enemies = &enemies,
        .shots = &shots,
    };
}

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

    var input_x: f32 = 0;
    var input_y: f32 = 0;
    if (isKeyDown(87)) input_y += 1;
    if (isKeyDown(65)) input_x -= 1;
    if (isKeyDown(83)) input_y -= 1;
    if (isKeyDown(68)) input_x += 1;

    const s = @sin(std.math.degreesToRadians(player.phi));
    const c = @cos(std.math.degreesToRadians(player.phi));
    const dir_x = (c * input_x + s * input_y);
    const dir_y = (-s * input_x + c * input_y);
    var move_x = dir_x * Player.speed * dt;
    var move_y = dir_y * Player.speed * dt;
    var test_x = player.x + move_x;
    var test_y = player.y + move_y;
    if (Level.distance(.{ test_x, test_y }) < Player.radius) {
        // compute normal
        var n_x = Level.distance(.{ player.x + 0.01, player.y }) - Level.distance(.{ player.x - 0.01, player.y });
        var n_y = Level.distance(.{ player.x, player.y + 0.01 }) - Level.distance(.{ player.x, player.y - 0.01 });
        const n_len = @sqrt(n_x * n_x + n_y * n_y);
        n_x /= n_len;
        n_y /= n_len;
        // clip
        const backoff = n_x * move_x + n_y * move_y;
        move_x -= backoff * n_x;
        move_y -= backoff * n_y;
        // try again
        test_x = player.x + move_x;
        test_y = player.y + move_y;
        if (Level.distance(.{ test_x, test_y }) > Player.radius) {
            player.x = test_x;
            player.y = test_y;
        }
    } else {
        player.x = test_x;
        player.y = test_y;
    }

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
        la.translation(-player.x, -player.y, -0.5),
    );
    // var mvp = la.mul(la.mul(projection, view), la.scale(2, 2, 2));
    // gpu.queueWriteBuffer(floor.uniform_buffer, 0, std.mem.sliceAsBytes(&mvp));
    gpu.queueWriteBuffer(Level.uniform_buffer, 0, std.mem.sliceAsBytes(&view));
    gpu.queueWriteBuffer(Level.uniform_buffer, @sizeOf(la.mat4), std.mem.sliceAsBytes(&projection));
    gpu.queueWriteBuffer(Shot.uniform_buffer, 0, std.mem.sliceAsBytes(&view));
    gpu.queueWriteBuffer(Shot.uniform_buffer, @sizeOf(la.mat4), std.mem.sliceAsBytes(&projection));
    gpu.queueWriteBuffer(Enemy.uniform_buffer, 0, std.mem.sliceAsBytes(&view));
    gpu.queueWriteBuffer(Enemy.uniform_buffer, @sizeOf(la.mat4), std.mem.sliceAsBytes(&projection));

    // update lights
    light_data.lights[0].position = .{ player.x, player.y, 0.5 };
    light_data.lights[0].color = .{ 1, 1, 1 };
    light_data.active_lights = 1;
    for (shots) |shot| {
        if (shot.active) {
            light_data.lights[light_data.active_lights].position = .{ shot.position[0], shot.position[1], 0.5 };
            light_data.lights[light_data.active_lights].color = .{ 1.0, 0.5, 0 };
            light_data.active_lights += 1;
            break;
        }
    }
    gpu.queueWriteBuffer(light_buffer, 0, std.mem.asBytes(&light_data));

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

    // floor.draw(render_pass);
    Level.draw(render_pass);
    Shot.drawAll(render_pass);
    Enemy.drawAll(render_pass);
    render_pass.end();

    const command_buffer = command_encoder.finish();
    defer command_buffer.release();
    gpu.queueSubmit(command_buffer);
}
