struct VertexIn {
    @location(0) position: vec3f,
    @location(1) texcoord: vec2f,
    @location(2) texindex: f32,
}

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
    @location(1) texindex: f32,
}

@group(0) @binding(0) var<uniform> mvp: mat4x4f;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var ourTexture: texture_2d_array<f32>;

@vertex fn vs(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = mvp * vec4f(in.position, 1.0);
    out.texcoord = in.texcoord;
    out.texindex = in.texindex;
    return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
    return textureSample(ourTexture, ourSampler, in.texcoord, u32(in.texindex));
}