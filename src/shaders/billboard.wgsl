struct UniformData {
    view: mat4x4f,
    projection: mat4x4f,
    scale: vec2f,
    offset: vec2f,
}

struct InstanceData {
    position: vec2f,
    texindex: u32,
}

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
    @location(1) @interpolate(flat) texindex: u32,
}

@group(0) @binding(0) var<uniform> uniformData: UniformData;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var ourTexture: texture_2d_array<f32>;
@group(0) @binding(3) var<storage, read> instanceData: array<InstanceData>;

@vertex fn vs(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32,
) -> VertexOut {
    let quad = array(
        vec2f(-1.0, -1.0),
        vec2f( 1.0, -1.0),
        vec2f( 1.0,  1.0),

        vec2f(-1.0, -1.0),
        vec2f( 1.0,  1.0),
        vec2f(-1.0,  1.0),
    );

    let instance = instanceData[instanceIndex];
    let spritePosition = uniformData.view * vec4f(instance.position, 0.0, 1.0);
    let vertexPosition = uniformData.scale * quad[vertexIndex] + uniformData.offset;

    var out: VertexOut;
    out.position = uniformData.projection * vec4f(spritePosition.xyz + vec3f(vertexPosition, 0.0), 1.0);
    out.texcoord = vec2f(0.5) + 0.5 * quad[vertexIndex];
    out.texindex = instance.texindex;
    return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
    let color = textureSample(ourTexture, ourSampler, in.texcoord, in.texindex);
    if (color.a == 0) {
        discard;
    }
    return color;
}