struct UniformData {
    view: mat4x4f,
    projection: mat4x4f,
}

struct Light {
    position: vec3f,
    color: vec3f,
}

struct LightData {
    lights: array<Light, 8>,
    activeLights: u32,
}

struct InstanceData {
    model: mat4x4f,
    // texindex: u32,
}

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
    @location(1) @interpolate(flat) texindex: u32,
    @location(2) worldPosition: vec3f,
}

@group(0) @binding(0) var<uniform> uniformData: UniformData;
@group(0) @binding(1) var ourSampler: sampler;
@group(0) @binding(2) var ourTexture: texture_2d_array<f32>;
@group(0) @binding(3) var<storage, read> instanceData: array<InstanceData>;
@group(0) @binding(4) var<uniform> lightData: LightData;

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
    let vertexPosition = instance.model * vec4f(quad[vertexIndex], 0.0, 1.0);

    var out: VertexOut;
    out.position = uniformData.projection * uniformData.view * vertexPosition;
    out.texcoord = vec2f(0.5) + 0.5 * quad[vertexIndex];
    out.texindex = 0;
    out.worldPosition = vertexPosition.xyz;
    return out;
}

@fragment fn fs(in: VertexOut) -> @location(0) vec4f {
    var color = textureSample(ourTexture, ourSampler, in.texcoord, in.texindex);
    if (color.a == 0) {
        discard;
    }

    var lightColor = vec3f(0.0);
    for (var i: u32; i < lightData.activeLights; i++) {
        let diff = in.worldPosition - lightData.lights[i].position;
        let dist = sqrt(dot(diff, diff));
        let atten = max(0.0, 1.0 - dist / 8.0);
        lightColor += atten * atten * lightData.lights[i].color;
    }

    color = vec4f(0.5 * color.rgb + 0.5 * lightColor, color.a);

    return color;
}