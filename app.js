const GRID_SIZE = 32;

async function webGPU() {
    const canvas = document.getElementById('canvas');

    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported on this browser!");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: canvasFormat,
    });
    const encoder = device.createCommandEncoder();
    const cellShaderModule = device.createShaderModule({
        label: "Cell Shader",
        code: `
        @group(0) @binding(0) var<uniform> grid: vec2f;
        @group(0) @binding(1) var<storage> cellState: array<u32>;

        struct VertexInput {
            @location(0) pos: vec2f,
            @builtin(instance_index) instance: u32,
        }

        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) cell: vec2f,
        }

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
            let i = f32(input.instance);
            let cell = vec2f(i % grid.x, floor(i / grid.x));
            let cellOffset = cell / grid * 2;
            let gridPos = (input.pos + 1) / grid - 1 + cellOffset;

            var output: VertexOutput;
            output.pos = vec4f(gridPos, 0, 1);
            output.cell = cell;
            return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            let c = input.cell / grid;
            return vec4f(c, 1-c.y, 1);
        }
            `
    });

    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
        label: "Grid Uniforms",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    const vertices = new Float32Array([
        -0.8, -0.8,
        -0.8, 0.8,
        0.8, 0.8,

        -0.8, -0.8,
        0.8, -0.8,
        0.8, 0.8,
    ]);
    const vertexBuffer = device.createBuffer({
        label: "Cell Vertices",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);
    const vertexBufferLayout = {
        arrayStride: 8,
        attributes: [{
            format: "float32x2",
            offset: 0,
            shaderLocation: 0,
        }],
    };
    const cellPipeline = device.createRenderPipeline({
        label: "Cell pipeline",
        layout: "auto",
        vertex: {
            module: cellShaderModule,
            entryPoint: "vertexMain",
            buffers: [vertexBufferLayout]
        },
        fragment: {
            module: cellShaderModule,
            entryPoint: "fragmentMain",
            targets: [{
                format: canvasFormat
            }]
        }
    });

    const bindGroup = device.createBindGroup({
        label: "Cell Renderer bind group",
        layout: cellPipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: uniformBuffer }
        }],
    });
    const cellStateArray = new Uint32Array(GRID_SIZE, GRID_SIZE);
    const cellStateStorage = device.createBuffer({
        label: "Cell State",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DIST,
    });
    for (let i = 0; i < cellStateArray.length; i += 3) {
        cellStateArray[i] = 1;
    }
    device.queue.writeBuffer(cellStateStorage, 0, cellStateArray);

    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: {r: 0.32, g: 0.68, b: 0.67, a: 0},
            storeOp: "store",
        }]
    });
    pass.setPipeline(cellPipeline);
    pass.setVertexBuffer(0, vertexBuffer);

    pass.setBindGroup(0, bindGroup);

    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

document.addEventListener("DOMContentLoaded", webGPU);
