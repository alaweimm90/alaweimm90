// WebGPU type definitions for TypeScript
// Based on the WebGPU specification

interface Navigator {
  readonly gpu?: GPU;
}

interface GPU {
  requestAdapter(options?: GPURequestAdapterOptions): Promise<GPUAdapter | null>;
}

interface GPURequestAdapterOptions {
  powerPreference?: 'low-power' | 'high-performance';
  forceFallbackAdapter?: boolean;
}

interface GPUAdapter {
  readonly features: GPUSupportedFeatures;
  readonly limits: GPUSupportedLimits;
  readonly info?: GPUAdapterInfo;
  
  requestDevice(descriptor?: GPUDeviceDescriptor): Promise<GPUDevice>;
}

interface GPUAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
}

interface GPUSupportedFeatures {
  has(feature: GPUFeatureName): boolean;
  values(): IterableIterator<GPUFeatureName>;
}

interface GPUSupportedLimits {
  readonly maxTextureDimension1D: number;
  readonly maxTextureDimension2D: number;
  readonly maxTextureDimension3D: number;
  readonly maxTextureArrayLayers: number;
  readonly maxBindGroups: number;
  readonly maxDynamicUniformBuffersPerPipelineLayout: number;
  readonly maxDynamicStorageBuffersPerPipelineLayout: number;
  readonly maxSampledTexturesPerShaderStage: number;
  readonly maxSamplersPerShaderStage: number;
  readonly maxStorageBuffersPerShaderStage: number;
  readonly maxStorageTexturesPerShaderStage: number;
  readonly maxUniformBuffersPerShaderStage: number;
  readonly maxUniformBufferBindingSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly minUniformBufferOffsetAlignment: number;
  readonly minStorageBufferOffsetAlignment: number;
  readonly maxVertexBuffers: number;
  readonly maxVertexAttributes: number;
  readonly maxVertexBufferArrayStride: number;
  readonly maxInterStageShaderComponents: number;
  readonly maxComputeWorkgroupStorageSize: number;
  readonly maxComputeInvocationsPerWorkgroup: number;
  readonly maxComputeWorkgroupSizeX: number;
  readonly maxComputeWorkgroupSizeY: number;
  readonly maxComputeWorkgroupSizeZ: number;
  readonly maxComputeWorkgroupsPerDimension: number;
}

type GPUFeatureName = 
  | 'depth-clip-control'
  | 'depth32float-stencil8'
  | 'texture-compression-bc'
  | 'texture-compression-etc2'
  | 'texture-compression-astc'
  | 'timestamp-query'
  | 'indirect-first-instance'
  | 'shader-f16'
  | 'rg11b10ufloat-renderable'
  | 'bgra8unorm-storage'
  | 'float32-filterable';

interface GPUDeviceDescriptor {
  label?: string;
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: Record<string, number>;
}

interface GPUDevice {
  readonly features: GPUSupportedFeatures;
  readonly limits: GPUSupportedLimits;
  readonly queue: GPUQueue;
  readonly lost: Promise<GPUDeviceLostInfo>;

  destroy(): void;
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
  createTexture(descriptor: GPUTextureDescriptor): GPUTexture;
  createSampler(descriptor?: GPUSamplerDescriptor): GPUSampler;
  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout;
  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout;
  createShaderModule(descriptor: GPUShaderModuleDescriptor): GPUShaderModule;
  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline;
  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline;
  createBindGroup(descriptor: GPUBindGroupDescriptor): GPUBindGroup;
  createQuerySet(descriptor: GPUQuerySetDescriptor): GPUQuerySet;
  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder;
}

interface GPUDeviceLostInfo {
  readonly reason: 'unknown' | 'destroyed';
  readonly message: string;
}

interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: GPUBufferUsageFlags;
  mappedAtCreation?: boolean;
}

type GPUBufferUsageFlags = number;

interface GPUBufferUsage {
  readonly MAP_READ: 0x0001;
  readonly MAP_WRITE: 0x0002;
  readonly COPY_SRC: 0x0004;
  readonly COPY_DST: 0x0008;
  readonly INDEX: 0x0010;
  readonly VERTEX: 0x0020;
  readonly UNIFORM: 0x0040;
  readonly STORAGE: 0x0080;
  readonly INDIRECT: 0x0100;
  readonly QUERY_RESOLVE: 0x0200;
}

declare const GPUBufferUsage: GPUBufferUsage;

interface GPUBuffer {
  readonly label: string;
  readonly size: number;
  readonly usage: GPUBufferUsageFlags;
  readonly mapState: GPUBufferMapState;

  mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

type GPUBufferMapState = 'unmapped' | 'pending' | 'mapped';
type GPUMapModeFlags = number;

interface GPUMapMode {
  readonly READ: 0x0001;
  readonly WRITE: 0x0002;
}

declare const GPUMapMode: GPUMapMode;

interface GPUTextureDescriptor {
  label?: string;
  size: GPUExtent3D;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: GPUTextureDimension;
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
  viewFormats?: GPUTextureFormat[];
}

type GPUExtent3D = [number, number, number] | { width: number; height: number; depthOrArrayLayers: number };
type GPUTextureDimension = '1d' | '2d' | '3d';
type GPUTextureFormat = string;
type GPUTextureUsageFlags = number;

interface GPUTexture {
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly depthOrArrayLayers: number;
  readonly mipLevelCount: number;
  readonly sampleCount: number;
  readonly dimension: GPUTextureDimension;
  readonly format: GPUTextureFormat;
  readonly usage: GPUTextureUsageFlags;

  createView(descriptor?: GPUTextureViewDescriptor): GPUTextureView;
  destroy(): void;
}

interface GPUTextureViewDescriptor {
  label?: string;
  format?: GPUTextureFormat;
  dimension?: GPUTextureViewDimension;
  aspect?: GPUTextureAspect;
  baseMipLevel?: number;
  mipLevelCount?: number;
  baseArrayLayer?: number;
  arrayLayerCount?: number;
}

type GPUTextureViewDimension = '1d' | '2d' | '2d-array' | 'cube' | 'cube-array' | '3d';
type GPUTextureAspect = 'all' | 'stencil-only' | 'depth-only';

interface GPUTextureView {
  readonly label: string;
}

interface GPUSamplerDescriptor {
  label?: string;
  addressModeU?: GPUAddressMode;
  addressModeV?: GPUAddressMode;
  addressModeW?: GPUAddressMode;
  magFilter?: GPUFilterMode;
  minFilter?: GPUFilterMode;
  mipmapFilter?: GPUMipmapFilterMode;
  lodMinClamp?: number;
  lodMaxClamp?: number;
  compare?: GPUCompareFunction;
  maxAnisotropy?: number;
}

type GPUAddressMode = 'clamp-to-edge' | 'repeat' | 'mirror-repeat';
type GPUFilterMode = 'nearest' | 'linear';
type GPUMipmapFilterMode = 'nearest' | 'linear';
type GPUCompareFunction = 'never' | 'less' | 'equal' | 'less-equal' | 'greater' | 'not-equal' | 'greater-equal' | 'always';

interface GPUSampler {
  readonly label: string;
}

interface GPUBindGroupLayoutDescriptor {
  label?: string;
  entries: GPUBindGroupLayoutEntry[];
}

interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: GPUShaderStageFlags;
  buffer?: GPUBufferBindingLayout;
  sampler?: GPUSamplerBindingLayout;
  texture?: GPUTextureBindingLayout;
  storageTexture?: GPUStorageTextureBindingLayout;
}

type GPUShaderStageFlags = number;

interface GPUShaderStage {
  readonly VERTEX: 0x1;
  readonly FRAGMENT: 0x2;
  readonly COMPUTE: 0x4;
}

declare const GPUShaderStage: GPUShaderStage;

interface GPUBufferBindingLayout {
  type?: 'uniform' | 'storage' | 'read-only-storage';
  hasDynamicOffset?: boolean;
  minBindingSize?: number;
}

interface GPUSamplerBindingLayout {
  type?: 'filtering' | 'non-filtering' | 'comparison';
}

interface GPUTextureBindingLayout {
  sampleType?: 'float' | 'unfilterable-float' | 'depth' | 'sint' | 'uint';
  viewDimension?: GPUTextureViewDimension;
  multisampled?: boolean;
}

interface GPUStorageTextureBindingLayout {
  access?: 'write-only';
  format: GPUTextureFormat;
  viewDimension?: GPUTextureViewDimension;
}

interface GPUBindGroupLayout {
  readonly label: string;
}

interface GPUPipelineLayoutDescriptor {
  label?: string;
  bindGroupLayouts: GPUBindGroupLayout[];
}

interface GPUPipelineLayout {
  readonly label: string;
}

interface GPUShaderModuleDescriptor {
  label?: string;
  code: string;
  sourceMap?: object;
}

interface GPUShaderModule {
  readonly label: string;
}

interface GPUComputePipelineDescriptor {
  label?: string;
  layout: GPUPipelineLayout | 'auto';
  compute: GPUProgrammableStage;
}

interface GPUProgrammableStage {
  module: GPUShaderModule;
  entryPoint: string;
  constants?: Record<string, number>;
}

interface GPUComputePipeline {
  readonly label: string;
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

interface GPURenderPipelineDescriptor {
  label?: string;
  layout: GPUPipelineLayout | 'auto';
  vertex: GPUVertexState;
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
  multisample?: GPUMultisampleState;
  fragment?: GPUFragmentState;
}

interface GPUVertexState extends GPUProgrammableStage {
  buffers?: GPUVertexBufferLayout[];
}

interface GPUPrimitiveState {
  topology?: GPUPrimitiveTopology;
  stripIndexFormat?: GPUIndexFormat;
  frontFace?: GPUFrontFace;
  cullMode?: GPUCullMode;
  unclippedDepth?: boolean;
}

type GPUPrimitiveTopology = 'point-list' | 'line-list' | 'line-strip' | 'triangle-list' | 'triangle-strip';
type GPUIndexFormat = 'uint16' | 'uint32';
type GPUFrontFace = 'ccw' | 'cw';
type GPUCullMode = 'none' | 'front' | 'back';

interface GPUDepthStencilState {
  format: GPUTextureFormat;
  depthWriteEnabled?: boolean;
  depthCompare?: GPUCompareFunction;
  stencilFront?: GPUStencilFaceState;
  stencilBack?: GPUStencilFaceState;
  stencilReadMask?: number;
  stencilWriteMask?: number;
  depthBias?: number;
  depthBiasSlopeScale?: number;
  depthBiasClamp?: number;
}

interface GPUStencilFaceState {
  compare?: GPUCompareFunction;
  failOp?: GPUStencilOperation;
  depthFailOp?: GPUStencilOperation;
  passOp?: GPUStencilOperation;
}

type GPUStencilOperation = 'keep' | 'zero' | 'replace' | 'invert' | 'increment-clamp' | 'decrement-clamp' | 'increment-wrap' | 'decrement-wrap';

interface GPUMultisampleState {
  count?: number;
  mask?: number;
  alphaToCoverageEnabled?: boolean;
}

interface GPUFragmentState extends GPUProgrammableStage {
  targets: GPUColorTargetState[];
}

interface GPUColorTargetState {
  format: GPUTextureFormat;
  blend?: GPUBlendState;
  writeMask?: GPUColorWriteFlags;
}

interface GPUBlendState {
  color: GPUBlendComponent;
  alpha: GPUBlendComponent;
}

interface GPUBlendComponent {
  operation?: GPUBlendOperation;
  srcFactor?: GPUBlendFactor;
  dstFactor?: GPUBlendFactor;
}

type GPUBlendOperation = 'add' | 'subtract' | 'reverse-subtract' | 'min' | 'max';
type GPUBlendFactor = 'zero' | 'one' | 'src' | 'one-minus-src' | 'src-alpha' | 'one-minus-src-alpha' | 'dst' | 'one-minus-dst' | 'dst-alpha' | 'one-minus-dst-alpha' | 'src-alpha-saturated' | 'constant' | 'one-minus-constant';
type GPUColorWriteFlags = number;

interface GPUVertexBufferLayout {
  arrayStride: number;
  stepMode?: GPUVertexStepMode;
  attributes: GPUVertexAttribute[];
}

type GPUVertexStepMode = 'vertex' | 'instance';

interface GPUVertexAttribute {
  format: GPUVertexFormat;
  offset: number;
  shaderLocation: number;
}

type GPUVertexFormat = string;

interface GPURenderPipeline {
  readonly label: string;
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

interface GPUBindGroupDescriptor {
  label?: string;
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
}

interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindingResource;
}

type GPUBindingResource = GPUSampler | GPUTextureView | GPUBufferBinding;

interface GPUBufferBinding {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

interface GPUBindGroup {
  readonly label: string;
}

interface GPUQuerySetDescriptor {
  label?: string;
  type: GPUQueryType;
  count: number;
}

type GPUQueryType = 'occlusion' | 'timestamp';

interface GPUQuerySet {
  readonly label: string;
  readonly type: GPUQueryType;
  readonly count: number;
  destroy(): void;
}

interface GPUCommandEncoderDescriptor {
  label?: string;
}

interface GPUCommandEncoder {
  readonly label: string;
  beginRenderPass(descriptor: GPURenderPassDescriptor): GPURenderPassEncoder;
  beginComputePass(descriptor?: GPUComputePassDescriptor): GPUComputePassEncoder;
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  copyBufferToTexture(source: GPUImageCopyBuffer, destination: GPUImageCopyTexture, copySize: GPUExtent3D): void;
  copyTextureToBuffer(source: GPUImageCopyTexture, destination: GPUImageCopyBuffer, copySize: GPUExtent3D): void;
  copyTextureToTexture(source: GPUImageCopyTexture, destination: GPUImageCopyTexture, copySize: GPUExtent3D): void;
  resolveQuerySet(querySet: GPUQuerySet, firstQuery: number, queryCount: number, destination: GPUBuffer, destinationOffset: number): void;
  finish(descriptor?: GPUCommandBufferDescriptor): GPUCommandBuffer;
}

interface GPURenderPassDescriptor {
  label?: string;
  colorAttachments: (GPURenderPassColorAttachment | null)[];
  depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
  occlusionQuerySet?: GPUQuerySet;
  timestampWrites?: GPURenderPassTimestampWrites;
}

interface GPURenderPassColorAttachment {
  view: GPUTextureView;
  resolveTarget?: GPUTextureView;
  clearValue?: GPUColor;
  loadOp: GPULoadOp;
  storeOp: GPUStoreOp;
}

type GPUColor = [number, number, number, number] | { r: number; g: number; b: number; a: number };
type GPULoadOp = 'load' | 'clear';
type GPUStoreOp = 'store' | 'discard';

interface GPURenderPassDepthStencilAttachment {
  view: GPUTextureView;
  depthClearValue?: number;
  depthLoadOp?: GPULoadOp;
  depthStoreOp?: GPUStoreOp;
  depthReadOnly?: boolean;
  stencilClearValue?: number;
  stencilLoadOp?: GPULoadOp;
  stencilStoreOp?: GPUStoreOp;
  stencilReadOnly?: boolean;
}

interface GPURenderPassTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

interface GPUComputePassDescriptor {
  label?: string;
  timestampWrites?: GPUComputePassTimestampWrites;
}

interface GPUComputePassTimestampWrites {
  querySet: GPUQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

interface GPURenderPassEncoder {
  readonly label: string;
  setViewport(x: number, y: number, width: number, height: number, minDepth: number, maxDepth: number): void;
  setScissorRect(x: number, y: number, width: number, height: number): void;
  setPipeline(pipeline: GPURenderPipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  setIndexBuffer(buffer: GPUBuffer, format: GPUIndexFormat, offset?: number, size?: number): void;
  setVertexBuffer(slot: number, buffer: GPUBuffer, offset?: number, size?: number): void;
  draw(vertexCount: number, instanceCount?: number, firstVertex?: number, firstInstance?: number): void;
  drawIndexed(indexCount: number, instanceCount?: number, firstIndex?: number, baseVertex?: number, firstInstance?: number): void;
  drawIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): void;
  drawIndexedIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): void;
  end(): void;
}

interface GPUComputePassEncoder {
  readonly label: string;
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup, dynamicOffsets?: number[]): void;
  dispatchWorkgroups(workgroupCountX: number, workgroupCountY?: number, workgroupCountZ?: number): void;
  dispatchWorkgroupsIndirect(indirectBuffer: GPUBuffer, indirectOffset: number): void;
  end(): void;
}

interface GPUImageCopyBuffer {
  buffer: GPUBuffer;
  offset?: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}

interface GPUImageCopyTexture {
  texture: GPUTexture;
  mipLevel?: number;
  origin?: GPUOrigin3D;
  aspect?: GPUTextureAspect;
}

type GPUOrigin3D = [number, number, number] | { x: number; y: number; z: number };

interface GPUCommandBufferDescriptor {
  label?: string;
}

interface GPUCommandBuffer {
  readonly label: string;
}

interface GPUQueue {
  readonly label: string;
  submit(commandBuffers: GPUCommandBuffer[]): void;
  writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: ArrayBuffer | ArrayBufferView, dataOffset?: number, size?: number): void;
  writeTexture(destination: GPUImageCopyTexture, data: ArrayBuffer | ArrayBufferView, dataLayout: GPUImageDataLayout, size: GPUExtent3D): void;
  onSubmittedWorkDone(): Promise<void>;
}

interface GPUImageDataLayout {
  offset?: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}