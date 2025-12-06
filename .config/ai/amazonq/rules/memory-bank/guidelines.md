# Development Guidelines & Standards

## Code Quality Standards

### Naming Conventions

- **Constants**: Use SCREAMING_SNAKE_CASE for module-level constants and configuration values
  ```python
  AYRSHARE_BLOCK_IDS = ["cbd52c2a-06d2-43ed-9560-6576cc163283", ...]
  ```
- **Functions**: Use camelCase for JavaScript/TypeScript, snake_case for Python
  ```typescript
  export function findLastIndex<T = any>(arr: T[], criterion: (item: T) => boolean): number;
  ```
- **Variables**: Use descriptive names that clearly indicate purpose and scope
  ```cpp
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  ```

### Type Safety & Generics

- **Generic Functions**: Use meaningful type parameter names and constraints
  ```typescript
  export function findLast<T>(arr: T[], criterion: (item: T) => any): T | undefined;
  ```
- **Explicit Return Types**: Always specify return types for public functions
- **Null Safety**: Use optional types and undefined returns where appropriate

### Code Organization Patterns

#### Module Structure

- **Clean Exports**: Use explicit named exports for better tree-shaking
  ```typescript
  export { WebSearchButton } from './WebSearchButton';
  export { Combobox, ComboboxButton, ComboboxInput, ComboboxOption, ComboboxOptions };
  ```
- **Index Files**: Serve as clean aggregation points for module exports
- **Single Responsibility**: Each module should have a focused, well-defined purpose

#### Performance-Critical Code

- **Hardware Optimization**: Use platform-specific optimizations with fallbacks
  ```cpp
  #if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
      // Optimized ARM NEON implementation
  #else
      // Generic fallback implementation
  #endif
  ```
- **SIMD Instructions**: Leverage vectorized operations for computational workloads
- **Memory Layout**: Optimize data structures for cache efficiency and alignment

## Architectural Patterns

### Error Handling & Robustness

- **Graceful Degradation**: Always provide fallback implementations for platform-specific code
- **Assertion Usage**: Use assertions for invariant checking in performance-critical paths
  ```cpp
  assert(n % qk == 0);
  assert(nr % 4 == 0);
  ```
- **Parameter Validation**: Validate inputs at function boundaries

### Algorithm Implementation

- **Loop Optimization**: Structure loops for optimal performance and readability
  ```typescript
  for (let i = arr.length - 1; i >= 0; i--) {
    if (criterion(arr[i])) {
      return arr[i];
    }
  }
  ```
- **Early Returns**: Use early returns to reduce nesting and improve readability
- **Const Correctness**: Use const/readonly modifiers where data shouldn't change

### Hardware-Aware Programming

- **Platform Detection**: Use compile-time feature detection for optimal code paths
- **Vectorization**: Implement SIMD operations for mathematical computations
- **Memory Management**: Use RESTRICT keywords and alignment hints for performance

## Documentation Standards

### Code Comments

- **Minimal Comments**: Write self-documenting code that reduces need for comments
- **Algorithm Explanation**: Comment complex mathematical or algorithmic sections
- **Platform Notes**: Document platform-specific behavior and requirements

### Function Documentation

- **Type Annotations**: Use comprehensive type annotations as primary documentation
- **Parameter Descriptions**: Clear parameter names that indicate expected input
- **Return Value Clarity**: Explicit return types that communicate function behavior

## Testing & Validation Patterns

### Defensive Programming

- **Boundary Checking**: Validate array bounds and numerical limits
- **Type Guards**: Use type checking for runtime safety in dynamic contexts
- **Fallback Mechanisms**: Implement graceful fallbacks for unsupported features

### Performance Considerations

- **Algorithmic Complexity**: Choose algorithms appropriate for expected data sizes
- **Memory Efficiency**: Minimize allocations in hot code paths
- **Cache Optimization**: Structure data access patterns for cache efficiency

## Integration Patterns

### Module Boundaries

- **Clean Interfaces**: Design clear, minimal APIs between modules
- **Dependency Management**: Minimize coupling between components
- **Configuration Externalization**: Keep configuration separate from implementation

### Cross-Platform Compatibility

- **Feature Detection**: Use runtime or compile-time feature detection
- **Abstraction Layers**: Provide consistent APIs across different platforms
- **Graceful Degradation**: Ensure functionality on all supported platforms

## Scientific Computing Specific Guidelines

### Numerical Stability

- **Precision Awareness**: Choose appropriate data types for numerical accuracy
- **Algorithm Selection**: Use numerically stable algorithms for mathematical operations
- **Error Propagation**: Consider floating-point precision in computational chains

### Hardware Utilization

- **GPU Acceleration**: Design algorithms to leverage parallel processing capabilities
- **Memory Bandwidth**: Optimize memory access patterns for high-performance computing
- **Instruction-Level Parallelism**: Use SIMD instructions for vectorizable operations

### Research Code Quality

- **Reproducibility**: Write code that produces consistent, reproducible results
- **Modularity**: Design components that can be easily tested and validated
- **Performance Measurement**: Include benchmarking capabilities for optimization validation
