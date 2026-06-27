# Array Morphisms

[![Chicken Scheme](https://img.shields.io/badge/Chicken-Scheme-orange.svg)](https://call-cc.org/)

A unified backend for numerical computing in Chicken Scheme, combining fusion-based lazy evaluation, Mathematics of Arrays (MoA) index transformations, and automatic memory reuse.

## Features

- **Lazy Evaluation**: Operations build expression trees that are materialized on demand
- **Zero-Copy Views**: Structural operations (reshape, transpose, slice) via MoA affine index functions
- **Memory Reuse**: Automatic buffer planning with graph coloring for optimal allocation
- **BLAS Integration**: Transparent dispatch to optimized linear algebra kernels
- **Type Safety**: Multiple element types (f64, f32, s64, s32, u32, u64)
- **Category-Theoretic Foundation**: Array morphisms as structure-preserving transformations

## Installation

```bash
# Install array-morphisms
chicken-install array-morphisms
```

Or clone from GitHub:

```bash
git clone https://github.com/iraikov/array-morphisms.git
cd array-morphisms
chicken-install .
```

## Quick Start

```scheme
(import array-morphisms-core
        array-morphisms-basic-ops
        array-morphisms-structural-ops
        array-morphisms-realization)

;; Create concrete arrays
(define x (morph-from-list '(1.0 2.0 3.0 4.0 5.0) '(5) 'f64))
(define y (morph-from-list '(2.0 4.0 6.0 8.0 10.0) '(5) 'f64))

;; Build lazy computation (no allocation yet!)
(define z (morph+ (morph-map (lambda (a) (* a a)) x) y))

;; Materialize when needed
(define result (realize z))  ; Returns concrete array
(morph->list result)          ; Convert to Scheme list

;; Chain structural operations (all zero-copy views)
(define matrix (morph-reshape x #(2 3)))   ; Reshape to 2x3
(define transposed (morph-transpose matrix)) ; Transpose
(define slice (morph-slice transposed '(0 0) '(2 2))) ; Extract submatrix
```

## Core Concepts

### Morphisms vs Arrays

In array-morphisms, computation is represented as **morphisms** - structure-preserving transformations between arrays. There are two types:

- **Concrete Arrays**: Materialized data with shape, dtype, and strides
- **Abstract Morphisms**: Deferred computations represented as expression trees

```scheme
;; Concrete array - data is stored
(define concrete (morph-from-list '(1.0 2.0 3.0) '(3) 'f64))

;; Abstract morphism - represents computation
(define abstract (morph+ concrete concrete))

;; Realization materializes the morphism
(define result (realize abstract))  ; Now concrete
```

### Index Functions

Array morphisms use **index functions** to describe transformations algebraically:

- **Affine Index Functions**: Pure transformations (reshape, transpose, slice)
- **Compute Index Functions**: Element-wise arithmetic operations
- **Window Index Functions**: Convolution and pooling operations
- **Reduction Index Functions**: Aggregate operations (sum, mean, max)

```scheme
;; Affine: stride-2 slice
(define downsampled (morph-slice x '(0) '(16) 2))

;; Compute: element-wise multiplication
(define scaled (morph* x (morph-from-list '(2.0) '(1) 'f64)))

;; Reduction: sum all elements
(define total (morph-reduce 'sum x))
```

### Zero-Copy Structural Operations

Structural operations manipulate array views without copying data:

```scheme
(define x (morph-from-list '(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0) '(8) 'f64))

;; Non-contiguous slice (stride 2)
(define strided (morph-slice x '(0) '(8) 2))  ; (0.0 2.0 4.0 6.0)

;; Reshape works even on non-contiguous views
(define as-2x2 (morph-reshape strided #(2 2)))  ; ((0.0 2.0) (4.0 6.0))

;; Transpose
(define transposed (morph-transpose as-2x2 '(1 0)))  ; ((0.0 4.0) (2.0 6.0))
```

## Basic Operations

### Array Creation

```scheme
(morph-from-list '(1.0 2.0 3.0) '(3) 'f64)  ; From list
(make-morphism data-vector shape 'f64)        ; From typed vector
```

### Arithmetic

```scheme
(morph+ a b)    (morph- a b)    (morph* a b)    (morph/ a b)
(morph-pow a b)

;; Unary operations
(morph-negate a)  (morph-abs a)  (morph-sqrt a)
(morph-exp a)     (morph-log a)  (morph-sin a)   (morph-cos a)
```

### Comparison

```scheme
(morph> a b)  (morph< a b)  (morph= a b)  (morph>= a b)  (morph<= a b)
;; Returns 1.0 for true, 0.0 for false
```

### Structural Operations

```scheme
;; Reshape (supports -1 for automatic dimension inference)
(morph-reshape m #(2 3))      ; Reshape to 2x3
(morph-reshape m '(2 -1))     ; Infer second dimension

;; Transpose
(morph-transpose m)           ; Reverse all axes
(morph-transpose m '(1 0))    ; 2D transpose
(morph-transpose m '(0 2 1))  ; Swap last two axes

;; Slice
(morph-slice m '(0) '(10))      ; Elements 0 to 9
(morph-slice m '(0) '(10) 2)    ; Every other element

;; Stack/Concat
(morph-stack (list m1 m2 m3) 0)   ; Stack along new axis
(morph-concat (list m1 m2) 0)      ; Concatenate along existing axis
```

### Functional Operations

```scheme
;; Map applies function element-wise
(morph-map (lambda (x) (* x x)) arr)

;; Reduce aggregates over specified axes
(morph-reduce 'sum arr)           ; Sum all elements
(morph-reduce 'mean arr '(0))     ; Mean along axis 0
(morph-reduce 'max arr '(1 2))    ; Max along axes 1 and 2

;; Fold and scan (batch operations)
(batch-fold fn init batched-m)
(batch-scan fn init batched-m)
```

## Memory Reuse with Execution Context

For repeated computations, use execution contexts to enable buffer reuse:

```scheme
(import array-morphisms-context)

;; Create context for memory planning
(define ctx (make-morphism-context))

;; Trace phase: record allocations
(realize/ctx ctx morphism)
(finalize-context! ctx)

;; Replay phase: reuse buffers
(reset-context! ctx)
(realize/ctx ctx morphism)  ; Uses pre-allocated buffers
```

## Type System

| Type    | Description           | Size     |
|---------|-----------------------|----------|
| `'f64`  | Double float          | 64-bit   |
| `'f32`  | Single float          | 32-bit   |
| `'s64`  | 64-bit signed int     | 64-bit   |
| `'s32`  | 32-bit signed int     | 32-bit   |
| `'u64`  | 64-bit unsigned int   | 64-bit   |
| `'u32`  | 32-bit unsigned int   | 32-bit   |

Type promotion rules:
- Mixed operations promote to the higher precision type
- Transcendental functions promote integers to floating point
- Reductions preserve dtype (mean promotes to float)

## Performance Tips

1. **Laziness is your friend** - Build expression trees, materialize once
2. **Zero-copy views** - Structural operations are essentially free
3. **Use contexts** - For repeated computations, enable buffer reuse
4. **Batch operations** - Process multiple arrays together efficiently

```scheme
;; Good: Chain operations, materialize once
(define result (realize (morph-sqrt (morph+ (morph* a b) c))))

;; Good: Use contexts for repeated inference
(define ctx (make-morphism-context))
(realize/ctx ctx model-output)  ; First run traces
(finalize-context! ctx)
;; ... later ...
(realize/ctx ctx model-output)  ; Reuses buffers

;; Bad: Materializing intermediate results
(define temp1 (realize (morph* a b)))
(define temp2 (realize (morph+ temp1 c)))
(define result (realize (morph-sqrt temp2)))
```

## Comparison with Fusion Arrays

| Feature              | Fusion Arrays        | Array Morphisms                 |
|----------------------|----------------------|----------------------------------|
| Core abstraction     | Fusion arrays        | Array morphisms                  |
| Structural ops       | Copy on non-contiguous | Zero-copy via MoA              |
| Memory reuse         | Manual               | Automatic (context-based)        |
| Index functions      | Hidden               | First-class, composable          |
| Batch operations     | Limited              | First-class combinators          |
| BLAS integration     | No                   | Yes (GEMM, GEMV, DOT)            |

## Examples

### Layer Normalization

```scheme
(define (layer-norm x eps)
  (let* ((mean (morph-reduce 'mean x '(0)))
         (centered (morph- x mean))
         (variance (morph-reduce 'mean 
                               (morph* centered centered) 
                               '(0)))
         (std (morph-sqrt (morph+ variance 
                                  (morph-from-list 
                                    (make-list (vector-ref 
                                                (get-morphism-shape x) 1) 
                                               eps)
                                    (list (vector-ref 
                                           (get-morphism-shape x) 1))
                                    'f64)))))
    (morph/ centered std)))
```

### Signal Downsampling Pipeline

```scheme
(define (downsample-pipeline signal)
  ;; Polyphase downsampling via composed slices
  (let* ((even (morph-slice signal '(0) (get-morphism-shape signal) 2))
         (quarter (morph-slice even '(0) (get-morphism-shape even) 2)))
    ;; Both slices are zero-copy views
    ;; Final realization computes in single pass
    (realize quarter)))
```

### Batched Matrix Operations

```scheme
(import array-morphisms-batch-ops)

;; Stack matrices into batch
(define batch (morph-stack (list m1 m2 m3) 0))

;; Apply operation to each batch element
(define doubled (batch-map 
                   (lambda (m) (morph-map (lambda (x) (* x 2)) m))
                   batch))

;; Reduce across batch dimension
(define summed (batch-reduce 'sum batch))
```

## Requirements

- CHICKEN Scheme 5.0+
- Dependencies: datatype, matchable, srfi-1, srfi-4, srfi-69
- Optional: BLAS library for accelerated linear algebra

## API Reference

See [CHICKEN Scheme Wiki](https://wiki.call-cc.org/) for full documentation.

Key modules:
- `array-morphisms-core` - Core data types and utilities
- `array-morphisms-basic-ops` - Arithmetic and transcendental operations
- `array-morphisms-structural-ops` - Reshape, transpose, slice, stack
- `array-morphisms-realization` - Materialization and execution
- `array-morphisms-context` - Memory reuse contexts
- `array-morphisms-batch-ops` - Batch operations and combinators

## License

LGPL-3

## Acknowledgments

- Inspired by the Mathematics of Arrays (MoA) formalism by Lenore Mullin
- Category-theoretic foundation from functional programming research
- Memory reuse patterns from stream fusion and buffer optimization literature
