# Array Morphisms

[![Chicken Scheme](https://img.shields.io/badge/Chicken-Scheme-orange.svg)](https://call-cc.org/)

A unified backend for numerical computing in Chicken Scheme, combining fusion-based lazy evaluation, Mathematics of Arrays (MoA) index transformations, and automatic memory reuse.

## Features

- **Lazy Evaluation**: Operations build expression trees that are materialized on demand
- **Zero-Copy Views**: Structural operations (reshape, transpose, slice) via MoA affine index functions
- **Memory Reuse**: Automatic buffer planning with graph coloring for optimal allocation
- **BLAS Integration**: Transparent dispatch to optimized linear algebra kernels
- **Element-wise Kernel Backends**: Pluggable whole-array kernels for activations, binary arithmetic, reductions and copies, with results identical to the Scheme code they replace
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

`(morph-cast m dtype)` converts an array to another dtype element by
element: a conversion to `'f32` rounds to single precision, and a
conversion to an integer dtype truncates.

In a compiled SSA training graph, each parameter's gradient has the
parameter's dtype. When an `'f32` parameter meets `'f64` values, the
backward pass computes in `'f64` as promotion requires, and the gradient
is cast back to `'f32` at the end.

`ssa-vjp` returns one gradient per parameter, in the order of the
parameters. A parameter that no gradient reaches gets zeros of its shape
and dtype; this includes a parameter reached only through a max reduction,
whose derivative is not computed. A `morph-map` traces and replays like any
element-wise op, but its procedure has no derivative: `ssa-vjp` raises an
error when the input of a map depends on a trainable parameter.

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

## BLAS Backends

Matmul, matvec, dot, axpy, and conv2d are dispatched through a pluggable `blas-backend`
(`array-morphisms-blas-exec`). Four tiers are available:

| Tier | Package | Dependencies | Default? |
|---|---|---|---|
| Pure Scheme | built in | none | fallback only |
| microBLAS | built in (`array-morphisms-micro-blas-backend`) | none (vendored, header-only) | yes, auto-registered |
| System BLAS | separate egg: [`array-morphisms-blas`](https://github.com/iraikov/array-morphisms-blas) | the `blas` egg + a system BLAS library | opt-in |
| crunch | separate egg: `array-morphisms-crunch` (`make-crunch-blas-backend`) | the `crunch` egg (CHICKEN 6) | opt-in |

The dependency-free microBLAS backend is registered automatically at load time if nothing else
has registered a backend first, so `array-morphisms` alone never requires a system BLAS library.
For maximum performance, install the companion `array-morphisms-blas` egg and register it
explicitly (it overrides the default):

```scheme
(import array-morphisms-blas-egg-backend)
(register-blas-backend! (make-blas-egg-backend))
```

The crunch tier compiles its GEMM and convolution kernels from Scheme to C
with CHICKEN 6's crunch compiler and splits large GEMMs across threads. It
needs no C library beyond libm and pthreads.

### Prepared GEMM

`execute-blas-gemm/into!` and `execute-blas-gemm-strided/into!` check their
operands (dtype, rank, layout, size threshold) on every call. Code that
multiplies arrays of the same shapes, strides and dtype many times can do
those checks once:

```scheme
(define plan (prepare-blas-gemm/into A B))   ; or prepare-blas-gemm-strided/into
(execute-gemm-plan/into! plan A B result-data)
```

A plan records the backend kernel to call and its arguments. It is used
only while BLAS is enabled and the backend it was made for is still
registered; otherwise `execute-gemm-plan/into!` falls back to the
corresponding `execute-blas-*` procedure. `gemm-plan-uses-kernel?` tells
whether a plan calls a backend kernel. Compiled SSA replay plans prepare
their GEMM segops this way when they are compiled.

## Element-wise Kernel Backends

Element-wise operations, reductions and strided copies can be handed to a
pluggable *activation backend* (`array-morphisms-activation-exec`), which
is registered separately from the BLAS backend. A backend holds whole-array
kernels, one per op and dtype (f32 or f64), for:

- the activations `relu`, `sigmoid` and `tanh`, and the derivative ops
  `relu-deriv`, `sigmoid-deriv` and `tanh-deriv` that the backward pass emits;
- the binary ops `add`, `sub`, `mul` and `div`;
- reductions of row-major 2-D arrays over axis 0 or 1;
- copies of strided views into row-major order.

When an SSA replay plan is compiled, each element-wise binding is lowered
to a segop (see below). A binding that applies one of these ops directly to
row-major operands of the output's shape gets the `segop/activation-unary`
or `segop/activation-binary` executor, which uses the backend's kernel for
its op and dtype at execution time.
Reductions and strided copies use their kernels wherever arrays are
realized, not only during replay. When no backend is registered, no kernel
exists for that op and dtype, or the operand and output dtypes differ, the
same Scheme code runs as before.
The kernels compute exactly what that Scheme code computes, so registering
a backend changes speed, not results. Bindings produced by the element-wise
fusion pass have a composed body and never use a kernel.

The fusion pass does not change results either. A fused binding rounds
each intermediate value to its dtype, as storing it in an array would
(`(cast f32 ...)` in its body), so an `'f32` chain computes exactly what
the separate bindings compute. The parameter `segop-fusion`
(`array-morphisms-ssa`) turns the pass off: `(segop-fusion #f)` globally,
or `(parameterize ((segop-fusion #f)) ...)` around `ssa-vjp`.

No activation backend is registered by default. The `array-morphisms-crunch`
egg provides two backends; the threaded one splits large arrays across
threads:

```scheme
(import array-morphisms-activation-exec
        array-morphisms-crunch-activations)
(register-activation-backend! (make-crunch-threaded-activation-backend))
```

The activation op names are registered by default. The binary op names are
registered by the backend constructors, so until then `add`, `sub`, `mul`
and `div` use the ordinary `segop/flat-binary` executor. Other names can be
registered with `register-activation-op!` and `register-binary-op!`.
Registering a name only selects the executor, and only for ops that have an
entry in the scalar-op table. For a new activation to appear in
differentiable SSA graphs it also needs a table entry, a `morph-<op>`
constructor, a VJP rule in `ssa-vjp` and a case in `rebuild-morphism`.

## Segmented Operations

Between the SSA program and its replay plan, element-wise bindings are
represented as *segmented operations* (segops,
`array-morphisms-segop`). A segop has:

- an iteration space, a vector of extents;
- operands, each an array with its layout (dtype, shape, strides, offset)
  and an *index map* from iteration coordinates to the array's index;
- a *body*, a symbolic expression computing one output element, such as
  `(relu (in 0))` or `(add (mul (in 0) (in 1)) (in 2))`; `(cast f32 e)`
  rounds a value as storing it in an f32 array would;
- the Scheme function it was lowered from, used to execute it on the host.

There are three kinds: `seg-map` (element-wise maps, possibly with several
outputs), `seg-red` (reductions, with a combine spec giving the operator,
neutral element, accumulator precision and summation order) and `seg-lib`
(named library operations such as convolutions). Index maps cover
identities, broadcasts, transposes, slices, reshapes (with `floordiv` and
`mod`) and padded reads (with a guard and a fill value). Because bodies and
maps are data, a code generator can read them, which the Scheme functions
inside the older instructions did not allow.

The ops a body can name come from the *scalar-op table*
(`scalar-ops-table.scm`, module `array-morphisms-scalar-ops-table`). Each
entry gives an op's parameters and formula. The combiners of
`array-morphisms-basic-ops` are generated from the same entries
(`array-morphisms-scalar-ops`), and `array-morphisms-crunch` reads the
table at compile time to generate its kernels, so all of them compute the
same formula.

`segop-interpret` executes any `seg-map` or `seg-red` directly from its
definition. It is slow and is meant for testing.

When a replay plan is compiled, every binding except convolutions and
zero-copy views is lowered to a segop:

- element-wise ops become a `seg-map` whose inputs are read through
  broadcast maps and their own strides, so transposed views are read in
  place;
- reductions become a `seg-red`, whose combine spec records how the
  reduction loop rounds: f32 sums over axis 0 round after every step, sums
  over axis 1 accumulate in double precision, and so on;
- a matrix product `A[M,K] B[K,N]` becomes a `seg-red` over `(M N K)` with
  the body `(mul (in 0) (in 1))`; transposed operands are described by their
  strides. Its combine spec sums in the BLAS backend's order when the
  prepared plan calls a kernel, and otherwise in increasing k with a double
  accumulator, as the Scheme loop does. An adjacent element-wise consumer
  (an activation, or a bias add) of the same shape and dtype is fused as
  the reduction's finalize, applied to the stored product
  (`(sigmoid (cast f32 (acc)))`);
- materialized views (such as a reshape of a transposed array) become a
  gather, a `seg-map` with the body `(in 0)`;
- the remaining index functions run on the host through a `seg-lib`.

Each segop is offered to the segop backends in turn; the first that accepts
it supplies an executor, and the binding becomes an `ri-segop` instruction. The backends in use are the
value of the parameter `segop-backends`:

```scheme
(import array-morphisms-segop)
(segop-backends)                      ; the backends, in the order tried
(register-segop-backend! b)           ; try b before the others
(parameterize ((segop-backends (list b)))
  ...)                                ; use only b within this extent
```

Two backends are always present. `registry` handles the patterns that use
the BLAS backend (matrix products, through prepared GEMM plans) and the
activation backend's kernels (element-wise, reduction over axis 0 or 1 of a
row-major matrix, and strided copy), and `scheme` handles the rest with
Scheme loops; as a last resort it evaluates a segop with the reference
interpreter. Replay results are the same as those of the instructions they
replaced, bit for bit.

The timing profile (`replay-timing-results`) and `replay-plan-stats` name
each segop instruction by its pattern:

- element-wise: `segop/activation-unary`, `segop/activation-binary`,
  `segop/flat-unary`, `segop/flat-binary`, `segop/bias-broadcast`,
  `segop/flat-ternary`, `segop/flat-broadcast`, and `segop/strided-map` for
  any other layout;
- reductions: `segop/reduce-1d`, `segop/reduce-axis0`, `segop/reduce-axis1`
  and `segop/reduce-general`;
- matrix products: `segop/gemm` (row-major operands), `segop/gemm-strided`
  (transposed or strided operands) and `segop/gemm-epilogue` (with a fused
  consumer);
- `segop/gather` for materialized views, `segop/host-index-fn` for index
  functions run on the host, and `segop/interpreted`.

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

- CHICKEN Scheme 6.0+
- Dependencies: datatype, matchable, random-mtzig, srfi-1, srfi-4, srfi-69
- Optional companion eggs:
  - `array-morphisms-blas`: system-BLAS backend (needs a BLAS library)
  - `array-morphisms-crunch`: crunch-compiled BLAS, convolution and element-wise kernels (needs the `crunch` egg)

## API Reference

See [CHICKEN Scheme Wiki](https://wiki.call-cc.org/) for full documentation.

Key modules:
- `array-morphisms-core` - Core data types and utilities
- `array-morphisms-scalar-ops-table` - The scalar-op table (op formulas, as data)
- `array-morphisms-scalar-ops` - Combiners and expressions of the scalar ops
- `array-morphisms-segop` - Segmented operations: index maps, bodies, interpreter, backends
- `array-morphisms-basic-ops` - Arithmetic and transcendental operations
- `array-morphisms-structural-ops` - Reshape, transpose, slice, stack
- `array-morphisms-realization` - Materialization and execution
- `array-morphisms-context` - Memory reuse contexts
- `array-morphisms-batch-ops` - Batch operations and combinators
- `array-morphisms-blas-exec` - BLAS backend records, registration and dispatch
- `array-morphisms-micro-blas-backend` - Default dependency-free BLAS backend
- `array-morphisms-activation-exec` - Element-wise kernel backend records and registration
- `array-morphisms-grad` - Reverse-mode automatic differentiation over morphisms
- `array-morphisms-grad-check` - Numerical gradient checking
- `array-morphisms-ssa` - SSA compilation of training graphs, VJP and replay

## License

LGPL-3

## Acknowledgments

- Inspired by the Mathematics of Arrays (MoA) formalism by Lenore Mullin
- Category-theoretic foundation from functional programming research
- Memory reuse patterns from stream fusion and buffer optimization literature
