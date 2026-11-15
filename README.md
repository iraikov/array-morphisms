# Array Morphisms

[![Chicken Scheme](https://img.shields.io/badge/Chicken-Scheme-orange.svg)](https://call-cc.org/)

A CHICKEN Scheme library for numerical arrays. Operations build expressions that are computed only when their result is needed. Reshapes, transposes and slices are views that copy nothing; they are described by the index functions of the Mathematics of Arrays (MoA). Computations that run many times reuse their buffers.

## Features

- **Lazy evaluation**: operations build expression trees, which are computed when you call `realize`.
- **Views without copies**: reshape, transpose and slice change how an array is read, not its data.
- **Buffer reuse**: a computation that runs many times plans its buffers once and reuses them.
- **BLAS**: matrix products and convolutions run on a pluggable BLAS backend.
- **Kernel backends**: pluggable whole-array kernels for activations, arithmetic, reductions and copies, which give the same results as the Scheme loops they replace.
- **Scans**: cumulative sums, products, maxima and minima along an axis, and scans with your own combining expression; each value equals that of a plain loop.
- **Frame lifting**: apply a function to every cell of a frame (every example of a batch, or every attention head) as one computation; each cell gets exactly the values it would get on its own.
- **Element types**: f64, f32, s64 and s32.
- **Category-theoretic model**: array computations as morphisms, maps that preserve the structure of arrays.

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

;; Arrays with data
(define x (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) '(6) 'f64))
(define y (morph-from-list '(2.0 4.0 6.0 8.0 10.0 12.0) '(6) 'f64))

;; An expression: nothing is computed or allocated yet
(define z (morph+ (morph-map (lambda (a) (* a a)) x) y))

;; Compute it
(define result (realize z))   ; an array with data
(morph->list result)          ; => (3.0 8.0 15.0 24.0 35.0 48.0)

;; Views: none of these copies data
(define matrix (morph-reshape x #(2 3)))                ; 2 x 3
(define transposed (morph-transpose matrix))            ; 3 x 2
(define corner (morph-slice transposed '(0 0) '(2 2)))  ; top-left 2 x 2
```

## Core Concepts

### Morphisms vs Arrays

Every value is a **morphism**. It is one of two kinds:

- a **concrete array**, which holds data, with a shape, a dtype and strides;
- an **expression**, a computation not yet done, held as a tree of operations.

`realize` turns an expression into a concrete array.

```scheme
;; A concrete array: the data is stored
(define concrete (morph-from-list '(1.0 2.0 3.0) '(3) 'f64))

;; An expression: the sum is not computed yet
(define expr (morph+ concrete concrete))

;; realize computes it
(define result (realize expr))  ; a concrete array
```

### Index Functions

Each expression has an **index function**, which says how an element of
the result is found from its operands:

- **affine** index functions rearrange elements (reshape, transpose, slice);
- **compute** index functions compute elements (arithmetic);
- **window** index functions read windows of elements (convolution, pooling);
- **reduction** index functions combine elements (sum, mean, max).

```scheme
;; Affine: stride-2 slice
(define downsampled (morph-slice x '(0) '(16) 2))

;; Compute: element-wise multiplication
(define scaled (morph* x (morph-from-list '(2.0) '(1) 'f64)))

;; Reduction: sum all elements
(define total (morph-reduce 'sum x))
```

### Zero-Copy Structural Operations

Structural operations make views, which read the same data in another
order:

```scheme
(define x (morph-from-list '(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0) '(8) 'f64))

;; Non-contiguous slice (stride 2)
(define strided (morph-slice x '(0) '(8) 2))  ; (0.0 2.0 4.0 6.0)

;; Reshape works on views that skip elements, too
(define as-2x2 (morph-reshape strided #(2 2)))  ; ((0.0 2.0) (4.0 6.0))

;; Transpose
(define transposed (morph-transpose as-2x2 '(1 0)))  ; ((0.0 4.0) (2.0 6.0))
```

## Basic Operations

### Array Creation

```scheme
(morph-from-list '(1.0 2.0 3.0) '(3) 'f64)  ; from a list
(make-morphism data-vector #(3) 'f64)       ; from an SRFI-4 vector (f64vector, ...)
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
;; The result is f64: 1.0 where true, 0.0 where false
```

### Structural Operations

```scheme
;; Reshape (-1 stands for the size that makes the total match)
(morph-reshape m #(2 3))      ; to 2 x 3
(morph-reshape m '(2 -1))     ; to 2 x whatever fits

;; Transpose
(morph-transpose m)           ; reverse all axes
(morph-transpose m '(1 0))    ; 2-D transpose
(morph-transpose m '(0 2 1))  ; swap the last two axes

;; Slice
(morph-slice m '(0) '(10))      ; elements 0 to 9
(morph-slice m '(0) '(10) 2)    ; every other element

;; Stack and concatenate
(morph-stack (list m1 m2 m3) 0)   ; along a new axis 0
(morph-concat (list m1 m2) 0)     ; along the existing axis 0
```

### Functional Operations

```scheme
;; Map applies a function to each element
(morph-map (lambda (x) (* x x)) arr)

;; Reduce combines elements over the given axes
(morph-reduce 'sum arr)           ; sum of all elements
(morph-reduce 'mean arr '(0))     ; mean over axis 0
(morph-reduce 'max arr '(1 2))    ; maximum over axes 1 and 2

;; With a frame rank (the fifth argument), the leading axes form a frame,
;; and each cell is reduced exactly as it would be on its own (see Frame
;; Lifting)
(morph-reduce 'sum arr '(2) #f 1) ; [F M N] -> [F M]: the row sums of each [M N] cell

;; Scan along an axis (see Scans)
(morph-cumsum arr 0)                         ; running sums down axis 0
(morph-scan 'max arr 1 reverse?: #t)         ; running maxima from the right
(morph-scan '(lambda (a b) (+ (* a 0.9) b)) arr 1)   ; your own combining step

;; Fold and scan over the batch axis
(batch-fold fn init batched-m)
(batch-scan fn init batched-m)
```

## Memory Reuse with Execution Context

A computation that runs many times can reuse its buffers through an
execution context. The first run records the buffers it allocates; the
context then plans one set of buffers, and later runs use them:

```scheme
(import array-morphisms-context)

(define ctx (make-morphism-context))

;; First run: record the allocations, then plan the buffers
(realize/ctx ctx morphism)
(finalize-context! ctx)

;; Later runs: reset, then compute into the planned buffers
(reset-context! ctx)
(realize/ctx ctx morphism)
```

A later run overwrites the buffers of the run before it, so copy a result
you want to keep.

## Type System

Arrays hold one of four element types (dtypes):

| dtype   | Elements                      |
|---------|-------------------------------|
| `'f64`  | double-precision floats       |
| `'f32`  | single-precision floats       |
| `'s64`  | 64-bit signed integers        |
| `'s32`  | 32-bit signed integers        |

How results get their dtype:
- an operation on two dtypes gives the one that comes first in the order
  f64, f32, s64, s32 (so s64 with f32 gives f32);
- transcendental functions and activations (exp, log, sin, relu, ...) of
  integers give f64;
- reductions keep the dtype, except that a mean of integers gives f64.

`(morph-cast m dtype)` converts each element to another dtype: to `'f32`
it rounds to single precision, and to an integer dtype it truncates.

In a compiled SSA training graph, each parameter's gradient has the
parameter's dtype. When an `'f32` parameter meets `'f64` values, the
backward pass computes in `'f64`, as the rules above require, and casts
the gradient to `'f32` at the end.

`ssa-vjp` returns one gradient per parameter, in the order of the
parameters. A parameter that no gradient reaches gets zeros of its shape
and dtype. This includes a parameter reached only through a max reduction,
whose derivative is not computed. A `morph-map` runs like any element-wise
op, but its procedure has no derivative: `ssa-vjp` raises an error when
the input of a map depends on a trainable parameter.

Slices, constant pads, stacks and concatenations can be part of a compiled
SSA graph, and their gradients go back to their inputs: a slice passes back
the incoming gradient padded with zeros, a pad passes back the part of it
that holds the input, and a stack or concatenation gives each input its
part. A slice with a step above 1, and edge or reflect padding, have no
derivative: `ssa-vjp` raises an error when their input depends on a
trainable parameter, as for `morph-map`.

## Performance Tips

1. **Realize once.** Build the whole expression, then call `realize` on
   the result, not on each step.
2. **Use views.** Reshape, transpose and slice cost almost nothing.
3. **Use a context** for a computation that runs many times.
4. **Batch.** Process many arrays as one (see Frame Lifting).

```scheme
;; Good: one expression, realized once
(define result (realize (morph-sqrt (morph+ (morph* a b) c))))

;; Good: a context for repeated inference
(define ctx (make-morphism-context))
(realize/ctx ctx model-output)  ; first run: records the buffers
(finalize-context! ctx)
;; ... later ...
(reset-context! ctx)
(realize/ctx ctx model-output)  ; reuses the buffers

;; Slower: realizing every step stores every intermediate array
(define temp1 (realize (morph* a b)))
(define temp2 (realize (morph+ temp1 c)))
(define result (realize (morph-sqrt temp2)))
```

## BLAS Backends

Matrix products (matmul, matvec, dot, axpy) and convolutions run on a
pluggable BLAS backend (`array-morphisms-blas-exec`). There are four tiers:

| Tier | Package | Dependencies | Default? |
|---|---|---|---|
| Pure Scheme | built in | none | fallback only |
| microBLAS | built in (`array-morphisms-micro-blas-backend`) | none (vendored, header-only) | yes, auto-registered |
| System BLAS | separate egg: [`array-morphisms-blas`](https://github.com/iraikov/array-morphisms-blas) | the `blas` egg + a system BLAS library | opt-in |
| crunch | separate egg: `array-morphisms-crunch` (`make-crunch-blas-backend`) | the `crunch` egg (CHICKEN 6) | opt-in |

The microBLAS backend needs no other library. It is registered when the
library loads, unless another backend was registered first, so
`array-morphisms` never needs a system BLAS library. For more speed,
install the `array-morphisms-blas` egg and register its backend, which
replaces the default:

```scheme
(import array-morphisms-blas-egg-backend)
(register-blas-backend! (make-blas-egg-backend))
```

The crunch tier compiles its matrix and convolution kernels from Scheme to
C with CHICKEN 6's crunch compiler, and splits large products across
threads. It needs no C library beyond libm and pthreads.

### Batched matrix multiply

`morph-matmul` multiplies batches as well as matrices. When an operand has
rank 3 or more, its leading axes are a *frame* and its last two axes are
the matrices (the *cells*):

```scheme
(morph-matmul A B)   ; A [F.. M K], B [F.. K N] -> [F.. M N]
(morph-matmul A W)   ; A [F.. M K], W [K N]     -> [F.. M N]  (W shared by every cell)
(morph-matmul W B)   ; W [M K],     B [F.. K N] -> [F.. M N]
```

Two matrices give the plain `matmul` node. Batches give one `batch-matmul`
node, which `morph-batch-matmul` also builds. Both operands must have the
same frame, unless one of them is a plain matrix shared by every cell.

Each cell of the result has exactly the values that the 2-D product of
that cell would have, because it runs the same GEMM kernel. When the rows
of a cell lie further apart than its width, as when frame axes come
between the cell's axes, the cell counts as the same matrix stored
contiguously.

To read a cell in place, the backend needs GEMM kernels that take an
offset into a vector. `make-blas-backend` takes these, and DOT kernels
with offsets, as four optional arguments after its 20 required ones:

```scheme
(make-blas-backend name gemm-f64 gemm-f32 ...   ; the 20 required kernels
                   gemm-offset-f64 gemm-offset-f32 dot-offset-f64 dot-offset-f32)
;;   gemm-offset: (M N K alpha A off-A lda transa B off-B ldb transb beta C off-C)
;;   dot-offset:  (N x off-x y off-y) -> number
```

microBLAS, the system-BLAS egg and the crunch egg supply them. A backend
without them still works: each cell is copied into a scratch buffer
first, which gives the same values more slowly.

### Prepared GEMM

`execute-blas-gemm/into!` and `execute-blas-gemm-strided/into!` check
their operands (dtype, rank, layout, size) on every call. Code that
multiplies arrays of the same shapes, strides and dtype many times can
check them once:

```scheme
(define plan (prepare-blas-gemm/into A B))   ; or prepare-blas-gemm-strided/into
(execute-gemm-plan/into! plan A B result-data)
```

A plan records the backend kernel to call and its arguments. It is used
only while BLAS is enabled and the backend it was made for is still the
registered one; otherwise `execute-gemm-plan/into!` calls the matching
`execute-blas-*` procedure. `gemm-plan-uses-kernel?` tells whether a plan
calls a backend kernel. SSA replay plans prepare their matrix products
this way when they are compiled.

## Element-wise Kernel Backends

Element-wise operations, reductions and strided copies can run on a
pluggable *activation backend* (`array-morphisms-activation-exec`), which
is registered apart from the BLAS backend. It holds whole-array kernels,
one per op and dtype (f32 or f64), for:

- the activations `relu`, `sigmoid` and `tanh`, and the derivative ops
  `relu-deriv`, `sigmoid-deriv` and `tanh-deriv` that the backward pass emits;
- the binary ops `add`, `sub`, `mul` and `div`;
- reductions of row-major 2-D arrays over axis 0 or 1;
- scans (sum, prod, max, min) of row-major arrays along any axis;
- copies of strided views into row-major order.

When an SSA replay plan is compiled, each element-wise binding is lowered
to a segop (see below). A binding that applies one of these ops directly to
row-major operands of the output's shape gets the `segop/activation-unary`
or `segop/activation-binary` executor, which uses the backend's kernel for
its op and dtype at execution time.

Reductions and strided copies use their kernels wherever arrays are
realized, not only during replay. When no backend is registered, when it
has no kernel for that op and dtype, or when the operand and output
dtypes differ, the Scheme loops run instead.

The kernels compute exactly what the Scheme loops compute, so registering
a backend changes speed, not results.

### Fusion

`ssa-vjp` runs a fusion pass on the program it builds. A forward program
from `morphism-to-ssa` is fused by `ssa-realize/ctx` at its first trace, in
place, under the policy and the backends in effect then;
`ssa-program-fused?` tells whether a program has been through the pass.
Differentiate a forward program before realizing it through a context:
`ssa-vjp` refuses a program with fused bindings. The pass can fuse an
element-wise binding into the element-wise binding that consumes it,
anywhere later in the program, when that consumer is the binding's only use
and reads it once. Either binding may read inputs that broadcast to its
shape; a broadcast read of the binding itself is not fused. A program
output counts as a use, so a binding whose value is an output is never
fused away. A fused binding (op `fused`) computes both bindings in one
segop and never stores the intermediate value. Chains are fused to the end
in one pass, and an input that both bindings read is read once.

The pass also fuses element-wise bindings with reductions, in two ways:

- An element-wise binding read by a reduction is computed inside the
  reduction, so its values are summed without being stored first. If other
  bindings also read it, the fused reduction stores it for them as a second
  output. This covers bias gradients, whose values the weight gradient also
  reads.
- An element-wise binding that reads only a reduction's result becomes the
  reduction's final step (its *finalize*). For example, rmsnorm's
  `(mean(x*x) + eps)^-0.5` becomes one segop.

A fused reduction adds its values in the same order as the unfused one, so
results do not change. Sums of rank-1 arrays are not fused, because their
order is left to the BLAS backend.

A matrix product is treated as a reduction too. Element-wise bindings that
read only its result become its *epilogue*, which is applied to the stored
product in place, so the product needs no buffer of its own. The default
policy fuses one op on the product, or on the product and a bias row; `'all`
also fuses chains of ops and other inputs.

A fused binding can also have several outputs. An element-wise binding
with other readers can still be fused into one of them, and it keeps its
value as a second output for the others. Two element-wise bindings of the
same shape that read a common input, where neither depends on the other,
can be computed in one pass, which reads that input once.

Whether a legal fusion is made depends on the policy, the value of the
parameter `segop-fusion` (`array-morphisms-ssa`):

- `#t` (the default) fuses only what some segop backend can run natively,
  as one compiled kernel. On the host, a fused segop runs its body as
  compiled Scheme closures, which are slower than the separate bindings'
  loops and kernels. So the default fuses a pair only when the active
  activation backend holds a *fused kernel* for the body the fusion would
  make (see below), or when a code generator, such as the one of
  `array-morphisms-crunch`, will compile it. With no such backend, it
  fuses nothing.
- `'all` fuses every legal pair.
- `#f` runs no fusion pass.

Segop backends say what they can run natively. Each one has a kind: `host`
(Scheme executors only), `catalogue` (a fixed set of compiled kernels) or
`codegen` (backends that generate a kernel for each body, such as
`array-morphisms-crunch-codegen`). Each may also
answer a native-capability query, `native?`, on a layout-free segop
template (`make-segop-template`). The fusion pass asks this query, through
`segop-native?`, before it fuses, and predicts each input's layout: the
values of bindings that allocate, and reshapes of such values, are
expected to be flat; other views are not. The registry backend is a catalogue
backend. It answers from the fused kernels of the active activation
backend, which are keyed by the body's canonical form
(`canonical-scalar-body`): commutative operands in a fixed order, inputs
numbered by first occurrence. It runs a matching segop as
`segop/fused-kernel` when every input is flat and of the output's dtype.

A template made for a fusion also names its *fusion class*, the kind of
fusion that would make it: `map` (a map into a map), `kept` (a map into
one of its readers, kept as an output for the others), `horizontal` (two
maps over a common input), `redomap` (a map into a reduction), `epilogue`
(a map after a reduction) or `gemm-epilogue` (a map after a matrix
product). A backend can answer differently for each class, so that the
default policy makes only the kinds of fusion that it runs faster.

An activation backend holds fused kernels through
`activation-backend-add-fused-kernel!`. The crunch backends of
`array-morphisms-crunch` hold kernels for the bodies the pass forms most
often in backward passes, such as `g * relu-deriv(x)`, and for Llama's
`b * silu(a)`. An activation backend can also hold *column-sum kernels*
(`activation-backend-add-colsum-kernel!`), which compute a body and sum it
over the rows in one pass; the crunch backends currently hold none.
`replay-plan-stats` reports under `fused` how many fused bindings run as a
native kernel and how many on the host.
`ssa-fusion-pass-stats` lists the fused bodies the pass formed
(`candidates`, and `red-candidates` for reductions), counts the fusions
with several outputs (`kept`, `horizontal`, `dead-outputs`), counts the
fusions made in each class (`classes`), and counts the pairs it left
unfused, by reason (`missed`). Run under `'all`, it shows which fused
kernels would be worth adding.

Fusion does not change results. A fused binding rounds each intermediate
value to its dtype, as storing it in an array would (`(cast f32 ...)` in its
body), so an `'f32` chain computes exactly what the separate bindings
compute, and a program gives the same values under `'all` and `#f`.
Set the policy globally, as in `(segop-fusion 'all)`, or around `ssa-vjp`
with `(parameterize ((segop-fusion #f)) ...)`.
`segop-fusion-max-rounds` bounds the number of rounds of the pass (16 by
default), and `ssa-fusion-pass-stats` reports the last run.

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

There are four kinds: `seg-map` (element-wise maps, possibly with several
outputs), `seg-reduce` (reductions, with a combine spec giving the operator,
neutral element, accumulator precision and summation order), `seg-scan`
(scans along the last dimension of the space, with a combine spec, an
optional seed, and inclusive or exclusive results) and `seg-lib` (named
library operations, such as an index function run on the host).
Index maps cover identities, broadcasts, transposes, slices, reshapes
(with `floordiv` and `mod`) and padded reads (with a guard and a fill
value). Bodies and maps are data, so a code generator can read them and
compile a kernel.

The segop abstractions are inspired by and follow the design of the
Futhark compiler [1, 2]. Futhark's internal representation expresses
parallel code as segmented operations (`SegMap`, `SegRed`, `SegScan` and
others) over an iteration space, and `seg-map`, `seg-reduce` and
`seg-scan` take their structure from `SegMap`, `SegRed` and `SegScan`. The fusion of a map into the reduction that
reads it, with the map's value kept as an extra output when other code
needs it, follows Futhark's *redomap* construct [3].

The ops a body can name come from the *scalar-op table*
(`scalar-ops-table.scm`, module `array-morphisms-scalar-ops-table`). Each
entry gives an op's parameters and formula. The combiners of
`array-morphisms-basic-ops` are generated from the same entries
(`array-morphisms-scalar-ops`), and `array-morphisms-crunch` reads the
table at compile time to generate its kernels, so all of them compute the
same formula.

`segop-interpret` executes any `seg-map`, `seg-reduce` or `seg-scan`
directly from its definition. It is slow and is meant for testing.

When a replay plan is compiled, every binding except zero-copy views is
lowered to a segop:

- element-wise ops become a `seg-map` whose inputs are read through
  broadcast maps and their own strides, so transposed views are read in
  place;
- reductions become a `seg-reduce`, whose combine spec records how the
  reduction loop rounds: f32 sums over axis 0 round after every step, sums
  over axis 1 accumulate in double precision, and so on;
- a matrix product `A[M,K] B[K,N]` becomes a `seg-reduce` over `(M N K)` with
  the body `(mul (in 0) (in 1))`; transposed operands are described by their
  strides. Its combine spec sums in the BLAS backend's order when the
  prepared plan calls a kernel, and otherwise in increasing k with a double
  accumulator, as the Scheme loop does. An adjacent element-wise consumer
  (an activation, or a bias add) of the same shape and dtype is fused as
  the reduction's finalize, applied to the stored product
  (`(sigmoid (cast f32 (reduced)))`);
- a scan becomes a `seg-scan` whose space lists the other axes and then
  the scanned one; a reversed scan reads and writes the scanned axis
  from its end;
- materialized views (such as a reshape of a transposed array) become a
  gather, a `seg-map` with the body `(in 0)`;
- the convolution ops become segops over spaces with one dimension per
  loop index: `im2col` is a `seg-map` reading the image through a padded
  map, `col2im` a `seg-reduce` that gathers the windows covering each pixel,
  and the forward convolution and its two gradients are contractions like
  the matrix product. They run on the convolution kernels of the BLAS
  backend, which use a buffer for the image columns; the convolutions of a
  plan share these buffers;
- the remaining index functions run on the host through a `seg-lib`.

Each segop is offered to the segop backends in turn. The first one that
accepts it supplies an executor, and the binding becomes an `ri-segop`
instruction. The backends in use are the value of the parameter
`segop-backends`:

```scheme
(import array-morphisms-segop)
(segop-backends)                      ; the backends, in the order tried
(register-segop-backend! b)           ; try b before the others
(register-segop-backend! b '(after registry)) ; or at a given place
(parameterize ((segop-backends (list b)))
  ...)                                ; use only b within this extent
```

Two backends are always present. `registry` runs the patterns that have
kernels: matrix products on the BLAS backend, through prepared GEMM plans,
and element-wise ops, reductions over axis 0 or 1 of a row-major matrix,
scans of row-major arrays and strided copies on the activation backend. `scheme` runs the rest with
Scheme loops, and as a last resort evaluates a segop with the reference
interpreter. Every replay gives the same values, bit for bit, as the
first (traced) run.

A backend added in between can still fall back on the ones after it:
`(select-segop-executor s name)` asks only the backends that follow the
backend called `name`. While a plan is compiled, backends can also share
work across its segops. `segop-plan-property` and
`segop-plan-property-set!` keep values for the plan being compiled, and
`segop-plan-add-finalizer!` adds a procedure that runs once, after the
last segop. An executor can be given as an *executor cell*
(`make-segop-executor-cell`, `segop-executor-cell-executor`), whose
content a finalizer may replace. The code generator uses these to compile
all the kernels of a plan at once, and the convolutions use them to
allocate their shared image-column buffers. A finalizer may return
statistics, which `replay-plan-stats` reports with the plan's own.

The values a replay computes live in the context's buffers. An output that
the traced run produced as a strided view is copied into a buffer that the
plan owns. The next replay overwrites these buffers, so copy an output
that you want to keep across replays. Once the plan is compiled, the
arrays of the traced run are released.

The timing profile (`replay-timing-results`) and `replay-plan-stats` name
each segop instruction by its pattern:

- element-wise: `segop/activation-unary`, `segop/activation-binary`,
  `segop/flat-unary`, `segop/flat-binary`, `segop/bias-broadcast`,
  `segop/flat-ternary`, `segop/flat-broadcast`, and `segop/strided-map` for
  any other layout;
- reductions: `segop/reduce-1d`, `segop/reduce-axis0`, `segop/reduce-axis1`,
  `segop/reduce-rows-1d` (the vector sums of a reduction with a frame, one
  per row) and `segop/reduce-general`. A sum or mean over the leading axes
  of a row-major array of rank 3 or more runs as `segop/reduce-axis0`: it
  adds in the same order and precision as the general loop;
- matrix products: `segop/gemm` (row-major operands), `segop/gemm-strided`
  (transposed or strided operands), `segop/gemm-epilogue` (with an
  epilogue run as a flat kernel), `segop/gemm-epilogue-host` (with an
  epilogue run on the host) and `segop/gemm-batched` (a batched product);
- convolutions: `segop/im2col`, `segop/col2im`, `segop/conv-fwd`,
  `segop/conv-bwd-data` and `segop/conv-bwd-weights`;
- scans: `segop/scan-kernel` (a kernel of the activation backend),
  `segop/scan` (a Scheme loop for the standard ops on f32 and f64) and
  `segop/strided-scan` (any other scan);
- `segop/gather` for materialized views, `segop/host-index-fn` for index
  functions run on the host, and `segop/interpreted`;
- generated kernels, with the code generator of `array-morphisms-crunch`:
  `segop/codegen-map`, `segop/codegen-red`, `segop/codegen-epilogue` and
  `segop/codegen-scan`.

`replay-plan-stats` also reports `conv-scratch-bytes`, the size of the
image-column buffers that the plan's convolutions share.

## Scans

A scan computes running results along one axis: element `r` of the
output combines the elements `0` to `r` of its line.

```scheme
(morph-scan op m axis #!key exclusive? reverse? init neutral)
(morph-cumsum m axis ...)     ; (morph-scan 'sum m axis ...)
(morph-cumprod m axis ...)    ; (morph-scan 'prod m axis ...)
```

- `op` is `sum`, `prod`, `max`, `min`, or a combining expression (below).
- `exclusive?: #t` shifts the results by one position: element `r`
  combines the elements before `r`, and element 0 is `init` or the
  neutral element (0, 1, or the lowest or highest value of the dtype).
- `reverse?: #t` scans from the last element down to the first.
- `init` is an array with the shape of `m` without the scanned axis: the
  value each line starts from.

Each output is the previous output combined with one more element and
rounded to the result's dtype, exactly as a loop over the elements in
order computes it. The first output is the first element itself, so a
first element `-0.0` or NaN is kept. `max` replaces the running value
only by a greater element, so a NaN after the first element is passed
over; `min` likewise.

### Your own combining step

A combining expression `(lambda (a b) e)` gives the step: `a` is the
value so far and `b` the next element. `e` uses the ops of the scalar-op
table, numbers, `cast`, and `if` with a comparison (`<`, `>`, `<=`, `>=`,
`=`) as its test:

```scheme
;; an exponential moving sum: out[r] = 0.9 * out[r-1] + x[r]
(morph-scan '(lambda (a b) (+ (* a 0.9) b)) x 1)
```

`e` is computed in double precision and rounded to the result's dtype
after every step; `(cast f32 ...)` rounds an intermediate value. The
elements are combined in order, so `e` does not need to be associative.
An exclusive scan with an expression needs `init` or `neutral:`.

`morph-scan/fn` (`array-morphisms-lift`) takes the step as a procedure of
two morphisms instead, and turns it into an expression:

```scheme
(import array-morphisms-lift)
(morph-scan/fn (lambda (acc x) (morph+ (morph* acc k) x)) m 0 init: m0)  ; k a one-element array
```

The procedure may use element-wise ops of its two arguments and of
one-element constants. Each intermediate result is rounded as storing it
would round it, so the scan gives the values of applying the procedure
step by step. A procedure that does something else (`morph-map`, a
reduction, an array captured from outside) signals the `lift-unsupported`
condition. A step that adds or multiplies its two arguments becomes the
`sum` or `prod` scan.

`batch-scan` (`array-morphisms-batch-ops`) uses `morph-scan/fn` along the
batch axis: the result is one scan, with as many nodes for any batch
size. When the step cannot be turned into an expression, or `init` is not
an array of one element's shape and dtype, it runs `batch-scan/slices`,
which applies the step to each slice and stacks the results; that version
also runs in compiled SSA graphs, with as many nodes as elements.

### Gradients

`ssa-vjp` differentiates `sum` scans: the gradient of the scanned array is
the scan of the incoming gradient in the opposite direction, and that of
`init` is the incoming gradient summed along the axis, in the same order as
the step-by-step program adds it. `var-scan`
(`array-morphisms-grad`) does the same for `backward!`. Scans with other
ops or with an expression have no gradient; using one on a trainable
parameter is an error.

### Execution

Scans are never fused with their neighbours. A replay runs each scan on
the first of these that applies:

- a kernel of the activation backend (`segop/scan-kernel`), for the
  standard ops on row-major f32 or f64 arrays; the crunch backends have
  one;
- a kernel generated by the code generator of `array-morphisms-crunch`
  (`segop/codegen-scan`), for expressions, strided arrays and integer
  dtypes, with the lines split across threads;
- Scheme loops (`segop/scan`, `segop/strided-scan`).

All of them give the same values. Measured on f32 and f64 cumulative sums
of up to 1024 x 4096 elements, the crunch kernel runs about 30 to 125 times
faster than the Scheme loop; for an exponential moving sum, a generated
kernel runs 250 to 1000 times faster than the Scheme loop.

## Frame Lifting

`array-morphisms-lift` applies a function to every cell of a frame as one
computation:

```scheme
(import array-morphisms-lift)

;; x [F B D]: each cell is a [B D] matrix; W and b are shared by every cell
(define y (morph-lift (lambda (xc) (morph-tanh-am (morph+ (morph-matmul xc W) b)))
                      2 x))   ; cell rank 2 -> y [F B D]
```

`(morph-lift fn cell-ranks arg ...)` takes the rank of the cells that
`fn` expects, one per argument or one for all. The leading axes of an
argument, beyond its cell rank, are its frame. The longest frame becomes
the result's frame, and every other frame must be a prefix of it. An
argument with a shorter frame is reused along the frame axes it lacks.

`morph-lift` calls `fn` once, on placeholder arrays with the cell shapes,
and widens the expression `fn` builds by the frame:

- element-wise operations stay one element-wise node;
- a reduction becomes one reduction with a frame rank;
- a matrix product becomes one batched matrix product;
- transposes, reshapes, slices, padding, stacks, concatenations and scans
  keep the frame axes in front;
- attention takes the frame as leading axes (up to rank 4 in all), and
  `im2col` and `col2im` fold it into the image axis.

Parts of the expression that use no argument, such as `W` above, are kept
as they are and computed once. So the expression has as many nodes for a
frame of 4 cells as for 64.

Every cell of the result has exactly the values that `fn` gives for that
cell stored in an array of its own: elements, reductions and products are
computed in the same order and precision.

`fn` must only build expressions from the placeholders; it must not read
their values. When `fn` uses an operation that has no widening rule (such
as `morph-matvec`), `morph-lift` signals a condition. `lift-unsupported?`
recognizes it, and `lift-unsupported-op` gives the operation's name.

`batch-map` and `batch-zip` (`array-morphisms-batch-ops`) use
`morph-lift`. When it cannot lift their function, they fall back to
`batch-map/slices` and `batch-zip/slices`, which apply the function to
each element slice and stack the results.

### Training lifted computations

`var-lift` (`array-morphisms-grad`) is the gradient-tracking form:

```scheme
(define y (var-lift (lambda (xc) (morph-tanh-am (morph+ (morph-matmul xc (var-value W))
                                                        (var-value b))))
                    2 x-var))
```

`ssa-vjp` derives the gradients of lifted computations; nanograd's
`am-training-step/ssa` uses it. Each cell of a framed value gets exactly
the gradient that the cell's own program would give.

An array shared by every cell, such as `W`, gets the sum of the cells'
gradients. The lifted program adds them in a different order than adding
up per-cell gradients would, so the two results agree only up to
rounding.

`backward!` does not differentiate through `var-lift`. It signals an error
that names `am-training-step/ssa`.

### Performance

Measured in f32, a lifted layer `tanh(x W + b)` with frames of 8 to 64
cells:

- **Forward pass:** 6 to 17 times faster than slice-and-stack.
- **Training step on the SSA path:** faster than one program per cell by
  1.7 to 3 times on microBLAS, 2.4 to 4.7 times on system BLAS and 17 to
  37 times on crunch.

For the attention of the transformer benchmark, lifting over the heads
beats the per-head loop when the code generator of `array-morphisms-crunch`
is on: 16 against 26 ms per step. Without the code generator it is slower
(154 against 83 ms), because the copies that regroup the heads then run in
host loops.

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
    ;; Both slices are views; realize computes the result in one pass
    (realize quarter)))
```

### Batched Matrix Operations

```scheme
(import array-morphisms-batch-ops)

;; Stack matrices into a batch
(define batch (morph-stack (list m1 m2 m3) 0))

;; Apply an operation to every batch element, as one computation
(define doubled (batch-map
                   (lambda (m) (morph-map (lambda (x) (* x 2)) m))
                   batch))

;; One batched product: each [M K] element times a shared [K N] matrix
(define products (morph-matmul batch W))

;; Sum over the batch axis
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
- `array-morphisms-lift` - Frame lifting (`morph-lift`) and scans with a procedure as step (`morph-scan/fn`)
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
- Segmented operations and map-reduce fusion modeled on the Futhark
  compiler (https://futhark-lang.org), by Troels Henriksen, Cosmin Oancea
  and colleagues at DIKU, University of Copenhagen
- Category-theoretic model from functional programming research
- Buffer reuse patterns from the stream fusion and buffer optimization literature

## References

1. T. Henriksen, N. G. Serup, M. Elsman, F. Henglein and C. E. Oancea.
   Futhark: Purely Functional GPU-Programming with Nested Parallelism and
   In-Place Array Updates. In *Proceedings of the 38th ACM SIGPLAN
   Conference on Programming Language Design and Implementation (PLDI)*,
   2017.
2. T. Henriksen, F. Thorøe, M. Elsman and C. E. Oancea. Incremental
   Flattening for Nested Data Parallelism. In *Proceedings of the 24th ACM
   SIGPLAN Symposium on Principles and Practice of Parallel Programming
   (PPoPP)*, 2019.
3. T. Henriksen, K. F. Larsen and C. E. Oancea. Design and GPGPU
   Performance of Futhark's Redomap Construct. In *Proceedings of the 3rd
   ACM SIGPLAN International Workshop on Libraries, Languages, and
   Compilers for Array Programming (ARRAY)*, 2016.
