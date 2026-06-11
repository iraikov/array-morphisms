;;; array-morphisms-grad.scm
;;; Automatic Differentiation Infrastructure
;;;
;;; Provides gradient tracking on morph-variables and a reverse-mode
;;; backward pass.  Gradients are themselves lazy morphisms: every
;;; backward rule is expressed in terms of the existing lazy morphism
;;; combinators (morph*, morph+, morph-reduce, morph-transpose, ...).
;;; No realize call is made during symbolic differentiation.
;;;
;;; The only allocation that occurs during a standard backward! call
;;; is morph-ones-like, which builds the scalar seed gradient once.
;;;
;;; Design: MoA principle: gradients are morphisms, not matrices.
;;;   - Backward graph is a morphism-expr tree of the same kind as
;;;     the forward graph.
;;;   - Zero-copy structural ops (reshape, transpose) propagate
;;;     through gradient ops: the gradient of a reshape is another
;;;     morph-reshape backed by the same flat index function.
;;;   - The realization engine and context.scm memory-reuse
;;;     infrastructure apply uniformly to forward and backward trees.
;;;
;;; var-matmul backward is fully lazy: morph-transpose produces a zero-copy
;;; strided view, and the realization engine routes strided matmul operands
;;; through execute-scheme-gemm (stride-aware) via the gemm-strided dispatch
;;; tag in blas-compat.scm.

(module array-morphisms-grad

  (;; Variable type predicate
   morph-variable?

   ;; Construction
   make-var

   ;; Accessors
   var-value
   var-grad
   var-requires-grad?

   ;; Gradient management
   zero-grad!
   accumulate-grad!

   ;; Record mutators (needed for testing/injection of custom grad-fns)
   morph-variable-grad-fn-set!
   morph-variable-parents-set!

   ;; Utilities (no realization)
   morph-ones-like
   reduce-sum-to

   ;; Gradient-tracking operations - all return morph-variable
   var+
   var-
   var*
   var/
   var-pow
   var-sqrt
   var-exp
   var-log
   var-sin
   var-cos
   var-negate
   var-abs
   var-sum
   var-mean
   var-reshape
   var-transpose
   var-matmul

   ;; Backward pass
   backward!)

  (import scheme (chicken base))
  (import (only srfi-1 iota any filter every map fold))
  (import (only srfi-4
                f32vector? f64vector? s32vector? s64vector?
                f32vector-length f64vector-length
                s32vector-length s64vector-length
                f32vector-set! f64vector-set!
                s32vector-set! s64vector-set!
                u32vector-set! u64vector-set!))
  (import (only srfi-69
                make-hash-table
                hash-table-ref/default
                hash-table-set!))
  (import datatype matchable)
  (import array-morphisms-core)
  (import array-morphisms-index-fn)
  (import array-morphisms-basic-ops)
  (import array-morphisms-structural-ops)
  (import array-morphisms-blas-exec)


;;;; ============================================================
;;;; morph-variable Record Type
;;;;
;;;; Wraps a morphism with gradient state.  All gradient values
;;;; are lazy morphisms. Realization is deferred until the user
;;;; explicitly calls realize on var-grad.
;;;; ============================================================

(define-record morph-variable
  value         ; morphism (abstract or concrete): the forward value
  grad          ; accumulated gradient, itself a morphism (#f before backward!)
  requires-grad ; boolean: leaf inputs that need derivatives set this #t
  grad-fn       ; backward closure (grad-out-morphism -> void), or #f for leaves
  parents)      ; list of parent morph-variable nodes for graph traversal


;;;; ============================================================
;;;; Public Constructors and Accessors
;;;; ============================================================

(define (make-var m #!optional (rg #f))
  "Create a gradient-tracking variable wrapping morphism m.
   rg: if #t, gradients will be accumulated into this variable."
  (unless (array-morphism? m)
    (error "make-var: value must be an array morphism" m))
  (make-morph-variable m #f rg #f '()))

(define (var-value v)
  "Extract the morphism value from a variable."
  (morph-variable-value v))

(define (var-grad v)
  "Extract the accumulated gradient morphism, or #f if not yet computed."
  (morph-variable-grad v))

(define (var-requires-grad? v)
  "True if gradients will be computed for this variable."
  (morph-variable-requires-grad v))


;;;; ============================================================
;;;; Gradient Management
;;;; ============================================================

(define (zero-grad! v)
  "Reset accumulated gradient to #f."
  (morph-variable-grad-set! v #f))

(define (accumulate-grad! v g)
  "Add lazy morphism g to v's accumulated gradient.
   First call: sets gradient directly.
   Subsequent calls: combines via morph+ (lazy, no realization)."
  (let ((existing (morph-variable-grad v)))
    (morph-variable-grad-set!
     v (if existing (morph+ existing g) g))))

(define (grad-requires? vs)
  "True if any variable in vs has requires-grad=#t."
  (any var-requires-grad? vs))


;;;; ============================================================
;;;; Utility: effective-grad-dtype
;;;; ============================================================

(define (effective-grad-dtype dtype)
  "Return floating-point dtype suitable for gradient accumulation.
   Integer dtypes are promoted to f64."
  (if (dtype-floating? dtype) dtype 'f64))


;;;; ============================================================
;;;; Utility: morph-ones-like
;;;;
;;;; Creates a concrete-array of ones matching the shape+dtype of
;;;; the given morphism.  morph-shape works on abstract morphisms
;;;; without triggering realization.  This is the only concrete
;;;; allocation in a standard backward! call (used once for seed).
;;;; ============================================================

(define (morph-ones-like m)
  "Create a concrete-array of ones with same shape as m.
   Integer dtypes are promoted to f64 for gradient arithmetic."
  (let* ((shape (morph-shape m))
         (dtype (effective-grad-dtype (morph-dtype m)))
         (n     (shape-size shape))
         (data  (allocate-typed-vector dtype n)))
    (let loop ((i 0))
      (when (< i n)
        (typed-vector-set! data dtype i 1.0)
        (loop (+ i 1))))
    (make-morphism data (vector->list shape) dtype)))


;;;; ============================================================
;;;; Utility: reduce-sum-to
;;;;
;;;; Reduces gradient morphism g to match target-shape by summing
;;;; over broadcast/extra dimensions.  Purely lazy: morph-reduce
;;;; returns a reduction-morphism; morph-shape reads stored metadata
;;;; without triggering realization.
;;;; ============================================================

(define (reduce-sum-to g target-shape)
  "Sum gradient morphism g over axes not present in target-shape.
   Handles leading extra dims and broadcast (size-1) dims.
   Returns a lazy morphism."
  (let* ((g-shape (morph-shape g))
         (g-rank  (vector-length g-shape))
         (t-vec   (if (vector? target-shape)
                      target-shape
                      (list->vector target-shape)))
         (t-rank  (vector-length t-vec)))

    ;; Step 1: sum over leading extra dimensions (g has more axes than target)
    (let* ((extra (- g-rank t-rank))
           (g1    (if (> extra 0)
                      (morph-reduce 'sum g (iota extra))
                      g)))

      ;; Step 2: sum over broadcast axes (target has 1, gradient has > 1)
      (let loop ((k       0)
                 (cur     g1)
                 (cur-shp (morph-shape g1)))
        (if (>= k t-rank)
            cur
            (let ((t-dim (vector-ref t-vec k))
                  (g-dim (vector-ref cur-shp k)))
              (if (and (= t-dim 1) (> g-dim 1))
                  ;; Sum axis k with keepdims=#t so shape stays consistent
                  (let* ((r   (morph-reduce 'sum cur (list k) #t)))
                    (loop (+ k 1) r (morph-shape r)))
                  (loop (+ k 1) cur cur-shp))))))))


;;;; ============================================================
;;;; Internal: broadcast-grad
;;;;
;;;; Inverse of reduction: expands a gradient back to full input shape.
;;;; Used by var-sum and var-mean backward rules.
;;;; All operations return lazy morphisms.
;;;; ============================================================

(define (broadcast-grad g x-morph input-shape reduced-axes keepdims?)
  "Broadcast gradient g back to input-shape.
   keepdims? #t: g already has 1s at reduced positions, just broadcast.
   keepdims? #f: g has lower rank; insert 1s at reduced positions first.
   Returns a lazy morphism."
  (let ((ones-x (morph-ones-like x-morph)))
    (if keepdims?
        ;; g has 1s at reduced positions: morph* broadcasts to full size
        (morph* g ones-x)
        ;; g has lower rank: reshape to insert 1s, then broadcast
        (let* ((in-rank (vector-length input-shape))
               (expanded-shape
                (list->vector
                 (map (lambda (i)
                        (if (member i reduced-axes) 1
                            (vector-ref input-shape i)))
                      (iota in-rank)))))
          (morph* (morph-reshape g expanded-shape) ones-x)))))


;;;; ============================================================
;;;; Internal: compute-reduction-n
;;;; ============================================================

(define (compute-reduction-n x-shape axes)
  "Product of sizes of the reduced axes (denominator for mean backward)."
  (apply * (map (lambda (a) (vector-ref x-shape a)) axes)))


;;;; ============================================================
;;;; Internal: morph-sign
;;;;
;;;; Sign function via morph-map. Returns a lazy morphism-expr.
;;;; ============================================================

(define (morph-sign m)
  "Element-wise sign: -1.0 / 0.0 / 1.0.  Returns a lazy morphism."
  (morph-map (lambda (x)
               (cond ((> x 0.0)  1.0)
                     ((< x 0.0) -1.0)
                     (else        0.0)))
             m))


;;;; ============================================================
;;;; Binary Arithmetic Operations
;;;; ============================================================

(define (var+ v1 v2)
  "Gradient-tracking element-wise addition.
   d/dx1 = reduce-sum-to(g, shape(x1))
   d/dx2 = reduce-sum-to(g, shape(x2))"
  (let* ((x1  (var-value v1))
         (x2  (var-value v2))
         (rg  (grad-requires? (list v1 v2)))
         (out (make-var (morph+ x1 x2) rg)))
    (when rg
      (morph-variable-parents-set! out
        (filter var-requires-grad? (list v1 v2)))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (when (var-requires-grad? v1)
            (accumulate-grad! v1 (reduce-sum-to g (morph-shape x1))))
          (when (var-requires-grad? v2)
            (accumulate-grad! v2 (reduce-sum-to g (morph-shape x2)))))))
    out))

(define (var- v1 v2)
  "Gradient-tracking element-wise subtraction.
   d/dx1 = reduce-sum-to(g, shape(x1))
   d/dx2 = reduce-sum-to(-g, shape(x2))"
  (let* ((x1  (var-value v1))
         (x2  (var-value v2))
         (rg  (grad-requires? (list v1 v2)))
         (out (make-var (morph- x1 x2) rg)))
    (when rg
      (morph-variable-parents-set! out
        (filter var-requires-grad? (list v1 v2)))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (when (var-requires-grad? v1)
            (accumulate-grad! v1 (reduce-sum-to g (morph-shape x1))))
          (when (var-requires-grad? v2)
            (accumulate-grad! v2
              (reduce-sum-to (morph-negate g) (morph-shape x2)))))))
    out))

(define (var* v1 v2)
  "Gradient-tracking element-wise multiplication (product rule).
   d/dx1 = reduce-sum-to(g * x2, shape(x1))
   d/dx2 = reduce-sum-to(g * x1, shape(x2))"
  (let* ((x1  (var-value v1))
         (x2  (var-value v2))
         (rg  (grad-requires? (list v1 v2)))
         (out (make-var (morph* x1 x2) rg)))
    (when rg
      (morph-variable-parents-set! out
        (filter var-requires-grad? (list v1 v2)))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          ;; x1 and x2 are morphisms captured in the closure (still lazy)
          (when (var-requires-grad? v1)
            (accumulate-grad! v1
              (reduce-sum-to (morph* g x2) (morph-shape x1))))
          (when (var-requires-grad? v2)
            (accumulate-grad! v2
              (reduce-sum-to (morph* g x1) (morph-shape x2)))))))
    out))

(define (var/ v1 v2)
  "Gradient-tracking element-wise division (quotient rule).
   d/dx1 = reduce-sum-to(g / x2, shape(x1))
   d/dx2 = reduce-sum-to(-g * x1 / x2^2, shape(x2))"
  (let* ((x1  (var-value v1))
         (x2  (var-value v2))
         (rg  (grad-requires? (list v1 v2)))
         (out (make-var (morph/ x1 x2) rg)))
    (when rg
      (morph-variable-parents-set! out
        (filter var-requires-grad? (list v1 v2)))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (when (var-requires-grad? v1)
            (accumulate-grad! v1
              (reduce-sum-to (morph/ g x2) (morph-shape x1))))
          (when (var-requires-grad? v2)
            (accumulate-grad! v2
              (reduce-sum-to
               (morph* (morph-negate g)
                       (morph/ x1 (morph* x2 x2)))
               (morph-shape x2)))))))
    out))

(define (var-pow v1 v2)
  "Gradient-tracking element-wise exponentiation.
   d/dx1 = g * x2 * x1^(x2-1)
   d/dx2 = g * x1^x2 * log(x1)"
  (let* ((x1  (var-value v1))
         (x2  (var-value v2))
         (rg  (grad-requires? (list v1 v2)))
         (out (make-var (morph-pow x1 x2) rg)))
    (when rg
      (morph-variable-parents-set! out
        (filter var-requires-grad? (list v1 v2)))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (when (var-requires-grad? v1)
            (let* ((one (morph-from-list '(1.0) '(1) 'f64))
                   (dx1 (morph* g (morph* x2 (morph-pow x1 (morph- x2 one))))))
              (accumulate-grad! v1 (reduce-sum-to dx1 (morph-shape x1)))))
          (when (var-requires-grad? v2)
            (let ((dx2 (morph* g (morph* (morph-pow x1 x2) (morph-log x1)))))
              (accumulate-grad! v2 (reduce-sum-to dx2 (morph-shape x2))))))))
    out))


;;;; ============================================================
;;;; Unary Transcendental Operations
;;;;
;;;; For exp and sqrt, the backward rule reuses the forward output
;;;; morphism (out-morph) without realizing it.  When the gradient
;;;; is eventually realized, the engine evaluates exp(x)/sqrt(x)
;;;; as part of the combined expression, enabling forward-backward
;;;; fusion.
;;;; ============================================================

(define (var-sqrt v)
  "Gradient-tracking square root.
   d/dx = g / (2 * sqrt(x)) uses out-morph = sqrt(x) [lazy]"
  (let* ((x         (var-value v))
         (rg        (var-requires-grad? v))
         (out-morph (morph-sqrt x))
         (out       (make-var out-morph rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (let* ((two (morph-from-list '(2.0) '(1) 'f64))
                 (dx  (morph/ g (morph* two out-morph))))  ; lazy
            (accumulate-grad! v dx)))))
    out))

(define (var-exp v)
  "Gradient-tracking exponential.
   d/dx = g * exp(x) uses out-morph = exp(x) [lazy]"
  (let* ((x         (var-value v))
         (rg        (var-requires-grad? v))
         (out-morph (morph-exp x))
         (out       (make-var out-morph rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph* g out-morph)))))  ; lazy
    out))

(define (var-log v)
  "Gradient-tracking natural logarithm.
   d/dx = g / x  [lazy]"
  (let* ((x   (var-value v))
         (rg  (var-requires-grad? v))
         (out (make-var (morph-log x) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph/ g x)))))
    out))

(define (var-sin v)
  "Gradient-tracking sine.
   d/dx = g * cos(x)  [lazy]"
  (let* ((x   (var-value v))
         (rg  (var-requires-grad? v))
         (out (make-var (morph-sin x) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph* g (morph-cos x))))))
    out))

(define (var-cos v)
  "Gradient-tracking cosine.
   d/dx = -g * sin(x)  [lazy]"
  (let* ((x   (var-value v))
         (rg  (var-requires-grad? v))
         (out (make-var (morph-cos x) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph-negate (morph* g (morph-sin x)))))))
    out))


;;;; ============================================================
;;;; Unary Arithmetic Operations
;;;; ============================================================

(define (var-negate v)
  "Gradient-tracking negation.  d/dx = -g  [lazy]"
  (let* ((x   (var-value v))
         (rg  (var-requires-grad? v))
         (out (make-var (morph-negate x) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph-negate g)))))
    out))

(define (var-abs v)
  "Gradient-tracking absolute value.  d/dx = g * sign(x)  [lazy]"
  (let* ((x   (var-value v))
         (rg  (var-requires-grad? v))
         (out (make-var (morph-abs x) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph* g (morph-sign x))))))
    out))


;;;; ============================================================
;;;; Reduction Operations
;;;; ============================================================

(define (var-sum v #!optional (axes '()) (keepdims? #f))
  "Gradient-tracking sum reduction.
   Backward: broadcast gradient back to input shape via broadcast-grad."
  (let* ((x        (var-value v))
         (x-shape  (morph-shape x))
         (x-rank   (vector-length x-shape))
         (norm-axes (if (null? axes)
                        (iota x-rank)
                        (map (lambda (a) (normalize-axis a x-rank)) axes)))
         (rg  (var-requires-grad? v))
         (out (make-var (morph-reduce 'sum x axes keepdims?) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v
            (broadcast-grad g x x-shape norm-axes keepdims?)))))
    out))

(define (var-mean v #!optional (axes '()) (keepdims? #f))
  "Gradient-tracking mean reduction.
   Backward: broadcast gradient / n back to input shape."
  (let* ((x        (var-value v))
         (x-shape  (morph-shape x))
         (x-rank   (vector-length x-shape))
         (norm-axes (if (null? axes)
                        (iota x-rank)
                        (map (lambda (a) (normalize-axis a x-rank)) axes)))
         (n        (compute-reduction-n x-shape norm-axes))
         (n-morph  (morph-from-list (list (exact->inexact n)) '(1) 'f64))
         (rg  (var-requires-grad? v))
         (out (make-var (morph-reduce 'mean x axes keepdims?) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v
            (morph/ (broadcast-grad g x x-shape norm-axes keepdims?)
                    n-morph)))))
    out))


;;;; ============================================================
;;;; Structural Operations
;;;;
;;;; These use zero-copy affine index functions in both directions.
;;;; No realization occurs in the backward pass. Gradient morphisms
;;;; are views backed by the same index-function machinery.
;;;; ============================================================

(define (var-reshape v new-shape)
  "Gradient-tracking reshape.
   Backward: morph-reshape(g, original-shape) [zero-copy view]"
  (let* ((x       (var-value v))
         (x-shape (morph-shape x))
         (rg      (var-requires-grad? v))
         (out     (make-var (morph-reshape x new-shape) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph-reshape g x-shape)))))
    out))

(define (var-transpose v perm)
  "Gradient-tracking transpose.
   Backward: morph-transpose(g, inverse-perm) [zero-copy view]"
  (let* ((x        (var-value v))
         (inv-perm (invert-permutation perm))
         (rg       (var-requires-grad? v))
         (out      (make-var (morph-transpose x perm) rg)))
    (when rg
      (morph-variable-parents-set! out (list v))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          (accumulate-grad! v (morph-transpose g inv-perm)))))
    out))


;;;; ============================================================
;;;; Linear Algebra
;;;;
;;;; var-matmul backward uses lazy morphisms: morph-transpose produces
;;;; a zero-copy strided view, and morph-matmul on it creates a lazy
;;;; morphism-expr.  When realized, realize-morphism-expr routes the
;;;; strided operands through the gemm-strided path (execute-scheme-gemm),
;;;; which is stride-aware and handles non-row-major layouts correctly.
;;;; ============================================================

(define (var-matmul vA vB)
  "Gradient-tracking 2-D matrix multiply (M,K) x (K,N) -> (M,N).
   dA = g @ B^T   [lazy: morph-matmul g (morph-transpose B)]
   dB = A^T @ g   [lazy: morph-matmul (morph-transpose A) g]"
  (let* ((A   (var-value vA))
         (B   (var-value vB))
         (rg  (grad-requires? (list vA vB)))
         (out (make-var (morph-matmul A B) rg)))
    (when rg
      (morph-variable-parents-set! out
        (filter var-requires-grad? (list vA vB)))
      (morph-variable-grad-fn-set! out
        (lambda (g)
          ;; morph-transpose creates a lazy zero-copy strided view.
          ;; morph-matmul wraps both in a lazy morphism-expr.
          ;; The realization engine handles strided operands via gemm-strided.
          (when (var-requires-grad? vA)
            (accumulate-grad! vA
              (morph-matmul g (morph-transpose B '(1 0)))))
          (when (var-requires-grad? vB)
            (accumulate-grad! vB
              (morph-matmul (morph-transpose A '(1 0)) g))))))
    out))


;;;; ============================================================
;;;; Backward Pass
;;;;
;;;; topo-sort uses DFS post-order with prepend, producing a list
;;;; where the root variable appears first and leaves appear last.
;;;; backward! processes the list in that order: root first (seed
;;;; already set), then each node calls its grad-fn to propagate
;;;; gradients to its parents.
;;;; ============================================================

(define (topo-sort root)
  "Return morph-variable nodes in root-first topological order.
   Uses DFS post-order with prepend: root first, leaves last.
   Each node appears exactly once (visited table prevents duplicates)."
  (let ((visited (make-hash-table))
        (order   '()))
    (define (visit v)
      (unless (hash-table-ref/default visited v #f)
        (hash-table-set! visited v #t)
        (for-each visit (morph-variable-parents v))
        (set! order (cons v order))))
    (visit root)
    order))

(define (backward! var #!optional seed)
  "Run reverse-mode automatic differentiation from var.

   seed: optional upstream gradient morphism.  Defaults to
         morph-ones-like(var-value), which allocates a single
         concrete array of ones. This is the only allocation
         in a standard backward! call.

   After backward!, var-grad on each requires-grad leaf holds a
   lazy gradient morphism.  Call (realize (var-grad leaf)) to
   materialize it."
  (let ((g0 (or seed (morph-ones-like (var-value var)))))
    (accumulate-grad! var g0))
  (for-each
   (lambda (v)
     (let ((gfn (morph-variable-grad-fn v))
           (g   (morph-variable-grad v)))
       (when (and gfn g)
         (gfn g))))
   (topo-sort var)))

) ; end module array-morphisms-grad
