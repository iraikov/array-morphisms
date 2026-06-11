;;; array-morphisms-batch-ops.scm
;;; Phase 7: Batch Operations
;;;
;;; Provides first-class batch combinators and utilities built on the existing
;;; structural and arithmetic morphism primitives.  All combinators produce
;;; lazy morphism expression trees that are realized only when `realize` is
;;; called -- no eager execution occurs in this module.
;;;
;;; Batch dimension convention:
;;;   Batch axis is always dimension 0 by default.  Every batched morphism
;;;   carries batch-axis >= 0 in its internal field.
;;;
;;; BLAS compatibility:
;;;
;;;   morph-batch-matmul decomposes into N calls to morph-matmul (2-D
;;;   slices), which the realization engine automatically routes to
;;;   BLAS.  The realization engine also ensures that trace/replay
;;;   contexts benefit from BLAS for any matmul in the tree.

(module array-morphisms-batch-ops

  (;; Batch dimension management
   add-batch-dimension
   remove-batch-dimension
   extract-batch-element
   stack-into-batch
   concat-batch

   ;; Batch broadcasting
   broadcast-to-batch

   ;; Core batch combinators
   batch-map
   batch-reduce
   batch-zip
   batch-fold
   batch-scan

   ;; Batch-aware arithmetic (auto-broadcasts non-batched operand)
   morph+/batch
   morph-/batch
   morph*/batch
   morph-div/batch

   ;; Batched matrix multiply (BLAS-compatible)
   morph-batch-matmul)

  (import scheme (chicken base))
  (import (only srfi-1 iota every any fold take drop))
  (import datatype)
  (import array-morphisms-core)
  (import array-morphisms-basic-ops)
  (import array-morphisms-structural-ops)
  (import array-morphisms-blas-exec)

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Batch Dimension Management
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (add-batch-dimension m #!optional (axis 0))
    "Add a size-1 batch dimension to a non-batched morphism.

    Unlike morph-unsqueeze, this explicitly marks the inserted axis as the
    batch axis.  morph-unsqueeze delegates to morph-reshape whose batch-axis
    scan logic leaves batch-axis=-1 when the source was not batched; this
    function patches the result to carry the correct batch-axis.

    Args:
      m:    non-batched morphism
      axis: position to insert the new batch dimension (default: 0)

    Returns:
      Morphism with shape (...1...) at `axis` and batch-axis=norm-axis."
    (when (batched? m)
      (error "add-batch-dimension: morphism already has a batch dimension" m))
    (let* ((shape (get-morphism-shape m))
           (rank  (vector-length shape))
           (norm-axis (if (< axis 0) (+ rank 1 axis) axis)))
      (unless (and (>= norm-axis 0) (<= norm-axis rank))
        (error "add-batch-dimension: axis out of range" axis rank))
      ;; morph-unsqueeze produces the right shape but sets batch-axis=-1.
      ;; Reconstruct the morphism with the correct batch-axis.
      (let ((reshaped (morph-unsqueeze m axis)))
        (cases array-morphism reshaped
          (morphism-expr (op operands index-fn shape dtype metadata _)
            (morphism-expr op operands index-fn shape dtype metadata norm-axis))
          (concrete-array (data shape strides offset dtype alloc-id _)
            (concrete-array data shape strides offset dtype alloc-id norm-axis))
          (else
           (error "add-batch-dimension: unexpected morphism type" reshaped))))))

  (define (remove-batch-dimension m)
    "Remove a size-1 batch dimension from a batched morphism.

    Errors if the morphism is not batched or if the batch dimension is not
    size 1 (use batch-reduce to collapse larger batch dimensions).

    Returns:
      Morphism with batch dimension removed and batch-axis=-1."
    (unless (batched? m)
      (error "remove-batch-dimension: morphism has no batch dimension" m))
    (let* ((batch-axis (get-morphism-batch-axis m))
           (shape      (get-morphism-shape m)))
      (unless (= (vector-ref shape batch-axis) 1)
        (error "remove-batch-dimension: batch dimension must have size 1"
               (vector-ref shape batch-axis)))
      ;; morph-squeeze removes the size-1 axis; morph-reshape inside will
      ;; not find batch-size=1 as a distinguishing dimension, so the result
      ;; naturally has batch-axis=-1.
      (morph-squeeze m (list batch-axis))))

  (define (extract-batch-element m i #!optional (batch-axis 0))
    "Extract a single element from the batch dimension.

    Returns a non-batched morphism (batch-axis=-1) with the batch dimension
    removed.  The extraction is zero-copy: morph-slice + morph-squeeze both
    produce affine-index-fn views.

    Args:
      m:          batched morphism
      i:          batch index (0-based)
      batch-axis: which axis is the batch dimension (default: 0)

    Returns:
      Non-batched morphism corresponding to batch element i."
    (let* ((shape (get-morphism-shape m))
           (rank  (vector-length shape))
           (N     (vector-ref shape batch-axis)))
      (when (or (< i 0) (>= i N))
        (error "extract-batch-element: index out of range" i N))
      (let* ((start (map (lambda (ax)
                           (if (= ax batch-axis) i 0))
                         (iota rank)))
             (end   (map (lambda (ax)
                           (if (= ax batch-axis)
                               (+ i 1)
                               (vector-ref shape ax)))
                         (iota rank))))
        ;; morph-slice gives shape with size-1 at batch-axis;
        ;; morph-squeeze removes it -> non-batched element.
        (morph-squeeze (morph-slice m start end) (list batch-axis)))))

  (define (stack-into-batch morphisms #!optional (axis 0))
    "Stack a list of non-batched morphisms into a batch.

    Equivalent to morph-stack; provided as a named convenience that enforces
    the batch convention.  morph-stack already sets batch-axis=axis on the
    result (structural-ops.scm line 776).

    Args:
      morphisms: list of non-batched morphisms with identical shapes
      axis:      batch axis position in the result (default: 0)

    Returns:
      Batched morphism with shape (N, ...elem-shape...) and batch-axis=axis."
    (when (null? morphisms)
      (error "stack-into-batch: requires at least one morphism"))
    (when (any batched? morphisms)
      (error "stack-into-batch: input morphisms must not be batched"))
    (morph-stack morphisms axis))

  (define (concat-batch morphisms #!optional (axis 0))
    "Concatenate batched morphisms along the batch dimension.

    All inputs must be batched.  Delegates to morph-concat which preserves
    the batch-axis from the first input.

    Args:
      morphisms: list of batched morphisms with matching non-batch shapes
      axis:      batch axis (default: 0)

    Returns:
      Batched morphism with concatenated batch dimension."
    (when (null? morphisms)
      (error "concat-batch: requires at least one morphism"))
    (unless (every batched? morphisms)
      (error "concat-batch: all morphisms must be batched"))
    (morph-concat morphisms axis))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Batch Broadcasting
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (broadcast-to-batch m target-batch-size #!optional (batch-axis 0))
    "Broadcast morphism to include or expand a batch dimension.

    Cases:
      - Non-batched m: inserts a size-1 batch dim at batch-axis.
        (morph+ etc. use NumPy-style broadcasting to expand size-1 dims.)
      - Batched m with size 1: returned as-is; broadcasting handles expansion.
      - Batched m with size target-batch-size: returned as-is.
      - Any other size: error.

    Args:
      m:                 input morphism
      target-batch-size: desired batch size (used only for validation)
      batch-axis:        batch axis to insert/check (default: 0)

    Returns:
      Morphism with a batch dimension compatible with target-batch-size."
    (cond
      ((not (batched? m))
       ;; Insert a size-1 leading dimension; broadcasting will expand it.
       (add-batch-dimension m batch-axis))
      ((= (batch-size m) 1)
       m)
      ((= (batch-size m) target-batch-size)
       m)
      (else
       (error "broadcast-to-batch: batch size mismatch"
              (batch-size m) target-batch-size))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Core Batch Combinators
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; Slice-and-stack strategy:  batch-map, batch-zip, batch-scan all decompose
  ;; into N per-element operations and then stack the results.  The expression
  ;; tree has O(N) depth, which is acceptable for typical batch sizes (32-512).
  ;; Everything stays lazy until `realize` is called.

  (define (batch-map fn m #!optional (batch-axis 0))
    "Apply fn to each element along the batch dimension.

    fn must be a function morphism -> morphism.  It is applied to each
    non-batched element slice and the results are stacked back into a batched
    morphism at the same batch-axis position.

    Args:
      fn:         function morphism -> morphism
      m:          batched input morphism
      batch-axis: batch dimension to map over (default: 0)

    Returns:
      Batched morphism of same batch size with fn applied per element."
    (let* ((N       (vector-ref (get-morphism-shape m) batch-axis))
           (results (map (lambda (i) (fn (extract-batch-element m i batch-axis)))
                         (iota N))))
      (morph-stack results batch-axis)))

  (define (batch-reduce op m #!optional (keepdims? #f))
    "Reduce along the batch dimension of m.

    Delegates directly to morph-reduce on the batch axis.

    Args:
      op:        reduction operation: 'sum, 'mean, 'max, 'min, 'prod
      m:         batched morphism
      keepdims?: when #t, retain the reduced axis with size 1 (default: #f)

    Returns:
      Reduced morphism.  When keepdims?=#f the batch dimension is removed."
    (let ((axis (if (batched? m) (get-morphism-batch-axis m) 0)))
      (morph-reduce op m (list axis) keepdims?)))

  (define (batch-zip fn m1 m2)
    "Apply binary fn element-wise across two batched morphisms.

    Both morphisms must have matching batch sizes.  For each batch index i,
    computes (fn m1[i] m2[i]) and stacks the results.

    Args:
      fn:     binary function morphism morphism -> morphism
      m1, m2: batched morphisms with matching batch sizes

    Returns:
      Batched morphism of same batch size with fn applied per pair."
    (let ((N (vector-ref (get-morphism-shape m1) 0)))
      (unless (= N (vector-ref (get-morphism-shape m2) 0))
        (error "batch-zip: batch size mismatch"
               N (vector-ref (get-morphism-shape m2) 0)))
      (let ((results (map (lambda (i)
                            (fn (extract-batch-element m1 i 0)
                                (extract-batch-element m2 i 0)))
                          (iota N))))
        (morph-stack results 0))))

  (define (batch-fold fn init m)
    "Fold across the batch dimension of m.

    Applies (fn accumulator element) sequentially for each batch element,
    starting with `init` as the initial accumulator.  The return value is
    the final accumulator, which is a non-batched morphism.

    Unlike batch-map/batch-scan, this returns a single (non-batched) result.

    Args:
      fn:   binary function (acc morphism -> morphism)
      init: initial accumulator (non-batched morphism or any Scheme value
            that fn can process)
      m:    batched morphism

    Returns:
      Final accumulator morphism."
    (let ((N (vector-ref (get-morphism-shape m) 0)))
      (let loop ((i 0) (acc init))
        (if (= i N)
            acc
            (loop (+ i 1)
                  (fn acc (extract-batch-element m i 0)))))))

  (define (batch-scan fn init m)
    "Cumulative scan across the batch dimension of m.

    Like batch-fold but returns all intermediate accumulator values stacked
    into a batched morphism of the same batch size.

    Args:
      fn:   binary function (acc morphism -> morphism)
      init: initial accumulator (used before the first element)
      m:    batched morphism

    Returns:
      Batched morphism of same shape as m holding cumulative results."
    (let* ((N (vector-ref (get-morphism-shape m) 0))
           (results
            (let loop ((i 0) (acc init) (out '()))
              (if (= i N)
                  (reverse out)
                  (let ((next (fn acc (extract-batch-element m i 0))))
                    (loop (+ i 1) next (cons next out)))))))
      (morph-stack results 0)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Batch-Aware Arithmetic
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; make-batch-aware-binop wraps a base binary operation with automatic
  ;; broadcasting: a non-batched operand is promoted to a size-1 batched
  ;; morphism (broadcast-to-batch) and NumPy-style broadcasting in morph+
  ;; etc. handles the actual element expansion at realization time.

  (define (make-batch-aware-binop base-op)
    "Return a batch-aware version of base-op.

    The returned procedure accepts two morphisms m1 m2 and:
      - Both batched, same size:  (base-op m1 m2)
      - Only m1 batched:          (base-op m1 (broadcast-to-batch m2 N ax))
      - Only m2 batched:          (base-op (broadcast-to-batch m1 N ax) m2)
      - Neither batched:          (base-op m1 m2)"
    (lambda (m1 m2)
      (let ((b1 (batched? m1))
            (b2 (batched? m2)))
        (cond
          ((and b1 b2)
           (unless (= (batch-size m1) (batch-size m2))
             (error "batch arithmetic: batch size mismatch"
                    (batch-size m1) (batch-size m2)))
           (base-op m1 m2))
          (b1
           (base-op m1
                    (broadcast-to-batch m2
                                        (batch-size m1)
                                        (get-morphism-batch-axis m1))))
          (b2
           (base-op (broadcast-to-batch m1
                                        (batch-size m2)
                                        (get-morphism-batch-axis m2))
                    m2))
          (else
           (base-op m1 m2))))))

  (define morph+/batch    (make-batch-aware-binop morph+))
  (define morph-/batch    (make-batch-aware-binop morph-))
  (define morph*/batch    (make-batch-aware-binop morph*))
  (define morph-div/batch (make-batch-aware-binop morph/))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Batched Matrix Multiply
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (morph-batch-matmul A B)
    "Batched matrix multiply.

    Supported shapes:
      A (M, K), B (K, P)       -- non-batched: plain morph-matmul
      A (N, M, K), B (K, P)    -- shared weight: batch-map morph-matmul over A
      A (N, M, K), B (N, K, P) -- per-batch weight: batch-zip morph-matmul

    In all batched cases, each slice is a 2-D matrix multiply routed through
    morph-matmul, which the realization engine dispatches to BLAS (when
    available) via execute-blas-operation.

    Args:
      A: matrix or batched-matrix morphism
      B: matrix or batched-matrix morphism

    Returns:
      Matrix or batched-matrix morphism of appropriate shape."
    (let* ((A-shape (get-morphism-shape A))
           (B-shape (get-morphism-shape B))
           (A-rank  (vector-length A-shape))
           (B-rank  (vector-length B-shape)))
      (cond
        ;; Both non-batched: delegate directly.
        ((and (= A-rank 2) (= B-rank 2))
         (morph-matmul A B))

        ;; A rank-3 (N,M,K), B rank-2 (K,P): map matmul over N.
        ;; A may carry batch-axis=-1 when built from morph-from-list.
        ((and (= A-rank 3) (= B-rank 2))
         (let ((K-a (vector-ref A-shape 2))
               (K-b (vector-ref B-shape 0)))
           (unless (= K-a K-b)
             (error "morph-batch-matmul: inner dimensions must match" K-a K-b)))
         (batch-map (lambda (a) (morph-matmul a B)) A 0))

        ;; A rank-3 (N,M,K), B rank-3 (N,K,P): per-element weight.
        ((and (= A-rank 3) (= B-rank 3))
         (let ((N-a (vector-ref A-shape 0))
               (N-b (vector-ref B-shape 0)))
           (unless (= N-a N-b)
             (error "morph-batch-matmul: batch size mismatch" N-a N-b)))
         (let ((K-a (vector-ref A-shape 2))
               (K-b (vector-ref B-shape 1)))
           (unless (= K-a K-b)
             (error "morph-batch-matmul: inner dimensions must match" K-a K-b)))
         (batch-zip morph-matmul A B))

        (else
         (error "morph-batch-matmul: unsupported shape combination"
                A-shape B-shape)))))

) ;; end module array-morphisms-batch-ops
