;;; MoA Realization Engine
;;;
;;; Materializes abstract morphisms into concrete arrays through
;;; specialized execution kernels for each index function type.

(module array-morphisms-realization
        
  (;; Core realization
   realize
   realize!
   force-morphism

   ;; Memory reuse context
   current-morphism-context

   ;; Specialized execution
   execute-index-fn
   execute-affine-morphism
   execute-compute-morphism
   execute-window-morphism
   execute-reduction-morphism
   execute-routing-morphism
   
   ;; Value retrieval
   retrieve-value
   retrieve-value-with-padding
   
   ;; Zero-copy detection
   can-use-zero-copy?
   create-view

   ;; BLAS configuration (re-exported from array-morphisms-blas-exec)
   blas-enabled?
   enable-blas!
   disable-blas!
   blas-available?
   register-blas-backend!
   active-blas-backend
   *active-backend*
   *blas-size-threshold*

   ;; Conv2D via im2col+BLAS
   detect-conv2d-pattern
   execute-conv2d-blas

   ;; Fused attention kernel
   ;; attention-morphism constructor lives in array-morphisms-blas-exec; re-exported here
   attention-morphism
   *attention-fusion-threshold*
   should-fuse-attention?
   execute-fused-attention

   ;; Flat-loop fast-path kernels (zero allocation per element; called by SSA replay plan)
   execute-flat-unary-compute
   execute-flat-binary-compute
   execute-flat-bias-broadcast-compute
   execute-flat-unary-compute-inplace!
   execute-flat-bias-broadcast-inplace!

   ;; Fast im2col / col2im kernels (dtype-specialized tight loops; called by SSA ri-im2col/ri-col2im)
   execute-im2col-unbatched
   execute-im2col-batched
   execute-im2col-batched-mr
   execute-im2col-nhwc-mr
   execute-im2col-nhwc-classic
   execute-col2im-unbatched
   execute-col2im-batched
   execute-col2im-batched-mr
   execute-col2im-nhwc-mr
   execute-col2im-nhwc-classic
   ;; Implicit GEMM convolution
   execute-conv-fwd-nchw
   execute-conv-fwd-nhwc
   execute-conv-bwd-data-nchw
   execute-conv-bwd-data-nhwc
   execute-conv-bwd-weights-nchw
   execute-conv-bwd-weights-nhwc
   ;; BLAS-backed convolution: im2col-mr + single BLAS gemm per op
   execute-conv-fwd-blas
   execute-conv-fwd-nhwc-blas
   execute-conv-bwd-data-blas
   execute-conv-bwd-data-nhwc-blas
   execute-conv-bwd-weights-blas
   execute-conv-bwd-weights-nhwc-blas
   )

  (import scheme chicken.base chicken.module)
  (import (only srfi-1 make-list fold iota every zip drop-right take last drop append-map filter-map filter count fold-right))
  (import (only srfi-4 f32vector f64vector s32vector s64vector u32vector u64vector
                       f32vector-length f64vector-length s32vector-length
                       s64vector-length u32vector-length u64vector-length
                       f32vector-ref f64vector-ref s32vector-ref s64vector-ref
                       u32vector-ref u64vector-ref
                       f32vector-set! f64vector-set! s32vector-set! s64vector-set!
                       u32vector-set! u64vector-set!))
  (import datatype matchable)

  (import array-morphisms-core)
  (import array-morphisms-index-fn)
  (import array-morphisms-structural-ops)
  (import array-morphisms-blas-compat)
  (import array-morphisms-blas-exec)
  (import array-morphisms-micro-blas-backend)

  ;; Default-backend bootstrap: register the dependency-free microBLAS
  ;; backend unless something (e.g. the system-BLAS egg backend) has
  ;; already registered a backend first.  This must live here rather than
  ;; in blas-exec.scm itself: array-morphisms-micro-blas-backend imports
  ;; array-morphisms-blas-exec, so registering from within blas-exec.scm
  ;; would be a circular module dependency.  realization.scm already
  ;; depends on blas-exec and is never depended on by micro-blas-backend,
  ;; so this is the natural place for the auto-registration side effect.
  (unless (blas-available?)
    (register-blas-backend! (make-micro-blas-backend)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Convolution kernel loop macros

  ;; Simple counted loop: v from 0 to N-1.
  (define-syntax for-range
    (syntax-rules ()
      ((_ (v N) body ...)
       (do ((v 0 (+ v 1))) ((= v N))
         body ...))))

  ;; Counted loop with zero or more parallel stride variables.
  ;; Each (aux init step) clause becomes a do-variable starting at init
  ;; and advancing by step each iteration.
  (define-syntax for-stride
    (syntax-rules ()
      ((_ (v N) ((aux init step) ...) body ...)
       (do ((v 0 (+ v 1)) (aux init (+ aux step)) ...)
           ((= v N))
         body ...))))

  ;; Skip body when spatial coordinate lies outside [0, limit).
  (define-syntax when-valid
    (syntax-rules ()
      ((_ coord limit body ...)
       (when (and (>= coord 0) (< coord limit))
         body ...))))

  ;; NOTE: CHICKEN Scheme's let-syntax does not propagate scope through
  ;; define-syntax macro expansions (e.g. for-stride/for-range), so
  ;; with-typed-vecs cannot inject vec-ref/vec-set! into loop bodies.
  ;; Use explicit case dtype with f32vector-ref/f64vector-ref instead.

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Memory Reuse Context Parameter
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; When #f (default), realize behaves as in phases 1-5.
  ;; When a dispatch vector #(mode next-id! record! get-buf!), every
  ;; recursive realize call within the dynamic extent uses the context.
  ;; Set by realize/ctx in array-morphisms-context via parameterize.
  (define current-morphism-context (make-parameter #f))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Main Realization Entry Point
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (realize m)
    "Realize morphism to concrete array.

    When current-morphism-context is #f (the default), all existing
    behaviour is unchanged.  When it holds a dispatch vector installed
    by realize/ctx, context-aware variants are used for allocation."

    (let ((ctx (current-morphism-context)))
      (cases array-morphism m
        ;; Already concrete - return as-is, no counter increment
        (concrete-array (data shape strides offset dtype alloc-id batch-axis)
          m)

        ;; Abstract morphism - needs realization
        (morphism-expr (morph-id op operands index-fn shape dtype metadata batch-axis)
          (if ctx
              (realize-morphism-expr/ctx ctx m)
              (realize-morphism-expr m)))

        ;; Reduction morphism
        (reduction-morphism (morph-id op operand reduce-axes index-fn shape dtype batch-axis)
          (if ctx
              (realize-reduction/ctx ctx m)
              (realize-reduction m))))))
  
  (define (realize! m)
    "Realize morphism in-place (alias for realize)
    
    Note: Currently allocates fresh arrays. Future optimization
    will support in-place realization for mutable operations."
    (realize m))
  
  (define (force-morphism m)
    "Force realization of morphism (alias for realize)"
    (realize m))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Morphism Expression Realization
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Conv2D Pattern Detection and Execution
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (detect-conv2d-pattern m)
    "Detect the fused convolution pattern: reshape(matmul(weight, im2col(input))).

    Must be called on the unrealized morphism-expr tree so the nested
    im2col/matmul/reshape structure is visible.

    Returns a list (input weight im2col-meta) on a match, or #f otherwise.
      input      - the raw input morphism (may be concrete or abstract)
      weight     - the weight morphism   (may be concrete or abstract)
      im2col-meta - the metadata alist from the im2col morphism-expr, which
                   contains at minimum keys: kernel-size, stride, padding."
    (cases array-morphism m
      (morphism-expr (morph-id op operands _ _ _ _ _)
        (and (eq? op 'reshape)
             (= (length operands) 1)
             (cases array-morphism (car operands)
               (morphism-expr (morph-id mm-op mm-operands _ _ _ _ _)
                 (and (eq? mm-op 'matmul)
                      (= (length mm-operands) 2)
                      (cases array-morphism (cadr mm-operands)
                        (morphism-expr (morph-id ic-op ic-operands _ _ _ ic-meta _)
                          (and (eq? ic-op 'im2col)
                               (= (length ic-operands) 1)
                               (list (car ic-operands)   ; input
                                     (car mm-operands)   ; weight
                                     ic-meta)))          ; metadata alist
                        (else #f))))
               (else #f))))
      (else #f)))

  (define (execute-conv2d-blas input weight im2col-meta output-shape dtype)
    "Execute convolution using im2col + BLAS GEMM.

    Fuses three logical operations:
      1. im2col : extract receptive-field windows from input
      2. GEMM   : weight-matrix times column-matrix
      3. reshape: flatten back to image format

    Args:
      input       - raw input morphism (C,H,W) or (N,C,H,W); will be realized
      weight      - weight morphism (C_out,...); will be realized and flattened
                    to 2-D (C_out, C_in*KH*KW) before GEMM
      im2col-meta - alist with keys kernel-size, stride, padding (from im2col node)
      output-shape - final shape of the result (matches the reshape target)
      dtype        - element dtype ('f32 or 'f64)

    Returns a fresh concrete-array with the given output-shape."
    (let* ((kernel-size (cdr (assq 'kernel-size im2col-meta)))
           (stride      (cdr (assq 'stride      im2col-meta)))
           (padding     (cdr (assq 'padding     im2col-meta)))

           ;; Realize input and weight (may be abstract morphism-exprs)
           (ri (realize input))
           (rw (realize weight))

           (in-shape (get-morphism-shape ri))
           (batched? (= (vector-length in-shape) 4))
           (N        (if batched? (vector-ref in-shape 0) 1))

           ;; Flatten weight to 2-D: (C_out, C_in*KH*KW).
           ;; Weight may be (C_out, C_in, KH, KW) or already 2-D.
           (w-shape  (get-morphism-shape rw))
           (C-out    (vector-ref w-shape 0))
           (inner    (quotient (shape-size w-shape) C-out))
           (w2d      (realize (morph-reshape rw (vector C-out inner))))

           ;; im2col of the realized input
           (col      (realize (im2col-morph ri kernel-size stride padding)))
           (col-shape (get-morphism-shape col)))

      (if batched?

          ;; --- Batched path: N independent GEMMs ----------------------------
          ;; col shape: (N, C_in*KH*KW, OH*OW)
          (let* ((KK          (vector-ref col-shape 1))   ; C_in*KH*KW
                 (OO          (vector-ref col-shape 2))   ; OH*OW
                 (slab        (* KK OO))                  ; elements per sample
                 (out-per-n   (* C-out OO))               ; output elements per sample
                 (result-data (allocate-typed-vector dtype (* N out-per-n))))
            (do ((n 0 (+ n 1)))
                ((= n N))
              ;; Copy col[n] into a fresh contiguous 2-D array so that
              ;; execute-blas-gemm sees a contiguous row-major operand.
              (let* ((cn-data  (allocate-typed-vector dtype slab))
                     (col-base (* n slab)))
                (cases array-morphism col
                  (concrete-array (cd _ _ _ _ _ _)
                    (do ((i 0 (+ i 1)))
                        ((= i slab))
                      (typed-vector-set! cn-data dtype i
                        (typed-vector-ref cd dtype (+ col-base i)))))
                  (else (error "execute-conv2d-blas: col must be concrete" col)))
                (let* ((cn  (concrete-array cn-data (vector KK OO)
                                            (compute-strides (vector KK OO))
                                            0 dtype -1 -1))
                       (gn  (execute-blas-gemm w2d cn))
                       (out-base (* n out-per-n)))
                  (cases array-morphism gn
                    (concrete-array (gd _ _ _ _ _ _)
                      (do ((i 0 (+ i 1)))
                          ((= i out-per-n))
                        (typed-vector-set! result-data dtype (+ out-base i)
                          (typed-vector-ref gd dtype i))))
                    (else (error "execute-conv2d-blas: GEMM result must be concrete"))))))
            (concrete-array result-data output-shape
                            (compute-strides output-shape)
                            0 dtype -1 0))      ; batch-axis = 0 for batched

          ;; --- Non-batched path: single GEMM --------------------------------
          ;; col shape: (C_in*KH*KW, OH*OW)
          (cases array-morphism (execute-blas-gemm w2d col)
            (concrete-array (gd _ _ _ _ _ _)
              ;; gd has shape (C_out, OH*OW); reinterpret as output-shape
              (concrete-array gd output-shape
                              (compute-strides output-shape)
                              0 dtype -1 -1))
            (else (error "execute-conv2d-blas: GEMM result must be concrete"))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Fused Attention Kernel (FUSED_ATTENTION_IMPLEMENTATION_PROPOSAL Items 3-4)
  ;;;
  ;;; Implements scaled dot-product attention as a single fused realization:
  ;;;   - No n x n score matrix is materialised
  ;;;   - K is accessed by row (K[k,j]) -- transpose buffer never created
  ;;;   - Numerically stable softmax via running row-max subtraction (O(n) scratch)
  ;;;
  ;;; The user-facing constructor `attention-morphism` lives in
  ;;; array-morphisms-blas-exec (alongside morph-matmul et al.) and creates a
  ;;; morphism-expr with op 'attention.  The realization engine detects that op
  ;;; here and dispatches directly, before the conv2d/BLAS checks.
  ;;;
  ;;; Sequence-length dispatch threshold (Item 4b):
  ;;;   *attention-fusion-threshold* = 0  =>  always use fused kernel.
  ;;; Raise this once an unfused softmax path exists and a crossover can be
  ;;; measured empirically.  This is the attention analogue of *blas-size-threshold*.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define *attention-fusion-threshold* 0)

  (define (should-fuse-attention? n)
    "True when sequence length n warrants the fused O(n)-scratch kernel."
    (>= n *attention-fusion-threshold*))

  (define (%attn-2d! qd qr0 qr1 q-base
                     kd kr0 kr1 k-base
                     vd vr0 vr1 v-base
                     out-data out-base
                     n dk dv scale dtype
                     s-vec e-vec)
    "Inner 2-D fused attention kernel for one (n,dk)/(n,dv) block.

    Implements Algorithm 1 of Mullin & Hains (arXiv:2606.07713):
      Pass 1 -- score computation: s[k] = scale * sum_j Q[i,j]*K[k,j]
                K accessed by row (no transpose buffer; Item 2 of the proposal)
                running row-max tracked for numerical stability.
      Pass 2 -- stable softmax numerator: e[k] = exp(s[k] - row-max), accumulate Z.
      Pass 3 -- output: Out[i,p] = sum_k (e[k]/Z) * V[k,p].

    Scratch s-vec and e-vec (typed vectors of length n) are provided by the
    caller and reused across rows and batch elements; no allocation occurs here."
    (do ((i 0 (+ i 1))) ((= i n))
      (let ((qi-base (+ q-base (* i qr0))))
        ;; --- Pass 1: scores + row max ---
        (let ((row-max -inf.0))
          (do ((k 0 (+ k 1))) ((= k n))
            (let* ((ki-base (+ k-base (* k kr0)))
                   (score
                    (let lj ((j 0) (acc 0.0))
                      (if (= j dk)
                          (* scale acc)
                          (lj (+ j 1)
                              (+ acc
                                 (* (typed-vector-ref qd dtype (+ qi-base (* j qr1)))
                                    (typed-vector-ref kd dtype (+ ki-base (* j kr1))))))))))
              (typed-vector-set! s-vec dtype k score)
              (when (> score row-max) (set! row-max score))))
          ;; --- Pass 2: exp(score - row-max), accumulate Z ---
          (let ((Z 0.0))
            (do ((k 0 (+ k 1))) ((= k n))
              (let ((ek (exp (- (typed-vector-ref s-vec dtype k) row-max))))
                (typed-vector-set! e-vec dtype k ek)
                (set! Z (+ Z ek))))
            ;; --- Pass 3: Out[i,p] = sum_k (e[k]/Z) * V[k,p] ---
            (do ((p 0 (+ p 1))) ((= p dv))
              (let ((acc 0.0))
                (do ((k 0 (+ k 1))) ((= k n))
                  (set! acc (+ acc (* (/ (typed-vector-ref e-vec dtype k) Z)
                                      (typed-vector-ref vd dtype
                                        (+ v-base (* k vr0) (* p vr1)))))))
                (typed-vector-set! out-data dtype
                                   (+ out-base (* i dv) p) acc))))))))

  (define (execute-fused-attention m)
    "Realize a morphism-expr with op 'attention using the streaming fused kernel.

    Handles non-batched (rank 2) and batched (rank 3 or 4) inputs.  The leading
    (B, h) dimensions are partitioned into independent 2-D sub-problems
    (Remark 4.1 of Mullin & Hains), each dispatched to %attn-2d!.

    Scratch vectors s and e (length n) are allocated once and shared across
    all batch elements, keeping peak scratch at O(n) regardless of batch size."
    (cases array-morphism m
      (morphism-expr (morph-id op operands index-fn shape dtype metadata batch-axis)
        (let* ((scale     (cdr (assq 'scale     metadata)))
               (n         (cdr (assq 'n         metadata)))
               (dk        (cdr (assq 'dk        metadata)))
               (dv        (cdr (assq 'dv        metadata)))
               (n-leading (cdr (assq 'n-leading metadata)))
               (Q (realize (list-ref operands 0)))
               (K (realize (list-ref operands 1)))
               (V (realize (list-ref operands 2))))
          (cases array-morphism Q
            (concrete-array (qd qshape qst qoff _ _ _)
              (cases array-morphism K
                (concrete-array (kd _kshape kst koff _ _ _)
                  (cases array-morphism V
                    (concrete-array (vd _vshape vst voff _ _ _)
                      (let* (;; Total 2-D blocks: product of all leading dims
                             (n-batch
                              (if (= n-leading 0) 1
                                  (let loop ((i 0) (acc 1))
                                    (if (= i n-leading) acc
                                        (loop (+ i 1)
                                              (* acc (vector-ref qshape i)))))))
                             ;; 2-D strides (last two of each strides vector)
                             (qrank (vector-length qst))
                             (vrank (vector-length vst))
                             (qr0 (vector-ref qst (- qrank 2)))
                             (qr1 (vector-ref qst (- qrank 1)))
                             (kr0 (vector-ref kst (- qrank 2)))
                             (kr1 (vector-ref kst (- qrank 1)))
                             (vr0 (vector-ref vst (- vrank 2)))
                             (vr1 (vector-ref vst (- vrank 1)))
                             ;; Per-block base-offset increment (last leading dim stride).
                             ;; For 3-D (B,n,dk): qst[0]=n*dk; for 4-D (B,h,n,dk): qst[1]=n*dk.
                             ;; Linear batch index b maps to base qoff + b * q-batch-st.
                             (q-batch-st (if (= n-leading 0) 0
                                             (vector-ref qst (- qrank 3))))
                             (k-batch-st (if (= n-leading 0) 0
                                             (vector-ref kst (- qrank 3))))
                             (v-batch-st (if (= n-leading 0) 0
                                             (vector-ref vst (- vrank 3))))
                             ;; Output and scratch
                             (out-data (allocate-typed-vector dtype (* n-batch n dv)))
                             (s-vec    (allocate-typed-vector dtype n))
                             (e-vec    (allocate-typed-vector dtype n)))
                        (do ((b 0 (+ b 1))) ((= b n-batch))
                          (%attn-2d!
                           qd qr0 qr1 (+ qoff (* b q-batch-st))
                           kd kr0 kr1 (+ koff (* b k-batch-st))
                           vd vr0 vr1 (+ voff (* b v-batch-st))
                           out-data (* b n dv)
                           n dk dv scale dtype
                           s-vec e-vec))
                        (concrete-array out-data shape
                                        (compute-strides shape)
                                        0 dtype -1 batch-axis)))
                    (else (error "execute-fused-attention: V must be concrete after realization"))))
                (else (error "execute-fused-attention: K must be concrete after realization"))))
            (else (error "execute-fused-attention: Q must be concrete after realization")))))
      (else (error "execute-fused-attention: expected morphism-expr with op 'attention" m))))

(define (realize-morphism-expr m)
  "Realize morphism-expr to concrete array.

  Fused attention: op 'attention is checked first (no tree scan required;
  faster than conv2d pattern detection).  Dispatches to execute-fused-attention.

  Then checks for the fused conv2d pattern
    reshape(matmul(weight, im2col(input)))
  on the unrealized expression tree.  When matched and BLAS is enabled,
  executes as a single fused im2col + GEMM + reshape without materializing
  the intermediate im2col column matrix as a separate allocation.

  Falls back to BLAS-accelerated matmul/matvec/dot via
  execute-blas-operation after realizing all operands.

  Standard path: zero-copy view or generic kernel execution."

  (cases array-morphism m
    (morphism-expr (morph-id op operands index-fn shape dtype metadata batch-axis)

      ;; Fused attention: explicit op tag, no tree scan needed.
      (if (eq? op 'attention)
          (execute-fused-attention m)

          ;; Detect conv2d pattern on the unrealized tree.
          ;; Must run BEFORE (map realize operands) so the im2col/matmul
          ;; nodes are still visible as morphism-exprs.
          (let ((conv-result
                 (and (blas-enabled?)
                      (let ((pat (detect-conv2d-pattern m)))
                        (and pat
                             (apply execute-conv2d-blas
                                    (append pat (list shape dtype))))))))

            (if conv-result
                conv-result

                (let* ((realized-operands (map realize operands))

                       ;; BLAS dispatch for matmul/matvec/dot.
                       ;; Returns #f for other ops, allowing standard path below.
                       ;; For matmul/matvec/dot: always dispatches here since those
                       ;; morphisms carry an identity-fn placeholder that
                       ;; execute-index-fn cannot handle.
                       (blas-result
                        (execute-blas-operation
                         (morphism-expr morph-id op realized-operands index-fn
                                        shape dtype metadata batch-axis))))

                  (if blas-result
                      blas-result

                      ;; Standard path: zero-copy view or kernel execution
                      (let* ((size (shape-size shape))
                             (can-zero-copy?
                              (and (= (length realized-operands) 1)
                                   (concrete-array? (car realized-operands))
                                   (can-use-zero-copy? index-fn
                                                       (car realized-operands)
                                                       shape))))
                        (if can-zero-copy?
                            (create-view (car realized-operands) index-fn shape batch-axis)
                            (let ((result-data (allocate-typed-vector dtype size)))
                              (execute-index-fn index-fn result-data shape
                                                realized-operands dtype)
                              (concrete-array result-data shape
                                              (compute-strides shape) 0 dtype -1
                                              batch-axis))))))))))

    (else (error "realize-morphism-expr: expected morphism-expr" m))))

  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Context-Aware Realization
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (realize-morphism-expr/ctx ctx m)
    "Realize morphism-expr with an active dispatch context vector.

    ctx is #(mode next-id! record! get-buf!) where:
      mode     - symbol 'trace or 'replay
      next-id! - (lambda ()) -> integer
      record!  - (lambda (id dtype size shape input-ids)) -> void
      get-buf! - (lambda (id dtype size)) -> typed-vector

    Uses execute-blas-gemm/into! and execute-blas-gemv/into! to fill
    the pool buffer with BLAS kernels in both trace and replay modes, resolving
    the prior limitation where ctx always used the pure Scheme kernel path."

    (cases array-morphism m
      (morphism-expr (morph-id op operands index-fn shape dtype metadata batch-axis)
        ;; Fused attention: delegate outside the context buffer pool.
        ;; The attention kernel allocates its own output; context integration
        ;; for attention output buffers is future work.
        (if (eq? op 'attention)
            (execute-fused-attention m)
        (let* ((size              (shape-size shape))
               (realized-operands (map realize operands))
               (can-zero-copy?
                (and (= (length realized-operands) 1)
                     (concrete-array? (car realized-operands))
                     (can-use-zero-copy? index-fn
                                         (car realized-operands)
                                         shape))))
          (if can-zero-copy?
              ;; Zero-copy: no counter increment, no trace entry.
              ;; The source alloc-id is inherited inside create-view,
              ;; so downstream deps on this view correctly extend the
              ;; source's lifetime.
              (create-view (car realized-operands) index-fn shape batch-axis)

              ;; Non-zero-copy: obtain buffer, fill via BLAS or kernel, record.
              (let* ((mode     (vector-ref ctx 0))
                     (next-id! (vector-ref ctx 1))
                     (record!  (vector-ref ctx 2))
                     (get-buf! (vector-ref ctx 3))
                     (alloc-id (next-id!))
                     (input-ids (map get-allocation-id realized-operands))
                     ;; Obtain result buffer for this allocation slot.
                     (result-data
                      (case mode
                        ((trace)  (allocate-typed-vector dtype size))
                        ((replay) (get-buf! alloc-id dtype size))
                        (else (error "realize-morphism-expr/ctx: unknown mode"
                                     mode))))
                     ;; Check BLAS eligibility on the concrete-operand morphism.
                     ;; Do NOT gate on blas-enabled? here: the /into! variants
                     ;; and execute-blas-dot each carry their own fallback to
                     ;; pure-Scheme when BLAS is disabled.  Gating would cause
                     ;; matmul/matvec/dot (which use identity-fn as placeholder)
                     ;; to fall through to execute-index-fn, which cannot handle
                     ;; identity-fn with two operands.
                     (blas-info
                      (blas-compatible-operation?
                       (morphism-expr morph-id op realized-operands index-fn
                                      shape dtype metadata batch-axis))))
                ;; Fill result-data via BLAS/into! or the generic index kernel.
                (if blas-info
                    (let ((blas-op  (car blas-info))
                          (blas-ops (cdr blas-info)))
                      (case blas-op
                        ((gemm)
                         (execute-blas-gemm/into!
                          (car blas-ops) (cadr blas-ops) result-data))
                        ((gemm-strided)
                         ;; Strided/transposed operands: extract Trans flags and
                         ;; physical lda/ldb from array strides, then call the
                         ;; backend's gemm-strided kernel (dgemm! with Trans)
                         ;; when available.  Falls back to the stride-aware
                         ;; pure Scheme triple loop otherwise.
                         (execute-blas-gemm-strided/into!
                          (car blas-ops) (cadr blas-ops) result-data))
                        ((gemv)
                         (execute-blas-gemv/into!
                          (car blas-ops) (cadr blas-ops) result-data))
                        ((dot)
                         ;; DOT returns a scalar.  Copy the single value into
                         ;; result-data[0]; avoids a separate /into! variant.
                         (let ((r (execute-blas-dot (car blas-ops) (cadr blas-ops))))
                           (cases array-morphism r
                             (concrete-array (rd _ _ _ dt _ _)
                               (typed-vector-set! result-data dt 0
                                 (typed-vector-ref rd dt 0)))
                             (else
                              (error "realize-morphism-expr/ctx dot: expected concrete"
                                     r)))))
                        (else
                         (execute-index-fn index-fn result-data shape
                                           realized-operands dtype))))
                    (execute-index-fn index-fn result-data shape
                                      realized-operands dtype))
                ;; Record only in trace mode (replay reuses existing entries).
                (when (eq? mode 'trace)
                  (record! alloc-id dtype size shape input-ids))
                (concrete-array result-data shape
                                (compute-strides shape) 0 dtype
                                alloc-id batch-axis))))))  ; closes (if (eq? op 'attention) ... (let* ...))
      (else (error "realize-morphism-expr/ctx: expected morphism-expr" m))))

  (define (realize-reduction/ctx ctx m)
    "Realize reduction-morphism with an active dispatch context vector."

    (cases array-morphism m
      (reduction-morphism (morph-id op operand reduce-axes index-fn shape dtype batch-axis)
        ;; Realize operand - inherits active context via parameter
        (let ((src (realize operand)))
          (cases array-morphism src
            (concrete-array (src-data src-shape src-strides src-offset src-dtype _ _)
              (let* ((size     (shape-size shape))
                     (mode     (vector-ref ctx 0))
                     (next-id! (vector-ref ctx 1))
                     (record!  (vector-ref ctx 2))
                     (get-buf! (vector-ref ctx 3))
                     (alloc-id (next-id!))
                     (input-id (get-allocation-id src)))
                (case mode
                  ((trace)
                   (let ((result-data (allocate-typed-vector dtype size)))
                     (execute-reduction-morphism
                      op result-data shape
                      src-data src-shape src-strides src-offset
                      reduce-axes (reduction-index-fn-reducer index-fn)
                      (reduction-index-fn-keepdims? index-fn) dtype src-dtype)
                     (record! alloc-id dtype size shape (list input-id))
                     (concrete-array result-data shape
                                     (compute-strides shape) 0 dtype
                                     alloc-id batch-axis)))
                  ((replay)
                   (let ((result-data (get-buf! alloc-id dtype size)))
                     (execute-reduction-morphism
                      op result-data shape
                      src-data src-shape src-strides src-offset
                      reduce-axes (reduction-index-fn-reducer index-fn)
                      (reduction-index-fn-keepdims? index-fn) dtype src-dtype)
                     (concrete-array result-data shape
                                     (compute-strides shape) 0 dtype
                                     alloc-id batch-axis)))
                  (else
                   (error "realize-reduction/ctx: unknown mode" mode)))))
            (else (error "realize-reduction/ctx: source must be concrete")))))
      (else (error "realize-reduction/ctx: expected reduction-morphism" m))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Zero-Copy Optimization
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  
  (define (contiguous-array? m)
    "Check if concrete array has standard row-major (contiguous) layout.
     A contiguous array's strides match compute-strides of its shape
     and its offset is zero."
    (cases array-morphism m
           (concrete-array
            (data shape strides offset dtype alloc-id batch-axis)
            (and (= offset 0)
                 (let ((expected (compute-strides shape)))
                   (let loop ((i 0))
                     (cond
                      ((= i (vector-length strides)) #t)
                      ((= (vector-ref strides i)
                          (vector-ref expected i))
                       (loop (+ i 1)))
                      (else #f))))))
           (else #f)))


  (define (dot-product v1 v2)
    "Dot product of two lists of numbers."
    (fold + 0 (map * v1 v2)))
  
  (define (can-use-zero-copy? index-fn operand result-shape)
    "Check if zero-copy view creation is possible.
   
     Zero-copy is possible for affine index functions on concrete arrays
     when the transformation can be expressed purely as a stride/offset change:
   
   - Reshape (identity A, zero b): requires contiguous source
   - Transpose (permutation A, zero b): always feasible
   - Slice (diagonal A, offset b): always feasible
   
   General affine transforms fall back to copy-based realization."
  
    (and (affine-index-fn? index-fn)
       (concrete-array? operand)
       (cases affine-index-fn index-fn
         (identity-fn   ()    (contiguous-array? operand))
         (permutation-fn (p)  #t)
         (diagonal-fn   (d b) #t)
         (general-fn    (A b) #f))))

  (define (create-view operand index-fn shape batch-axis)
    "Create a zero-copy view of operand through affine index transformation.
   
     Computes new strides and offset from the affine parameters:
   
     For src_idx = A * out_idx + b with source strides s and offset o:
       new_strides = A^T * s   (matrix-transpose of A times source strides)
       new_offset  = o + b . s (source offset plus dot product of bias and strides)
   
     Specializations:
       Reshape:    new strides from shape, same offset (requires contiguity)
       Transpose:  permuted source strides, same offset
       Slice:      element-wise scaled strides, shifted offset"
  
    (cases array-morphism operand
           
    (concrete-array (data src-shape src-strides src-offset dtype alloc-id _)
      (cases affine-index-fn index-fn

        ;; Reshape: reinterpret flat buffer with new row-major strides.
        (identity-fn ()
          (concrete-array data shape (compute-strides shape)
                          src-offset dtype alloc-id batch-axis))

        ;; Transpose: permute source strides.
        (permutation-fn (perm)
          (let ((new-strides
                 (list->vector
                  (map (lambda (k) (vector-ref src-strides k)) perm))))
            (concrete-array data shape new-strides
                            src-offset dtype alloc-id batch-axis)))

        ;; Slice: scale strides and shift offset.
        (diagonal-fn (steps starts)
          (let* ((src-strides-list (vector->list src-strides))
                 (new-strides
                  (list->vector (map * steps src-strides-list)))
                 (new-offset
                  (+ src-offset
                     (fold + 0 (map * starts src-strides-list)))))
            (concrete-array data shape new-strides
                            new-offset dtype alloc-id batch-axis)))

        ;; Should not be reached if can-use-zero-copy? is correct.
        (general-fn (A b)
          (error "create-view: general affine transforms are not zero-copy"
                 index-fn))))

    (else (error "create-view: operand must be a concrete array" operand))))

    
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Index Function Execution Dispatch
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


  (define (execute-index-fn index-fn output-buffer shape operands dtype)
    "Execute index function to fill output buffer

     Dispatches to specialized kernels based on index function type"

    ;; True when m is a zero-offset, row-major concrete array with shape = expected-shape.
    ;; Used to select zero-allocation flat-loop kernels for common element-wise ops.
    (define (flat-operand? m expected-shape)
      (cases array-morphism m
        (concrete-array (_ s st o _ _ _)
          (and (= o 0) (equal? s expected-shape) (equal? st (compute-strides s))))
        (else #f)))

    (cond
     ;; Pure affine (reshape, transpose, slice)
     ((affine-index-fn? index-fn)
      (execute-affine-morphism index-fn output-buffer shape operands dtype))

     ;; Computational (arithmetic, transcendental)
     ;; Fast paths avoid per-element linear-to-multi-index allocation when all
     ;; operands are row-major and the output shape matches.
     ((compute-index-fn? index-fn)
      (let* ((combiner (compute-index-fn-combiner index-fn))
             (size     (shape-size shape))
             (nops     (length operands)))
        (cond
          ;; Fast path 1: unary, row-major same shape
          ((and (= nops 1)
                (flat-operand? (car operands) shape))
           (cases array-morphism (car operands)
             (concrete-array (data _ _ _ src-dtype _ _)
               (execute-flat-unary-compute combiner data src-dtype output-buffer size dtype))
             (else
              (execute-compute-morphism index-fn output-buffer shape operands dtype))))
          ;; Fast path 2: binary, both row-major, same shape
          ((and (= nops 2)
                (flat-operand? (car operands)  shape)
                (flat-operand? (cadr operands) shape))
           (cases array-morphism (car operands)
             (concrete-array (data1 _ _ _ src-dtype1 _ _)
               (cases array-morphism (cadr operands)
                 (concrete-array (data2 _ _ _ src-dtype2 _ _)
                   (execute-flat-binary-compute combiner data1 src-dtype1 data2 src-dtype2 output-buffer size dtype))
                 (else
                  (execute-compute-morphism index-fn output-buffer shape operands dtype))))
             (else
              (execute-compute-morphism index-fn output-buffer shape operands dtype))))
          ;; Fast path 3: bias broadcast A[...,N] + B[N]
          ((and (= nops 2)
                (> (vector-length shape) 0)
                (flat-operand? (car operands) shape)
                (let ((N (vector-ref shape (- (vector-length shape) 1))))
                  (cases array-morphism (cadr operands)
                    (concrete-array (_ bs bst bo _ _ _)
                      (and (= bo 0)
                           (= (vector-length bs) 1)
                           (= (vector-ref bs 0) N)
                           (equal? bst (compute-strides bs))))
                    (else #f))))
           (let ((N (vector-ref shape (- (vector-length shape) 1))))
             (cases array-morphism (car operands)
               (concrete-array (data1 _ _ _ _ _ _)
                 (cases array-morphism (cadr operands)
                   (concrete-array (data2 _ _ _ _ _ _)
                     (execute-flat-bias-broadcast-compute
                      combiner data1 data2 output-buffer size N dtype))
                   (else
                    (execute-compute-morphism index-fn output-buffer shape operands dtype))))
               (else
                (execute-compute-morphism index-fn output-buffer shape operands dtype)))))
          ;; Generic fallback
          (else
           (execute-compute-morphism index-fn output-buffer shape operands dtype)))))
    
     ;; Window (im2col, padding)
     ((window-index-fn? index-fn)
      (execute-window-morphism index-fn output-buffer shape operands dtype))
    
     ;; col2im (accumulation)
     ((col2im-index-fn? index-fn)
      (execute-col2im-morphism index-fn output-buffer shape operands dtype))
    
     ;; Composed - inline execution
     ((composed-index-fn? index-fn)
      (execute-composed-morphism index-fn output-buffer shape operands dtype))
     
     ;; Multi-source routing: stack
    ((stack-index-fn? index-fn)
     (execute-routing-morphism  index-fn output-buffer shape operands dtype))

    ;; Multi-source routing: concat
    ((concat-index-fn? index-fn)
     (execute-routing-morphism  index-fn output-buffer shape operands dtype))
     
     ;; Direct procedure call
     ((procedure? index-fn)
      (execute-procedure-index-fn index-fn output-buffer shape operands dtype))
    
     (else
      (error "Unknown index function type" index-fn))))

  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Affine Morphism Execution (Reshape, Transpose, Slice)
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (affine-effective-strides fn src-strides src-offset)
    "Compute (eff-offset . eff-strides-vector) for any affine index fn.
     physical = eff-offset + sum(out[k] * eff-strides[k]) for each output dim k.
     Lets execute-affine-morphism iterate over output shape with tight loops
     instead of calling linear-to-multi-index + apply-affine-index-fn per element."
    (cases affine-index-fn fn
      (identity-fn ()
       (cons src-offset src-strides))
      (permutation-fn (perm)
       ;; source-idx[j] = output[perm[j]] => eff-strides[perm[j]] = src-strides[j]
       (let* ((src-rank (vector-length src-strides))
              (eff      (make-vector src-rank 0)))
         (let loop ((ps perm) (j 0))
           (unless (null? ps)
             (vector-set! eff (car ps) (vector-ref src-strides j))
             (loop (cdr ps) (+ j 1))))
         (cons src-offset eff)))
      (diagonal-fn (diag bias)
       ;; source-idx[j] = diag[j]*out[j] + bias[j]
       ;; eff-strides[j] = diag[j]*src-strides[j]; eff-offset adds bias terms
       (let* ((src-rank (vector-length src-strides))
              (eff      (make-vector src-rank 0))
              (eff-offset
               (let loop ((ds diag) (bs bias) (j 0) (acc src-offset))
                 (if (null? ds)
                     acc
                     (begin
                       (vector-set! eff j (* (car ds) (vector-ref src-strides j)))
                       (loop (cdr ds) (cdr bs) (+ j 1)
                             (+ acc (* (car bs) (vector-ref src-strides j)))))))))
         (cons eff-offset eff)))
      (general-fn (matrix bias)
       ;; matrix: src-rank rows x out-rank cols
       ;; eff-strides[k] = sum_j matrix[j][k]*src-strides[j]
       (let* ((src-rank  (vector-length src-strides))
              (out-rank  (if (null? matrix) 0 (length (car matrix))))
              (eff       (make-vector out-rank 0))
              (eff-offset
               (let loop ((rows matrix) (j 0) (acc src-offset))
                 (if (null? rows)
                     acc
                     (let* ((row  (car rows))
                            (sj   (vector-ref src-strides j))
                            (acc2 (if bias
                                      (+ acc (* (list-ref bias j) sj))
                                      acc)))
                       (let upd ((cols row) (k 0))
                         (unless (null? cols)
                           (vector-set! eff k (+ (vector-ref eff k) (* (car cols) sj)))
                           (upd (cdr cols) (+ k 1))))
                       (loop (cdr rows) (+ j 1) acc2))))))
         (cons eff-offset eff)))))

  (define (execute-affine-morphism fn output-buffer shape operands dtype)
    "Execute affine index function: A·i + b
    
    This handles reshape, transpose, and slice operations through
    affine coordinate transformations."
    
    (unless (= (length operands) 1)
      (error "Affine morphism requires exactly one operand" operands))
    
    (let* ((source (car operands))
           (size (shape-size shape)))
      
      (unless (concrete-array? source)
        (error "Affine morphism source must be concrete" source))
      
      (cases array-morphism source
        (concrete-array (src-data src-shape src-strides src-offset src-dtype _ _)

          (if (identity-index-fn? fn)
              ;; Reshape (identity-fn): output flat index i corresponds to
              ;; source flat index i.  Use direct nested loops over src-shape
              ;; dimensions to compute physical address without per-element
              ;; vector allocation.  Handles ranks 1-4 with specialized loops;
              ;; falls back to linear-to-multi-index for rank > 4 (rare).
              (let ((rank (vector-length src-shape)))
                (case rank
                  ((1)
                   (let* ((d0 (vector-ref src-shape 0))
                          (s0 (vector-ref src-strides 0)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (f32vector-set! output-buffer i0
                                  (f32vector-ref src-data (+ src-offset (* i0 s0))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (f64vector-set! output-buffer i0
                                  (f64vector-ref src-data (+ src-offset (* i0 s0))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((src-multi (vector->list (linear-to-multi-index i src-shape)))
                                      (physical  (multi-to-linear-index (list->vector src-multi) src-strides src-offset))
                                      (value (typed-vector-ref src-data src-dtype physical)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  ((2)
                   (let* ((d0 (vector-ref src-shape 0)) (d1 (vector-ref src-shape 1))
                          (s0 (vector-ref src-strides 0)) (s1 (vector-ref src-strides 1)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                  (f32vector-set! output-buffer (+ (* i0 d1) i1)
                                    (f32vector-ref src-data (+ src-offset (* i0 s0) (* i1 s1)))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                  (f64vector-set! output-buffer (+ (* i0 d1) i1)
                                    (f64vector-ref src-data (+ src-offset (* i0 s0) (* i1 s1)))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((src-multi (vector->list (linear-to-multi-index i src-shape)))
                                      (physical  (multi-to-linear-index (list->vector src-multi) src-strides src-offset))
                                      (value (typed-vector-ref src-data src-dtype physical)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  ((3)
                   (let* ((d0 (vector-ref src-shape 0)) (d1 (vector-ref src-shape 1)) (d2 (vector-ref src-shape 2))
                          (s0 (vector-ref src-strides 0)) (s1 (vector-ref src-strides 1)) (s2 (vector-ref src-strides 2)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                  (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                    (f32vector-set! output-buffer (+ (* (+ (* i0 d1) i1) d2) i2)
                                      (f32vector-ref src-data (+ src-offset (* i0 s0) (* i1 s1) (* i2 s2))))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                  (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                    (f64vector-set! output-buffer (+ (* (+ (* i0 d1) i1) d2) i2)
                                      (f64vector-ref src-data (+ src-offset (* i0 s0) (* i1 s1) (* i2 s2))))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((src-multi (vector->list (linear-to-multi-index i src-shape)))
                                      (physical  (multi-to-linear-index (list->vector src-multi) src-strides src-offset))
                                      (value (typed-vector-ref src-data src-dtype physical)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  ((4)
                   (let* ((d0 (vector-ref src-shape 0)) (d1 (vector-ref src-shape 1))
                          (d2 (vector-ref src-shape 2)) (d3 (vector-ref src-shape 3))
                          (s0 (vector-ref src-strides 0)) (s1 (vector-ref src-strides 1))
                          (s2 (vector-ref src-strides 2)) (s3 (vector-ref src-strides 3)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                  (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                    (do ((i3 0 (+ i3 1))) ((= i3 d3))
                                      (f32vector-set! output-buffer (+ (* (+ (* (+ (* i0 d1) i1) d2) i2) d3) i3)
                                        (f32vector-ref src-data (+ src-offset (* i0 s0) (* i1 s1) (* i2 s2) (* i3 s3)))))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                  (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                    (do ((i3 0 (+ i3 1))) ((= i3 d3))
                                      (f64vector-set! output-buffer (+ (* (+ (* (+ (* i0 d1) i1) d2) i2) d3) i3)
                                        (f64vector-ref src-data (+ src-offset (* i0 s0) (* i1 s1) (* i2 s2) (* i3 s3)))))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((src-multi (vector->list (linear-to-multi-index i src-shape)))
                                      (physical  (multi-to-linear-index (list->vector src-multi) src-strides src-offset))
                                      (value (typed-vector-ref src-data src-dtype physical)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  (else
                   ;; Fallback for rank > 4 (rare): original path
                   (do ((i 0 (+ i 1))) ((= i size))
                     (let* ((src-multi (vector->list (linear-to-multi-index i src-shape)))
                            (physical  (multi-to-linear-index (list->vector src-multi)
                                                              src-strides
                                                              src-offset))
                            (value (typed-vector-ref src-data src-dtype physical)))
                       (typed-vector-set! output-buffer dtype i value))))))
              
              ;; Non-identity affine (transpose, slice, composed): precompute
              ;; effective strides so nested loops need no per-element allocation.
              (let* ((ae         (affine-effective-strides fn src-strides src-offset))
                     (eff-offset (car ae))
                     (eff        (cdr ae))
                     (out-rank   (vector-length shape)))
                (case out-rank
                  ((1)
                   (let ((d0 (vector-ref shape 0))
                         (e0 (vector-ref eff 0)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (f32vector-set! output-buffer i0
                                  (f32vector-ref src-data (+ eff-offset (* i0 e0))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (f64vector-set! output-buffer i0
                                  (f64vector-ref src-data (+ eff-offset (* i0 e0))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
                                      (src-idx (apply-affine-index-fn fn out-idx))
                                      (value   (retrieve-value source src-idx)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  ((2)
                   (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1))
                         (e0 (vector-ref eff 0))   (e1 (vector-ref eff 1)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (let ((base0 (+ eff-offset (* i0 e0))))
                                  (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                    (f32vector-set! output-buffer (+ (* i0 d1) i1)
                                      (f32vector-ref src-data (+ base0 (* i1 e1))))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (let ((base0 (+ eff-offset (* i0 e0))))
                                  (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                    (f64vector-set! output-buffer (+ (* i0 d1) i1)
                                      (f64vector-ref src-data (+ base0 (* i1 e1))))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
                                      (src-idx (apply-affine-index-fn fn out-idx))
                                      (value   (retrieve-value source src-idx)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  ((3)
                   (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1)) (d2 (vector-ref shape 2))
                         (e0 (vector-ref eff 0))   (e1 (vector-ref eff 1))   (e2 (vector-ref eff 2)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (let ((base0 (+ eff-offset (* i0 e0))))
                                  (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                    (let ((base1 (+ base0 (* i1 e1))))
                                      (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                        (f32vector-set! output-buffer (+ (* (+ (* i0 d1) i1) d2) i2)
                                          (f32vector-ref src-data (+ base1 (* i2 e2))))))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (let ((base0 (+ eff-offset (* i0 e0))))
                                  (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                    (let ((base1 (+ base0 (* i1 e1))))
                                      (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                        (f64vector-set! output-buffer (+ (* (+ (* i0 d1) i1) d2) i2)
                                          (f64vector-ref src-data (+ base1 (* i2 e2))))))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
                                      (src-idx (apply-affine-index-fn fn out-idx))
                                      (value   (retrieve-value source src-idx)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  ((4)
                   (let ((d0 (vector-ref shape 0)) (d1 (vector-ref shape 1))
                         (d2 (vector-ref shape 2)) (d3 (vector-ref shape 3))
                         (e0 (vector-ref eff 0))   (e1 (vector-ref eff 1))
                         (e2 (vector-ref eff 2))   (e3 (vector-ref eff 3)))
                     (case dtype
                       ((f32) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (let ((base0 (+ eff-offset (* i0 e0))))
                                  (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                    (let ((base1 (+ base0 (* i1 e1))))
                                      (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                        (let ((base2 (+ base1 (* i2 e2))))
                                          (do ((i3 0 (+ i3 1))) ((= i3 d3))
                                            (f32vector-set! output-buffer
                                              (+ (* (+ (* (+ (* i0 d1) i1) d2) i2) d3) i3)
                                              (f32vector-ref src-data (+ base2 (* i3 e3))))))))))))
                       ((f64) (do ((i0 0 (+ i0 1))) ((= i0 d0))
                                (let ((base0 (+ eff-offset (* i0 e0))))
                                  (do ((i1 0 (+ i1 1))) ((= i1 d1))
                                    (let ((base1 (+ base0 (* i1 e1))))
                                      (do ((i2 0 (+ i2 1))) ((= i2 d2))
                                        (let ((base2 (+ base1 (* i2 e2))))
                                          (do ((i3 0 (+ i3 1))) ((= i3 d3))
                                            (f64vector-set! output-buffer
                                              (+ (* (+ (* (+ (* i0 d1) i1) d2) i2) d3) i3)
                                              (f64vector-ref src-data (+ base2 (* i3 e3))))))))))))
                       (else (do ((i 0 (+ i 1))) ((= i size))
                               (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
                                      (src-idx (apply-affine-index-fn fn out-idx))
                                      (value   (retrieve-value source src-idx)))
                                 (typed-vector-set! output-buffer dtype i value)))))))
                  (else
                   (do ((i 0 (+ i 1))) ((= i size))
                     (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
                            (src-idx (apply-affine-index-fn fn out-idx))
                            (value   (retrieve-value source src-idx)))
                       (typed-vector-set! output-buffer dtype i value))))))))

        (else (error "Affine source must be concrete array")))
      ))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Computational Morphism Execution (Arithmetic, Transcendental)
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (execute-compute-morphism fn output-buffer shape operands dtype)
    "Execute computational morphism with tight loop
    
    This handles element-wise arithmetic and transcendental operations
    with broadcasting support."
    
    (let* ((input-fns (compute-index-fn-input-fns fn))
           (combiner (compute-index-fn-combiner fn))
           (size (shape-size shape)))
      
      ;; Type-specialized tight loop
      (case dtype
        ((f64)
         (do ((i 0 (+ i 1)))
             ((= i size))
           (let* ((multi-idx (vector->list (linear-to-multi-index i shape)))
                  (input-values 
                   (map (lambda (idx-fn operand)
                          (retrieve-value operand (idx-fn multi-idx)))
                        input-fns operands)))
             (f64vector-set! output-buffer i 
                           (exact->inexact (apply combiner input-values))))))
        
        ((f32)
         (do ((i 0 (+ i 1)))
             ((= i size))
           (let* ((multi-idx (vector->list (linear-to-multi-index i shape)))
                  (input-values 
                   (map (lambda (idx-fn operand)
                          (retrieve-value operand (idx-fn multi-idx)))
                        input-fns operands)))
             (f32vector-set! output-buffer i 
                           (exact->inexact (apply combiner input-values))))))
        
        ((s32)
         (do ((i 0 (+ i 1)))
             ((= i size))
           (let* ((multi-idx (vector->list (linear-to-multi-index i shape)))
                  (input-values 
                   (map (lambda (idx-fn operand)
                          (retrieve-value operand (idx-fn multi-idx)))
                        input-fns operands)))
             (s32vector-set! output-buffer i 
                           (inexact->exact (truncate (apply combiner input-values)))))))
        
        ((s64)
         (do ((i 0 (+ i 1)))
             ((= i size))
           (let* ((multi-idx (vector->list (linear-to-multi-index i shape)))
                  (input-values 
                   (map (lambda (idx-fn operand)
                          (retrieve-value operand (idx-fn multi-idx)))
                        input-fns operands)))
             (s64vector-set! output-buffer i 
                           (inexact->exact (truncate (apply combiner input-values)))))))
        
        (else
         (error "Unsupported dtype for compute morphism" dtype)))))

  ;; src-dtype is the dtype of the input data vector; dtype is the output dtype.
  ;; These may differ when type promotion occurs (e.g. s32 -> f64 for sqrt).
  ;;
  ;; Fast paths below specialize both the input READ and the output WRITE
  ;; for the common no-promotion float cases (f32->f32, f64->f64), so the
  ;; loop body is a direct f32vector-ref/f32vector-set! pair with no
  ;; per-element dtype dispatch (typed-vector-ref does a fresh `case` on
  ;; every call otherwise).  Mixed/promoted/integer dtypes fall back to the
  ;; original generic path.
  (define (execute-flat-unary-compute combiner data src-dtype output-buffer size dtype)
    (cond
      ((and (eq? src-dtype 'f32) (eq? dtype 'f32))
       (do ((i 0 (+ i 1))) ((= i size))
         (f32vector-set! output-buffer i (combiner (f32vector-ref data i)))))
      ((and (eq? src-dtype 'f64) (eq? dtype 'f64))
       (do ((i 0 (+ i 1))) ((= i size))
         (f64vector-set! output-buffer i (combiner (f64vector-ref data i)))))
      (else
       (case dtype
         ((f64) (do ((i 0 (+ i 1))) ((= i size))
                  (f64vector-set! output-buffer i (combiner (typed-vector-ref data src-dtype i)))))
         ((f32) (do ((i 0 (+ i 1))) ((= i size))
                  (f32vector-set! output-buffer i (combiner (typed-vector-ref data src-dtype i)))))
         ((s32) (do ((i 0 (+ i 1))) ((= i size))
                  (s32vector-set! output-buffer i (inexact->exact (truncate (combiner (typed-vector-ref data src-dtype i)))))))
         ((s64) (do ((i 0 (+ i 1))) ((= i size))
                  (s64vector-set! output-buffer i (inexact->exact (truncate (combiner (typed-vector-ref data src-dtype i)))))))
         (else (error "execute-flat-unary-compute: unsupported dtype" dtype))))))

  ;; src-dtype1/src-dtype2 are the input dtypes; dtype is the output dtype.
  ;; See execute-flat-unary-compute above for why the f32/f32/f32 and
  ;; f64/f64/f64 fast paths exist.
  (define (execute-flat-binary-compute combiner data1 src-dtype1 data2 src-dtype2 output-buffer size dtype)
    (cond
      ((and (eq? src-dtype1 'f32) (eq? src-dtype2 'f32) (eq? dtype 'f32))
       (do ((i 0 (+ i 1))) ((= i size))
         (f32vector-set! output-buffer i (exact->inexact (combiner (f32vector-ref data1 i) (f32vector-ref data2 i))))))
      ((and (eq? src-dtype1 'f64) (eq? src-dtype2 'f64) (eq? dtype 'f64))
       (do ((i 0 (+ i 1))) ((= i size))
         (f64vector-set! output-buffer i (exact->inexact (combiner (f64vector-ref data1 i) (f64vector-ref data2 i))))))
      (else
       (case dtype
         ((f64) (do ((i 0 (+ i 1))) ((= i size))
                  (f64vector-set! output-buffer i (exact->inexact (combiner (typed-vector-ref data1 src-dtype1 i)
                                                                             (typed-vector-ref data2 src-dtype2 i))))))
         ((f32) (do ((i 0 (+ i 1))) ((= i size))
                  (f32vector-set! output-buffer i (exact->inexact (combiner (typed-vector-ref data1 src-dtype1 i)
                                                                             (typed-vector-ref data2 src-dtype2 i))))))
         ((s32) (do ((i 0 (+ i 1))) ((= i size))
                  (s32vector-set! output-buffer i (inexact->exact (truncate (combiner (typed-vector-ref data1 src-dtype1 i)
                                                                                       (typed-vector-ref data2 src-dtype2 i)))))))
         ((s64) (do ((i 0 (+ i 1))) ((= i size))
                  (s64vector-set! output-buffer i (inexact->exact (truncate (combiner (typed-vector-ref data1 src-dtype1 i)
                                                                                       (typed-vector-ref data2 src-dtype2 i)))))))
         (else (error "execute-flat-binary-compute: unsupported dtype" dtype))))))

  (define (execute-flat-bias-broadcast-compute combiner data1 data2 output-buffer size N dtype)
    ;; Use outer (row) + inner (col) loops to avoid (modulo i N) per element.
    (let ((M (quotient size N)))
      (case dtype
        ((f64) (do ((row 0 (+ row 1))) ((= row M))
                 (let ((base (* row N)))
                   (do ((col 0 (+ col 1))) ((= col N))
                     (let ((i (+ base col)))
                       (f64vector-set! output-buffer i
                         (combiner (f64vector-ref data1 i) (f64vector-ref data2 col))))))))
        ((f32) (do ((row 0 (+ row 1))) ((= row M))
                 (let ((base (* row N)))
                   (do ((col 0 (+ col 1))) ((= col N))
                     (let ((i (+ base col)))
                       (f32vector-set! output-buffer i
                         (combiner (f32vector-ref data1 i) (f32vector-ref data2 col))))))))
        ((s32) (do ((row 0 (+ row 1))) ((= row M))
                 (let ((base (* row N)))
                   (do ((col 0 (+ col 1))) ((= col N))
                     (let ((i (+ base col)))
                       (s32vector-set! output-buffer i
                         (inexact->exact (truncate (combiner (s32vector-ref data1 i) (s32vector-ref data2 col))))))))))
        ((s64) (do ((row 0 (+ row 1))) ((= row M))
                 (let ((base (* row N)))
                   (do ((col 0 (+ col 1))) ((= col N))
                     (let ((i (+ base col)))
                       (s64vector-set! output-buffer i
                         (inexact->exact (truncate (combiner (s64vector-ref data1 i) (s64vector-ref data2 col))))))))))
        (else (error "execute-flat-bias-broadcast-compute: unsupported dtype" dtype)))))

  (define (execute-flat-unary-compute-inplace! combiner buf size dtype)
    (case dtype
      ((f64) (do ((i 0 (+ i 1))) ((= i size))
               (f64vector-set! buf i (combiner (f64vector-ref buf i)))))
      ((f32) (do ((i 0 (+ i 1))) ((= i size))
               (f32vector-set! buf i (combiner (f32vector-ref buf i)))))
      (else (error "execute-flat-unary-compute-inplace!: unsupported dtype" dtype))))

  (define (execute-flat-bias-broadcast-inplace! combiner buf bias-data size N dtype)
    ;; Use outer (row) + inner (col) loops to avoid (modulo i N) per element.
    (let ((M (quotient size N)))
      (case dtype
        ((f64) (do ((row 0 (+ row 1))) ((= row M))
                 (let ((base (* row N)))
                   (do ((col 0 (+ col 1))) ((= col N))
                     (let ((i (+ base col)))
                       (f64vector-set! buf i (combiner (f64vector-ref buf i) (f64vector-ref bias-data col))))))))
        ((f32) (do ((row 0 (+ row 1))) ((= row M))
                 (let ((base (* row N)))
                   (do ((col 0 (+ col 1))) ((= col N))
                     (let ((i (+ base col)))
                       (f32vector-set! buf i (combiner (f32vector-ref buf i) (f32vector-ref bias-data col))))))))
        (else (error "execute-flat-bias-broadcast-inplace!: unsupported dtype" dtype)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Fast im2col kernels (dtype-specialized, zero per-element allocation)
  ;;; Called by SSA ri-im2col replay instruction.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (execute-im2col-unbatched out-buf src-data C H W KH KW SH SW PH PW OH OW dtype)
    "Fast im2col for [C,H,W] input -> [C*KH*KW, OH*OW] output.
    Loops over (c,kh,kw,oh,ow) directly -- no quotient/modulo in inner body."
    (let* ((col-cols (* OH OW))
           (H*W      (* H W)))
      (case dtype
        ((f32)
         (do ((c 0 (+ c 1))) ((= c C))
           (do ((kh 0 (+ kh 1))) ((= kh KH))
             (do ((kw 0 (+ kw 1))) ((= kw KW))
               (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                      (row-base (* col-row col-cols))
                      (c-base   (* c H*W)))
                 (do ((oh 0 (+ oh 1))
                      (ih (- kh PH) (+ ih SH)))
                     ((= oh OH))
                   (let* ((ih-ok?   (and (>= ih 0) (< ih H)))
                          (ih-W     (if ih-ok? (* ih W) 0))
                          (out-base (+ row-base (* oh OW))))
                     (do ((ow 0 (+ ow 1))
                          (iw (- kw PW) (+ iw SW)))
                         ((= ow OW))
                       (f32vector-set! out-buf (+ out-base ow)
                         (if (and ih-ok? (>= iw 0) (< iw W))
                             (f32vector-ref src-data (+ c-base ih-W iw))
                             0.0))))))))))
        ((f64)
         (do ((c 0 (+ c 1))) ((= c C))
           (do ((kh 0 (+ kh 1))) ((= kh KH))
             (do ((kw 0 (+ kw 1))) ((= kw KW))
               (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                      (row-base (* col-row col-cols))
                      (c-base   (* c H*W)))
                 (do ((oh 0 (+ oh 1))
                      (ih (- kh PH) (+ ih SH)))
                     ((= oh OH))
                   (let* ((ih-ok?   (and (>= ih 0) (< ih H)))
                          (ih-W     (if ih-ok? (* ih W) 0))
                          (out-base (+ row-base (* oh OW))))
                     (do ((ow 0 (+ ow 1))
                          (iw (- kw PW) (+ iw SW)))
                         ((= ow OW))
                       (f64vector-set! out-buf (+ out-base ow)
                         (if (and ih-ok? (>= iw 0) (< iw W))
                             (f64vector-ref src-data (+ c-base ih-W iw))
                             0.0))))))))))
        (else (error "execute-im2col-unbatched: unsupported dtype" dtype)))))

  (define (execute-im2col-batched out-buf src-data N C H W KH KW SH SW PH PW OH OW dtype)
    "Fast batched im2col for [N,C,H,W] input -> [N, C*KH*KW, OH*OW] output.
    Loops over (n,c,kh,kw,oh,ow) directly -- no quotient/modulo in inner body."
    (let* ((col-rows (* C KH KW))
           (col-cols (* OH OW))
           (C*H*W    (* C H W))
           (H*W      (* H W)))
      (case dtype
        ((f32)
         (do ((n 0 (+ n 1))) ((= n N))
           (let ((n-src (* n C*H*W))
                 (n-out (* n col-rows col-cols)))
             (do ((c 0 (+ c 1))) ((= c C))
               (do ((kh 0 (+ kh 1))) ((= kh KH))
                 (do ((kw 0 (+ kw 1))) ((= kw KW))
                   (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                          (row-base (+ n-out (* col-row col-cols)))
                          (c-base   (+ n-src (* c H*W))))
                     (do ((oh 0 (+ oh 1))
                          (ih (- kh PH) (+ ih SH)))
                         ((= oh OH))
                       (let* ((ih-ok?   (and (>= ih 0) (< ih H)))
                              (ih-W     (if ih-ok? (* ih W) 0))
                              (out-base (+ row-base (* oh OW))))
                         (do ((ow 0 (+ ow 1))
                              (iw (- kw PW) (+ iw SW)))
                             ((= ow OW))
                           (f32vector-set! out-buf (+ out-base ow)
                             (if (and ih-ok? (>= iw 0) (< iw W))
                                 (f32vector-ref src-data (+ c-base ih-W iw))
                                 0.0))))))))))))
        ((f64)
         (do ((n 0 (+ n 1))) ((= n N))
           (let ((n-src (* n C*H*W))
                 (n-out (* n col-rows col-cols)))
             (do ((c 0 (+ c 1))) ((= c C))
               (do ((kh 0 (+ kh 1))) ((= kh KH))
                 (do ((kw 0 (+ kw 1))) ((= kw KW))
                   (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                          (row-base (+ n-out (* col-row col-cols)))
                          (c-base   (+ n-src (* c H*W))))
                     (do ((oh 0 (+ oh 1))
                          (ih (- kh PH) (+ ih SH)))
                         ((= oh OH))
                       (let* ((ih-ok?   (and (>= ih 0) (< ih H)))
                              (ih-W     (if ih-ok? (* ih W) 0))
                              (out-base (+ row-base (* oh OW))))
                         (do ((ow 0 (+ ow 1))
                              (iw (- kw PW) (+ iw SW)))
                             ((= ow OW))
                           (f64vector-set! out-buf (+ out-base ow)
                             (if (and ih-ok? (>= iw 0) (< iw W))
                                 (f64vector-ref src-data (+ c-base ih-W iw))
                                 0.0))))))))))))
        (else (error "execute-im2col-batched: unsupported dtype" dtype)))))

  (define (execute-im2col-batched-mr out-buf src-data N C H W KH KW SH SW PH PW OH OW dtype)
    "Matmul-ready batched im2col: [N,C,H,W] -> [N*OH_OW, C*KH*KW].
    Loop order n,oh,ow,c,kh,kw -- output writes are sequential per row."
    (let* ((fan-in (* C KH KW))
           (KH*KW  (* KH KW))
           (H*W    (* H W))
           (C*H*W  (* C H W)))
      (case dtype
        ((f32)
         (for-range (n N)
           (let ((n-src (* n C*H*W)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((row-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base row-base KH*KW))
                     (let ((c-base (+ n-src (* c H*W))))
                       (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                         (let* ((ih-ok? (and (>= ih 0) (< ih H)))
                                (ih-W   (if ih-ok? (* ih W) 0)))
                           (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                             (f32vector-set! out-buf col-i
                               (if (and ih-ok? (>= iw 0) (< iw W))
                                   (f32vector-ref src-data (+ c-base ih-W iw))
                                   0.0)))))))))))))
        ((f64)
         (for-range (n N)
           (let ((n-src (* n C*H*W)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((row-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base row-base KH*KW))
                     (let ((c-base (+ n-src (* c H*W))))
                       (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                         (let* ((ih-ok? (and (>= ih 0) (< ih H)))
                                (ih-W   (if ih-ok? (* ih W) 0)))
                           (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                             (f64vector-set! out-buf col-i
                               (if (and ih-ok? (>= iw 0) (< iw W))
                                   (f64vector-ref src-data (+ c-base ih-W iw))
                                   0.0)))))))))))))
        (else (error "execute-im2col-batched-mr: unsupported dtype" dtype)))))

  (define (execute-im2col-nhwc-mr out-buf src-data N C H W KH KW SH SW PH PW OH OW dtype)
    "Matmul-ready NHWC im2col: [N,H,W,C] -> [N*OH_OW, C*KH*KW].
    Reads are sequential in C for fixed (n,oh,ow,kh,kw) -- better cache behavior."
    (let* ((fan-in (* C KH KW))
           (KH*KW  (* KH KW))
           (H*W*C  (* H W C)))
      (case dtype
        ((f32)
         (for-range (n N)
           (let ((n-src (* n H*W*C)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((row-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base row-base KH*KW))
                     (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                       (let ((ih-ok? (and (>= ih 0) (< ih H))))
                         (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                           (f32vector-set! out-buf col-i
                             (if (and ih-ok? (>= iw 0) (< iw W))
                                 (f32vector-ref src-data (+ n-src (* (+ (* ih W) iw) C) c))
                                 0.0))))))))))))
        ((f64)
         (for-range (n N)
           (let ((n-src (* n H*W*C)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((row-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base row-base KH*KW))
                     (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                       (let ((ih-ok? (and (>= ih 0) (< ih H))))
                         (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                           (f64vector-set! out-buf col-i
                             (if (and ih-ok? (>= iw 0) (< iw W))
                                 (f64vector-ref src-data (+ n-src (* (+ (* ih W) iw) C) c))
                                 0.0))))))))))))
        (else (error "execute-im2col-nhwc-mr: unsupported dtype" dtype)))))

  (define (execute-im2col-nhwc-classic out-buf src-data N C H W KH KW SH SW PH PW OH OW dtype)
    "Classic NHWC im2col: [N,H,W,C] -> [N, C*KH*KW, OH*OW].
    Loop (n,c,kh,kw,oh,ow): writes stride-1 (sequential), reads stride-C in ow."
    (let* ((col-rows (* C KH KW))
           (col-cols (* OH OW))
           (H*W      (* H W))
           (H*W*C    (* H W C)))
      (case dtype
        ((f32)
         (do ((n 0 (+ n 1))) ((= n N))
           (let ((n-src (* n H*W*C))
                 (n-out (* n col-rows col-cols)))
             (do ((c 0 (+ c 1))) ((= c C))
               (do ((kh 0 (+ kh 1))) ((= kh KH))
                 (do ((kw 0 (+ kw 1))) ((= kw KW))
                   (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                          (row-base (+ n-out (* col-row col-cols))))
                     (do ((oh 0 (+ oh 1))
                          (ih (- kh PH) (+ ih SH)))
                         ((= oh OH))
                       (let* ((ih-ok?   (and (>= ih 0) (< ih H)))
                              (ih-W     (if ih-ok? (* ih W) 0))
                              (out-base (+ row-base (* oh OW))))
                         (do ((ow 0 (+ ow 1))
                              (iw (- kw PW) (+ iw SW)))
                             ((= ow OW))
                           (f32vector-set! out-buf (+ out-base ow)
                             (if (and ih-ok? (>= iw 0) (< iw W))
                                 (f32vector-ref src-data (+ n-src (* (+ ih-W iw) C) c))
                                 0.0))))))))))))
        ((f64)
         (do ((n 0 (+ n 1))) ((= n N))
           (let ((n-src (* n H*W*C))
                 (n-out (* n col-rows col-cols)))
             (do ((c 0 (+ c 1))) ((= c C))
               (do ((kh 0 (+ kh 1))) ((= kh KH))
                 (do ((kw 0 (+ kw 1))) ((= kw KW))
                   (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                          (row-base (+ n-out (* col-row col-cols))))
                     (do ((oh 0 (+ oh 1))
                          (ih (- kh PH) (+ ih SH)))
                         ((= oh OH))
                       (let* ((ih-ok?   (and (>= ih 0) (< ih H)))
                              (ih-W     (if ih-ok? (* ih W) 0))
                              (out-base (+ row-base (* oh OW))))
                         (do ((ow 0 (+ ow 1))
                              (iw (- kw PW) (+ iw SW)))
                             ((= ow OW))
                           (f64vector-set! out-buf (+ out-base ow)
                             (if (and ih-ok? (>= iw 0) (< iw W))
                                 (f64vector-ref src-data (+ n-src (* (+ ih-W iw) C) c))
                                 0.0))))))))))))
        (else (error "execute-im2col-nhwc-classic: unsupported dtype" dtype)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Window Morphism Execution (im2col, Padding)
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (execute-window-morphism fn output-buffer shape operands dtype)
    "Execute window index function (im2col, padding)
    
    Handles padding modes: constant, edge, reflect"
    
    (unless (= (length operands) 1)
      (error "Window morphism requires exactly one operand" operands))
    
    (let* ((source (car operands))
           (size (shape-size shape)))
      
      (unless (concrete-array? source)
        (error "Window morphism source must be concrete" source))
      
      ;; Iterate over all output indices
      (do ((i 0 (+ i 1)))
          ((= i size))
        
        (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
               
               ;; Apply window index function
               (src-result (fn out-idx))
               
               ;; Handle different return types
               (value
                (cond
                  ;; Padding marker (constant mode)
                  ((and (pair? src-result) (eq? (car src-result) 'constant))
                   (cadr src-result))
                  
                  ;; Special padding marker
                  ((eq? src-result 'pad-zero)
                   0.0)
                  
                  ;; Regular index - retrieve value
                  ((list? src-result)
                   (retrieve-value source src-result))
                  
                  (else
                   (error "Invalid window index function result" src-result)))))
          
          (typed-vector-set! output-buffer dtype i value)))))

  (define (execute-routing-morphism fn output-buffer shape operands dtype)
    "Execute a routing index function (stack or concat).

  For each output position, calls (apply-index-fn fn out-idx) to
  obtain (source-id . source-idx), then reads from
  operands[source-id] at source-idx.

  This single kernel replaces both execute-stack-morphism and
  execute-concat-morphism."

    (when (null? operands)
      (error "Routing morphism requires at least one operand"))
    (unless (every concrete-array? operands)
      (error "All routing operands must be concrete" operands))

    (let* ((size   (shape-size shape))
           (n-ops  (length operands))
           (op-vec (list->vector operands)))  ; O(1) lookup
      
      (do ((i 0 (+ i 1)))
          ((= i size))

        (let* ((out-idx   (vector->list (linear-to-multi-index i shape)))
               (result    (apply-index-fn fn out-idx))
               (source-id (car result))
               (src-idx   (cdr result)))

          (unless (and (exact-integer? source-id)
                       (>= source-id 0)
                       (< source-id n-ops))
            (error "Routing index function returned invalid source-id"
                   source-id n-ops fn))
          
          (let* ((source (vector-ref op-vec source-id))
                 (value  (retrieve-value source src-idx)))
            (typed-vector-set! output-buffer dtype i value))))))

  
  (define (execute-col2im-morphism fn output-buffer shape operands dtype)
    "Execute col2im with accumulation for overlapping windows
  
  Algorithm:
  1. Initialize output to zeros
  2. For each column position (col_row, col_col):
     - Decompose into (c, kh, kw, oh, ow)
     - Compute input position: (c, oh*SH + kh - PH, ow*SW + kw - PW)
     - If in bounds, accumulate: output[c,h,w] += col[col_row, col_col]
  
  This is the adjoint (transpose) of im2col."
  
    (unless (= (length operands) 1)
      (error "col2im requires exactly one operand" operands))
  
    (let* ((col-morphism (car operands))
           (KH (col2im-index-fn-kernel-h fn))
           (KW (col2im-index-fn-kernel-w fn))
           (SH (col2im-index-fn-stride-h fn))
           (SW (col2im-index-fn-stride-w fn))
           (PH (col2im-index-fn-pad-h fn))
           (PW (col2im-index-fn-pad-w fn))
           (col-layout (col2im-index-fn-col-layout fn)))

      (unless (concrete-array? col-morphism)
        (error "col2im source must be concrete" col-morphism))

      (cases array-morphism col-morphism
             (concrete-array
              (col-data col-shape col-strides col-offset col-dtype _ _)
              (case col-layout
                ((nchw-standard)
                 (execute-col2im-unbatched output-buffer shape col-data col-shape
                                           KH KW SH SW PH PW dtype))
                ((nchw-classic)
                 (execute-col2im-batched output-buffer shape col-data col-shape
                                         KH KW SH SW PH PW dtype))
                ((nhwc-classic)
                 (execute-col2im-nhwc-classic output-buffer shape col-data col-shape
                                              KH KW SH SW PH PW dtype))
                (else (error "col2im: unknown col-layout" col-layout))))

             (else (error "col2im operand must be concrete array")))))

(define (execute-col2im-unbatched output-buffer output-shape
                                  col-data col-shape
                                  KH KW SH SW PH PW dtype)
  "Fast col2im for non-batched input (dtype-specialized, zero per-element allocation).
  Input col: (C*KH*KW, OH*OW)  Output: (C, H, W)
  Loops over (c,kh,kw,oh,ow) directly -- no quotient/modulo in inner body."
  (let* ((C         (vector-ref output-shape 0))
         (H         (vector-ref output-shape 1))
         (W         (vector-ref output-shape 2))
         (OH        (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
         (OW        (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))
         (col-cols  (* OH OW))
         (H*W       (* H W))
         (output-size (shape-size output-shape)))
    (case dtype
      ((f32)
       (do ((i 0 (+ i 1))) ((= i output-size)) (f32vector-set! output-buffer i 0.0))
       (do ((c 0 (+ c 1))) ((= c C))
         (do ((kh 0 (+ kh 1))) ((= kh KH))
           (do ((kw 0 (+ kw 1))) ((= kw KW))
             (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                    (cr-off   (* col-row col-cols))
                    (c-base   (* c H*W)))
               (do ((oh 0 (+ oh 1))
                    (ih (- kh PH) (+ ih SH)))
                   ((= oh OH))
                 (when (and (>= ih 0) (< ih H))
                   (let* ((ih-W     (* ih W))
                          (col-base (+ cr-off (* oh OW))))
                     (do ((ow 0 (+ ow 1))
                          (iw (- kw PW) (+ iw SW)))
                         ((= ow OW))
                       (when (and (>= iw 0) (< iw W))
                         (let ((out-i (+ c-base ih-W iw))
                               (col-i (+ col-base ow)))
                           (f32vector-set! output-buffer out-i
                             (+ (f32vector-ref output-buffer out-i)
                                (f32vector-ref col-data col-i))))))))))))))
      ((f64)
       (do ((i 0 (+ i 1))) ((= i output-size)) (f64vector-set! output-buffer i 0.0))
       (do ((c 0 (+ c 1))) ((= c C))
         (do ((kh 0 (+ kh 1))) ((= kh KH))
           (do ((kw 0 (+ kw 1))) ((= kw KW))
             (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                    (cr-off   (* col-row col-cols))
                    (c-base   (* c H*W)))
               (do ((oh 0 (+ oh 1))
                    (ih (- kh PH) (+ ih SH)))
                   ((= oh OH))
                 (when (and (>= ih 0) (< ih H))
                   (let* ((ih-W     (* ih W))
                          (col-base (+ cr-off (* oh OW))))
                     (do ((ow 0 (+ ow 1))
                          (iw (- kw PW) (+ iw SW)))
                         ((= ow OW))
                       (when (and (>= iw 0) (< iw W))
                         (let ((out-i (+ c-base ih-W iw))
                               (col-i (+ col-base ow)))
                           (f64vector-set! output-buffer out-i
                             (+ (f64vector-ref output-buffer out-i)
                                (f64vector-ref col-data col-i))))))))))))))
      (else (error "execute-col2im-unbatched: unsupported dtype" dtype)))))
    

  (define (execute-col2im-batched output-buffer output-shape
                                  col-data col-shape
                                  KH KW SH SW PH PW dtype)
    "Fast col2im for batched input (dtype-specialized, zero per-element allocation).
    Input col: (N, C*KH*KW, OH*OW)  Output: (N, C, H, W)
    Loops over (n,c,kh,kw,oh,ow) directly -- no quotient/modulo in inner body."
    (let* ((N         (vector-ref output-shape 0))
           (C         (vector-ref output-shape 1))
           (H         (vector-ref output-shape 2))
           (W         (vector-ref output-shape 3))
           (OH        (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
           (OW        (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))
           (col-rows  (* C KH KW))
           (col-cols  (* OH OW))
           (H*W       (* H W))
           (C*H*W     (* C H W))
           (output-size (shape-size output-shape)))
      (case dtype
        ((f32)
         (begin
           (do ((i 0 (+ i 1))) ((= i output-size)) (f32vector-set! output-buffer i 0.0))
           (do ((n 0 (+ n 1))) ((= n N))
             (let ((n-out (* n C*H*W))
                   (n-col (* n col-rows col-cols)))
               (do ((c 0 (+ c 1))) ((= c C))
                 (do ((kh 0 (+ kh 1))) ((= kh KH))
                   (do ((kw 0 (+ kw 1))) ((= kw KW))
                     (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                            (cr-off   (+ n-col (* col-row col-cols)))
                            (c-base   (+ n-out (* c H*W))))
                       (do ((oh 0 (+ oh 1))
                            (ih (- kh PH) (+ ih SH)))
                           ((= oh OH))
                         (when (and (>= ih 0) (< ih H))
                           (let* ((ih-W     (* ih W))
                                (col-base (+ cr-off (* oh OW))))
                             (do ((ow 0 (+ ow 1))
                                  (iw (- kw PW) (+ iw SW)))
                                 ((= ow OW))
                               (when (and (>= iw 0) (< iw W))
                                 (let ((out-i (+ c-base ih-W iw))
                                       (col-i (+ col-base ow)))
                                   (f32vector-set! output-buffer out-i
                                                   (+ (f32vector-ref output-buffer out-i)
                                                      (f32vector-ref col-data col-i))))))))))))))
             )))
        ((f64)
         (begin
           (do ((i 0 (+ i 1))) ((= i output-size)) (f64vector-set! output-buffer i 0.0))
           (do ((n 0 (+ n 1))) ((= n N))
             (let ((n-out (* n C*H*W))
                   (n-col (* n col-rows col-cols)))
               (do ((c 0 (+ c 1))) ((= c C))
                 (do ((kh 0 (+ kh 1))) ((= kh KH))
                   (do ((kw 0 (+ kw 1))) ((= kw KW))
                     (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                            (cr-off   (+ n-col (* col-row col-cols)))
                            (c-base   (+ n-out (* c H*W))))
                       (do ((oh 0 (+ oh 1))
                            (ih (- kh PH) (+ ih SH)))
                           ((= oh OH))
                         (when (and (>= ih 0) (< ih H))
                           (let* ((ih-W     (* ih W))
                                  (col-base (+ cr-off (* oh OW))))
                             (do ((ow 0 (+ ow 1))
                                  (iw (- kw PW) (+ iw SW)))
                                 ((= ow OW))
                               (when (and (>= iw 0) (< iw W))
                                 (let ((out-i (+ c-base ih-W iw))
                                       (col-i (+ col-base ow)))
                                   (f64vector-set! output-buffer out-i
                                                   (+ (f64vector-ref output-buffer out-i)
                                                      (f64vector-ref col-data col-i))))))))))))))
             ))
         )
        (else (error "execute-col2im-batched: unsupported dtype" dtype)))))

  (define (execute-col2im-nhwc-classic output-buffer output-shape
                                       col-data col-shape
                                       KH KW SH SW PH PW dtype)
    "Classic col2im: [N, C*KH*KW, OH*OW] -> NHWC [N,H,W,C].
    Reads stride-1 from col (sequential), scatter-add stride-C into NHWC output."
    (let* ((N         (vector-ref output-shape 0))
           (H         (vector-ref output-shape 1))
           (W         (vector-ref output-shape 2))
           (C         (vector-ref output-shape 3))
           (OH        (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
           (OW        (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))
           (col-rows  (* C KH KW))
           (col-cols  (* OH OW))
           (H*W       (* H W))
           (H*W*C     (* H W C))
           (output-size (shape-size output-shape)))
      (case dtype
        ((f32)
         (begin
           (do ((i 0 (+ i 1))) ((= i output-size)) (f32vector-set! output-buffer i 0.0))
           (do ((n 0 (+ n 1))) ((= n N))
             (let ((n-out (* n H*W*C))
                   (n-col (* n col-rows col-cols)))
               (do ((c 0 (+ c 1))) ((= c C))
                 (do ((kh 0 (+ kh 1))) ((= kh KH))
                   (do ((kw 0 (+ kw 1))) ((= kw KW))
                     (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                            (cr-off   (+ n-col (* col-row col-cols))))
                       (do ((oh 0 (+ oh 1))
                            (ih (- kh PH) (+ ih SH)))
                           ((= oh OH))
                         (when (and (>= ih 0) (< ih H))
                           (let ((col-base (+ cr-off (* oh OW))))
                             (do ((ow 0 (+ ow 1))
                                  (iw (- kw PW) (+ iw SW)))
                                 ((= ow OW))
                               (when (and (>= iw 0) (< iw W))
                                 (let ((out-i (+ n-out (* (+ (* ih W) iw) C) c))
                                       (col-i (+ col-base ow)))
                                   (f32vector-set! output-buffer out-i
                                                   (+ (f32vector-ref output-buffer out-i)
                                                      (f32vector-ref col-data col-i)))))))))))))))
           ))
        ((f64)
         (begin
           (do ((i 0 (+ i 1))) ((= i output-size)) (f64vector-set! output-buffer i 0.0))
           (do ((n 0 (+ n 1))) ((= n N))
             (let ((n-out (* n H*W*C))
                   (n-col (* n col-rows col-cols)))
               (do ((c 0 (+ c 1))) ((= c C))
                 (do ((kh 0 (+ kh 1))) ((= kh KH))
                   (do ((kw 0 (+ kw 1))) ((= kw KW))
                     (let* ((col-row  (+ (* c KH KW) (* kh KW) kw))
                            (cr-off   (+ n-col (* col-row col-cols))))
                       (do ((oh 0 (+ oh 1))
                            (ih (- kh PH) (+ ih SH)))
                           ((= oh OH))
                         (when (and (>= ih 0) (< ih H))
                           (let ((col-base (+ cr-off (* oh OW))))
                             (do ((ow 0 (+ ow 1))
                                  (iw (- kw PW) (+ iw SW)))
                                 ((= ow OW))
                               (when (and (>= iw 0) (< iw W))
                                 (let ((out-i (+ n-out (* (+ (* ih W) iw) C) c))
                                       (col-i (+ col-base ow)))
                                   (f64vector-set! output-buffer out-i
                                                   (+ (f64vector-ref output-buffer out-i)
                                                      (f64vector-ref col-data col-i)))))))))))))))
           ))
        (else (error "execute-col2im-nhwc-classic: unsupported dtype" dtype)))))

  (define (execute-col2im-batched-mr output-buffer output-shape
                                     col-data col-shape
                                     KH KW SH SW PH PW dtype)
    "Scatter col2im for matmul-ready col [N*OH_OW, fan_in] -> NCHW [N,C,H,W].
    Reads col sequentially per patch (cache-friendly for large fan_in); scattered dx writes."
    (let* ((N          (vector-ref output-shape 0))
           (C          (vector-ref output-shape 1))
           (H          (vector-ref output-shape 2))
           (W          (vector-ref output-shape 3))
           (OH         (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
           (OW         (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))
           (fan-in     (* C KH KW))
           (KH*KW      (* KH KW))
           (H*W        (* H W))
           (C*H*W      (* C H W))
           (output-size (shape-size output-shape)))
      (case dtype
        ((f32)
         (begin
           (for-range (i output-size) (f32vector-set! output-buffer i 0.0))
           (for-range (n N)
           (let ((n-out (* n C*H*W)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((col-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base col-base KH*KW))
                     (let ((c-base (+ n-out (* c H*W))))
                       (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                         (when-valid ih H
                           (let ((ih-W (* ih W)))
                             (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                               (when-valid iw W
                                 (let ((out-i (+ c-base ih-W iw)))
                                   (f32vector-set! output-buffer out-i
                                     (+ (f32vector-ref output-buffer out-i)
                                        (f32vector-ref col-data col-i))))))))))))))))))
        ((f64)
         (begin
         (for-range (i output-size) (f64vector-set! output-buffer i 0.0))
         (for-range (n N)
           (let ((n-out (* n C*H*W)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((col-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base col-base KH*KW))
                     (let ((c-base (+ n-out (* c H*W))))
                       (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                         (when-valid ih H
                           (let ((ih-W (* ih W)))
                             (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                               (when-valid iw W
                                 (let ((out-i (+ c-base ih-W iw)))
                                   (f64vector-set! output-buffer out-i
                                     (+ (f64vector-ref output-buffer out-i)
                                        (f64vector-ref col-data col-i))))))))))))))))))
        (else (error "execute-col2im-batched-mr: unsupported dtype" dtype)))))

  (define (execute-col2im-nhwc-mr output-buffer output-shape
                                  col-data col-shape
                                  KH KW SH SW PH PW dtype)
    "Scatter col2im for [N*OH_OW, fan_in] col -> NHWC [N,H,W,C].
    Reads col sequentially per patch (cache-friendly for large fan_in); scattered dx writes."
    (let* ((N          (vector-ref output-shape 0))
           (H          (vector-ref output-shape 1))
           (W          (vector-ref output-shape 2))
           (C          (vector-ref output-shape 3))
           (OH         (+ 1 (quotient (+ H (* 2 PH) (- KH)) SH)))
           (OW         (+ 1 (quotient (+ W (* 2 PW) (- KW)) SW)))
           (fan-in     (* C KH KW))
           (KH*KW      (* KH KW))
           (H*W*C      (* H W C))
           (output-size (shape-size output-shape)))
      (case dtype
        ((f32)
         (begin
         (for-range (i output-size) (f32vector-set! output-buffer i 0.0))
         (for-range (n N)
           (let ((n-out (* n H*W*C)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((col-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base col-base KH*KW))
                     (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                       (when-valid ih H
                         (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                           (when-valid iw W
                             (let ((out-i (+ n-out (* (+ (* ih W) iw) C) c)))
                               (f32vector-set! output-buffer out-i
                                 (+ (f32vector-ref output-buffer out-i)
                                    (f32vector-ref col-data col-i))))))))))))))))
        ((f64)
         (begin
         (for-range (i output-size) (f64vector-set! output-buffer i 0.0))
         (for-range (n N)
           (let ((n-out (* n H*W*C)) (n-row (* n OH OW)))
             (for-stride (oh OH) ((ih0 (- 0 PH) SH))
               (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                 (let ((col-base (* (+ n-row (* oh OW) ow) fan-in)))
                   (for-stride (c C) ((c-col-base col-base KH*KW))
                     (for-stride (kh KH) ((ih ih0 1) (kh-col-base c-col-base KW))
                       (when-valid ih H
                         (for-stride (kw KW) ((iw iw0 1) (col-i kh-col-base 1))
                           (when-valid iw W
                             (let ((out-i (+ n-out (* (+ (* ih W) iw) C) c)))
                               (f64vector-set! output-buffer out-i
                                 (+ (f64vector-ref output-buffer out-i)
                                    (f64vector-ref col-data col-i))))))))))))))))
        (else (error "execute-col2im-nhwc-mr: unsupported dtype" dtype)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Implicit GEMM convolution: no intermediate col buffer
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (execute-conv-fwd-nchw out-buf src wt b N C H W KH KW SH SW PH PW OH OW out-ch dtype)
    "Fused im2col+GEMM+bias for NCHW input. wt is [fan_in, out_ch] (W transposed).
    Output [N*OH_OW, out_ch]. Outer (n,oh,ow)/inner (c,kh,kw)/innermost (co).
    src_val is hoisted above co loop; wt reads and out writes are stride-1.
    When dtype is f32, BLAS is enabled, an active backend is registered, and
    the backend provides a conv-fwd-im2col-f32 hot-kernel, that C path is
    used instead of the scalar loops."
    (let* ((fan-in (* C KH KW))
           (C*H*W  (* C H W))
           (H*W    (* H W))
           (OH*OW  (* OH OW)))
            (cond
        ((and (eq? dtype 'f32) (blas-enabled?) (active-blas-backend)
              (blas-backend-conv-fwd-im2col-f32 (active-blas-backend)))
         (let* ((M (* N OH*OW))
                (K fan-in))
           (let ((col (allocate-typed-vector 'f32 (* M K))))
             ((blas-backend-conv-fwd-im2col-f32 (active-blas-backend))
              out-buf b col src wt M out-ch K out-ch N C H W KH KW SH SW PH PW OH OW))))
        (else
(case dtype
        ((f32)
         (begin
           ;; Initialize output rows to bias
           (for-range (m (* N OH*OW))
             (let ((out-base (* m out-ch)))
               (for-range (co out-ch)
                 (f32vector-set! out-buf (+ out-base co) (f32vector-ref b co)))))
           ;; Accumulate: outer (n,oh,ow) / inner (c,kh,kw) / innermost (co)
           (for-range (n N)
             (let ((n-src (* n C*H*W))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((row-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (let ((c-base (+ n-src (* c H*W))))
                         (for-stride (kh KH) ((ih ih0 1))
                           (when-valid ih H
                             (for-stride (kw KW) ((iw iw0 1))
                                (when-valid iw W
                                  (let* ((src-val (f32vector-ref src (+ c-base (* ih W) iw)))
                                         (col-idx (+ (* c KH KW) (* kh KW) kw))
                                         (wt-base (* col-idx out-ch)))
                                    (for-range (co out-ch)
                                      (f32vector-set! out-buf (+ row-base co)
                                        (+ (f32vector-ref out-buf (+ row-base co))
                                           (* src-val (f32vector-ref wt (+ wt-base co)))))))))))))))))))
            )
        ((f64)
         (begin
           (for-range (m (* N OH*OW))
             (let ((out-base (* m out-ch)))
               (for-range (co out-ch)
                 (f64vector-set! out-buf (+ out-base co) (f64vector-ref b co)))))
           (for-range (n N)
             (let ((n-src (* n C*H*W))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((row-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (let ((c-base (+ n-src (* c H*W))))
                         (for-stride (kh KH) ((ih ih0 1))
                           (when-valid ih H
                             (for-stride (kw KW) ((iw iw0 1))
                               (when-valid iw W
                                  (let* ((src-val (f64vector-ref src (+ c-base (* ih W) iw)))
                                         (col-idx (+ (* c KH KW) (* kh KW) kw))
                                         (wt-base (* col-idx out-ch)))
                                    (for-range (co out-ch)
                                      (f64vector-set! out-buf (+ row-base co)
                                        (+ (f64vector-ref out-buf (+ row-base co))
                                           (* src-val (f64vector-ref wt (+ wt-base co)))))))))))))))))))
            )
         (else (error "execute-conv-fwd-nchw: unsupported dtype" dtype)))))))  (define (execute-conv-fwd-nhwc out-buf src wt b N C H W KH KW SH SW PH PW OH OW out-ch dtype)
    "Fused im2col+GEMM+bias for NHWC [N,H,W,C] input. wt is [fan_in, out_ch]."
    (let* ((fan-in (* C KH KW))
           (H*W    (* H W))
           (H*W*C  (* H W C))
           (OH*OW  (* OH OW)))
      (case dtype
        ((f32)
         (begin
           (for-range (m (* N OH*OW))
             (let ((out-base (* m out-ch)))
               (for-range (co out-ch)
                 (f32vector-set! out-buf (+ out-base co) (f32vector-ref b co)))))
           (for-range (n N)
             (let ((n-src (* n H*W*C))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((row-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (for-stride (kh KH) ((ih ih0 1))
                         (when-valid ih H
                           (for-stride (kw KW) ((iw iw0 1))
                             (when-valid iw W
                               (let* ((src-val (f32vector-ref src (+ n-src (* (+ (* ih W) iw) C) c)))
                                      (col-idx (+ (* c KH KW) (* kh KW) kw))
                                      (wt-base (* col-idx out-ch)))
                                 (for-range (co out-ch)
                                   (f32vector-set! out-buf (+ row-base co)
                                     (+ (f32vector-ref out-buf (+ row-base co))
                                        (* src-val (f32vector-ref wt (+ wt-base co))))))))))))))))))
           )
        ((f64)
         (begin
           (for-range (m (* N OH*OW))
             (let ((out-base (* m out-ch)))
               (for-range (co out-ch)
                 (f64vector-set! out-buf (+ out-base co) (f64vector-ref b co)))))
           (for-range (n N)
             (let ((n-src (* n H*W*C))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((row-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (for-stride (kh KH) ((ih ih0 1))
                         (when-valid ih H
                           (for-stride (kw KW) ((iw iw0 1))
                             (when-valid iw W
                               (let* ((src-val (f64vector-ref src (+ n-src (* (+ (* ih W) iw) C) c)))
                                      (col-idx (+ (* c KH KW) (* kh KW) kw))
                                      (wt-base (* col-idx out-ch)))
                                 (for-range (co out-ch)
                                   (f64vector-set! out-buf (+ row-base co)
                                     (+ (f64vector-ref out-buf (+ row-base co))
                                        (* src-val (f64vector-ref wt (+ wt-base co)))))))))))))))))
           ))
        (else (error "execute-conv-fwd-nhwc: unsupported dtype" dtype)))))

  (define (execute-conv-bwd-data-nchw dx-buf x-shape g-data g-shape wt-data
                                        N C H W KH KW SH SW PH PW OH OW out-ch dtype)
    "Backward w.r.t. X for NCHW: dX[n,c,ih,iw] += g[m,co]*WT[col_idx,co].
    g reads and wt reads are stride-1 per co."
    (let* ((C*H*W  (* C H W))
           (H*W    (* H W))
           (OH*OW  (* OH OW))
           (dx-size (shape-size x-shape)))
      (case dtype
        ((f32)
         (begin
           (for-range (i dx-size) (f32vector-set! dx-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n C*H*W))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (let ((c-base (+ n-src (* c H*W))))
                         (for-stride (kh KH) ((ih ih0 1))
                           (when-valid ih H
                             (for-stride (kw KW) ((iw iw0 1))
                               (when-valid iw W
                                 (let* ((dx-i (+ c-base (* ih W) iw))
                                        (col-idx (+ (* c KH KW) (* kh KW) kw))
                                        (wt-base (* col-idx out-ch)))
                                   (for-range (co out-ch)
                                     (f32vector-set! dx-buf dx-i
                                       (+ (f32vector-ref dx-buf dx-i)
                                          (* (f32vector-ref g-data (+ g-base co))
                                             (f32vector-ref wt-data (+ wt-base co))))))))))))))))))
           ))
        ((f64)
         (begin
           (for-range (i dx-size) (f64vector-set! dx-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n C*H*W))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (let ((c-base (+ n-src (* c H*W))))
                         (for-stride (kh KH) ((ih ih0 1))
                           (when-valid ih H
                             (for-stride (kw KW) ((iw iw0 1))
                               (when-valid iw W
                                 (let* ((dx-i (+ c-base (* ih W) iw))
                                        (col-idx (+ (* c KH KW) (* kh KW) kw))
                                        (wt-base (* col-idx out-ch)))
                                   (for-range (co out-ch)
                                     (f64vector-set! dx-buf dx-i
                                       (+ (f64vector-ref dx-buf dx-i)
                                          (* (f64vector-ref g-data (+ g-base co))
                                             (f64vector-ref wt-data (+ wt-base co))))))))))))))))))
           ))
        (else (error "execute-conv-bwd-data-nchw: unsupported dtype" dtype)))))
      
  (define (execute-conv-bwd-data-nhwc dx-buf x-shape g-data g-shape wt-data
                                        N C H W KH KW SH SW PH PW OH OW out-ch dtype)
    "Backward w.r.t. X for NHWC: dX[n,ih,iw,c] += g[m,co]*WT[col_idx,co]."
    (let* ((H*W*C  (* H W C))
           (OH*OW  (* OH OW))
           (dx-size (shape-size x-shape)))
      (case dtype
        ((f32)
         (begin
           (for-range (i dx-size) (f32vector-set! dx-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n H*W*C))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (for-stride (kh KH) ((ih ih0 1))
                         (when-valid ih H
                           (for-stride (kw KW) ((iw iw0 1))
                             (when-valid iw W
                               (let* ((dx-i (+ n-src (* (+ (* ih W) iw) C) c))
                                      (col-idx (+ (* c KH KW) (* kh KW) kw))
                                      (wt-base (* col-idx out-ch)))
                                 (for-range (co out-ch)
                                   (f32vector-set! dx-buf dx-i
                                     (+ (f32vector-ref dx-buf dx-i)
                                        (* (f32vector-ref g-data (+ g-base co))
                                           (f32vector-ref wt-data (+ wt-base co)))))))))))))))))
           ))
        ((f64)
         (begin
           (for-range (i dx-size) (f64vector-set! dx-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n H*W*C))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (for-stride (kh KH) ((ih ih0 1))
                         (when-valid ih H
                           (for-stride (kw KW) ((iw iw0 1))
                             (when-valid iw W
                               (let* ((dx-i (+ n-src (* (+ (* ih W) iw) C) c))
                                      (col-idx (+ (* c KH KW) (* kh KW) kw))
                                      (wt-base (* col-idx out-ch)))
                                 (for-range (co out-ch)
                                   (f64vector-set! dx-buf dx-i
                                     (+ (f64vector-ref dx-buf dx-i)
                                        (* (f64vector-ref g-data (+ g-base co))
                                           (f64vector-ref wt-data (+ wt-base co)))))))))))))))))
           ))
        (else (error "execute-conv-bwd-data-nhwc: unsupported dtype" dtype)))))

  (define (execute-conv-bwd-weights-nchw dwt-buf wt-shape g-data g-shape src-data
                                           N C H W KH KW SH SW PH PW OH OW out-ch dtype)
    "Backward w.r.t. WT for NCHW: dwt[col_idx,co] += g[m,co]*src_val.
    src_val hoisted above co loop; dwt and g writes/reads are stride-1."
    (let* ((C*H*W  (* C H W))
           (H*W    (* H W))
           (OH*OW  (* OH OW))
           (dwt-size (shape-size wt-shape)))
      (case dtype
        ((f32)
         (begin
           (for-range (i dwt-size) (f32vector-set! dwt-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n C*H*W))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (let ((c-base (+ n-src (* c H*W))))
                         (for-stride (kh KH) ((ih ih0 1))
                           (when-valid ih H
                             (for-stride (kw KW) ((iw iw0 1))
                               (when-valid iw W
                                 (let* ((src-val (f32vector-ref src-data (+ c-base (* ih W) iw)))
                                        (col-idx (+ (* c KH KW) (* kh KW) kw))
                                        (wt-base (* col-idx out-ch)))
                                   (for-range (co out-ch)
                                     (f32vector-set! dwt-buf (+ wt-base co)
                                       (+ (f32vector-ref dwt-buf (+ wt-base co))
                                          (* (f32vector-ref g-data (+ g-base co)) src-val))))))))))))))))
           ))
        ((f64)
         (begin
           (for-range (i dwt-size) (f64vector-set! dwt-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n C*H*W))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (let ((c-base (+ n-src (* c H*W))))
                         (for-stride (kh KH) ((ih ih0 1))
                           (when-valid ih H
                             (for-stride (kw KW) ((iw iw0 1))
                               (when-valid iw W
                                 (let* ((src-val (f64vector-ref src-data (+ c-base (* ih W) iw)))
                                        (col-idx (+ (* c KH KW) (* kh KW) kw))
                                        (wt-base (* col-idx out-ch)))
                                   (for-range (co out-ch)
                                     (f64vector-set! dwt-buf (+ wt-base co)
                                       (+ (f64vector-ref dwt-buf (+ wt-base co))
                                          (* (f64vector-ref g-data (+ g-base co)) src-val))))))))))))))))
           ))
        (else (error "execute-conv-bwd-weights-nchw: unsupported dtype" dtype)))))

  (define (execute-conv-bwd-weights-nhwc dwt-buf wt-shape g-data g-shape src-data
                                           N C H W KH KW SH SW PH PW OH OW out-ch dtype)
    "Backward w.r.t. WT for NHWC: dwt[col_idx,co] += g[m,co]*src[n,ih,iw,c]."
    (let* ((H*W*C  (* H W C))
           (OH*OW  (* OH OW))
           (dwt-size (shape-size wt-shape)))
      (case dtype
        ((f32)
         (begin
           (for-range (i dwt-size) (f32vector-set! dwt-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n H*W*C))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (for-stride (kh KH) ((ih ih0 1))
                         (when-valid ih H
                           (for-stride (kw KW) ((iw iw0 1))
                             (when-valid iw W
                               (let* ((src-val (f32vector-ref src-data (+ n-src (* (+ (* ih W) iw) C) c)))
                                      (col-idx (+ (* c KH KW) (* kh KW) kw))
                                      (wt-base (* col-idx out-ch)))
                                 (for-range (co out-ch)
                                   (f32vector-set! dwt-buf (+ wt-base co)
                                     (+ (f32vector-ref dwt-buf (+ wt-base co))
                                        (* (f32vector-ref g-data (+ g-base co)) src-val)))))))))))))))
           ))
        ((f64)
         (begin
           (for-range (i dwt-size) (f64vector-set! dwt-buf i 0.0))
           (for-range (n N)
             (let ((n-src (* n H*W*C))
                   (n-row (* n OH*OW)))
               (for-stride (oh OH) ((ih0 (- 0 PH) SH))
                 (for-stride (ow OW) ((iw0 (- 0 PW) SW))
                   (let ((g-base (* (+ n-row (* oh OW) ow) out-ch)))
                     (for-range (c C)
                       (for-stride (kh KH) ((ih ih0 1))
                         (when-valid ih H
                           (for-stride (kw KW) ((iw iw0 1))
                             (when-valid iw W
                               (let* ((src-val (f64vector-ref src-data (+ n-src (* (+ (* ih W) iw) C) c)))
                                      (col-idx (+ (* c KH KW) (* kh KW) kw))
                                      (wt-base (* col-idx out-ch)))
                                 (for-range (co out-ch)
                                   (f64vector-set! dwt-buf (+ wt-base co)
                                     (+ (f64vector-ref dwt-buf (+ wt-base co))
                                        (* (f64vector-ref g-data (+ g-base co)) src-val)))))))))))))))
           ))
        (else (error "execute-conv-bwd-weights-nhwc: unsupported dtype" dtype)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; BLAS-Backed Convolution Execute Functions
  ;;;
  ;;; All three ops use the matmul-ready [N*OHOW, fan_in] col layout so that
  ;;; each operation maps to a single BLAS gemm call (no N-loop, no offsets).
  ;;; col buffer is freshly allocated per call.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (%conv-blas-gemm backend dtype M N K alpha A B beta C)
    "Select and call the dtype-appropriate gemm kernel from backend."
    (let ((fn (if (eq? dtype 'f64)
                  (blas-backend-gemm-f64 backend)
                  (blas-backend-gemm-f32 backend))))
      (fn M N K alpha A B beta C)))

  (define (%conv-blas-gemm-strided backend dtype M N K alpha A lda transa B ldb transb beta C)
    "Select and call the dtype-appropriate gemm-strided kernel from backend."
    (let ((fn (if (eq? dtype 'f64)
                  (blas-backend-gemm-strided-f64 backend)
                  (blas-backend-gemm-strided-f32 backend))))
      (fn M N K alpha A lda transa B ldb transb beta C)))

  (define (execute-conv-fwd-blas out-buf src wt b N C H W KH KW SH SW PH PW OH OW out-ch dtype
                                 #!optional (pre-col #f))
    "Forward conv via im2col-mr + single BLAS gemm + bias.
    col[M, fan_in] = im2col-mr(src);  out[M, out_ch] = col @ wt;  out += bias.
    When dtype is f32 and the active backend provides a conv-fwd-im2col-f32
    hot-kernel, that single C call is used instead of the Scheme im2col loop.
    pre-col: optional pre-allocated scratch buffer of size M*fan_in."
    (let* ((fan-in (* C KH KW))
           (M      (* N OH OW))
           (col    (or pre-col (allocate-typed-vector dtype (* M fan-in))))
           (be     (active-blas-backend))
           (hook   (and (eq? dtype 'f32) (blas-backend-conv-fwd-im2col-f32 be))))
      (if hook
          (hook out-buf b col src wt M out-ch fan-in out-ch N C H W KH KW SH SW PH PW OH OW)
          (begin
            (execute-im2col-batched-mr col src N C H W KH KW SH SW PH PW OH OW dtype)
            (%conv-blas-gemm be dtype M out-ch fan-in 1.0 col wt 0.0 out-buf)
            (if (eq? dtype 'f64)
                (for-range (m M)
                  (let ((base (* m out-ch)))
                    (for-range (co out-ch)
                      (f64vector-set! out-buf (+ base co)
                        (+ (f64vector-ref out-buf (+ base co))
                           (f64vector-ref b co))))))
                (for-range (m M)
                  (let ((base (* m out-ch)))
                    (for-range (co out-ch)
                      (f32vector-set! out-buf (+ base co)
                        (+ (f32vector-ref out-buf (+ base co))
                           (f32vector-ref b co)))))))))))

  (define (execute-conv-fwd-nhwc-blas out-buf src wt b N C H W KH KW SH SW PH PW OH OW out-ch dtype
                                      #!optional (pre-col #f))
    "Forward conv via im2col-nhwc-mr + single BLAS gemm + bias (NHWC input).
    When dtype is f32 and the active backend provides a conv-fwd-nhwc-im2col-f32
    hot-kernel, that single C call is used instead of the Scheme im2col loop."
    (let* ((fan-in (* C KH KW))
           (M      (* N OH OW))
           (col    (or pre-col (allocate-typed-vector dtype (* M fan-in))))
           (be     (active-blas-backend))
           (hook   (and (eq? dtype 'f32) (blas-backend-conv-fwd-nhwc-im2col-f32 be))))
      (if hook
          (hook out-buf b col src wt M out-ch fan-in out-ch N C H W KH KW SH SW PH PW OH OW)
          (begin
            (execute-im2col-nhwc-mr col src N C H W KH KW SH SW PH PW OH OW dtype)
            (%conv-blas-gemm be dtype M out-ch fan-in 1.0 col wt 0.0 out-buf)
            (if (eq? dtype 'f64)
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m out-ch)))
                    (do ((co 0 (+ co 1))) ((= co out-ch))
                      (f64vector-set! out-buf (+ base co)
                        (+ (f64vector-ref out-buf (+ base co))
                           (f64vector-ref b co))))))
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m out-ch)))
                    (do ((co 0 (+ co 1))) ((= co out-ch))
                      (f32vector-set! out-buf (+ base co)
                        (+ (f32vector-ref out-buf (+ base co))
                           (f32vector-ref b co)))))))))))
  
  (define (execute-conv-bwd-data-blas dx-buf x-shape g-data g-shape wt-data
                                        N C H W KH KW SH SW PH PW OH OW out-ch dtype
                                        #!optional (pre-col #f))
    "Backward w.r.t. X via single BLAS gemm (g @ wt^T) + col2im-mr.
     col[M, fan_in] = g[M, out_ch] @ wt[fan_in, out_ch]^T;  dx = col2im-mr(col).
     When dtype is f32 and the backend provides a conv-bwd-data-im2col-f32
     hot-kernel, that single C call is used instead."
    (let* ((fan-in (* C KH KW))
           (M      (* N OH OW))
           (col    (or pre-col (allocate-typed-vector dtype (* M fan-in))))
           (be     (active-blas-backend))
           (hook   (and (eq? dtype 'f32) (blas-backend-conv-bwd-data-im2col-f32 be))))
      (if hook
          (hook dx-buf col g-data wt-data M fan-in out-ch out-ch N C H W KH KW SH SW PH PW OH OW)
          (begin
            (%conv-blas-gemm-strided be dtype M fan-in out-ch
                                     1.0 g-data out-ch 'no-trans
                                     wt-data out-ch 'trans
                                     0.0 col)
            (execute-col2im-batched-mr dx-buf x-shape col (vector M fan-in) KH KW SH SW PH PW dtype)))))

  (define (execute-conv-bwd-data-nhwc-blas dx-buf x-shape g-data g-shape wt-data
                                            N C H W KH KW SH SW PH PW OH OW out-ch dtype
                                            #!optional (pre-col #f))
    "Backward w.r.t. X via single BLAS gemm (g @ wt^T) + col2im-nhwc-mr (NHWC output).
    When dtype is f32 and the backend provides a conv-bwd-data-nhwc-im2col-f32
    hot-kernel, that single C call is used instead."
    (let* ((fan-in (* C KH KW))
           (M      (* N OH OW))
           (col    (or pre-col (allocate-typed-vector dtype (* M fan-in))))
           (be     (active-blas-backend))
           (hook   (and (eq? dtype 'f32) (blas-backend-conv-bwd-data-nhwc-im2col-f32 be))))
      (if hook
          (hook dx-buf col g-data wt-data M fan-in out-ch out-ch N C H W KH KW SH SW PH PW OH OW)
          (begin
            (%conv-blas-gemm-strided be dtype M fan-in out-ch
                                     1.0 g-data out-ch 'no-trans
                                     wt-data out-ch 'trans
                                     0.0 col)
            (execute-col2im-nhwc-mr dx-buf x-shape col (vector M fan-in) KH KW SH SW PH PW dtype)))))

  (define (execute-conv-bwd-weights-blas dwt-buf wt-shape g-data g-shape src-data
                                           N C H W KH KW SH SW PH PW OH OW out-ch dtype
                                           #!optional (pre-col #f))
    "Backward w.r.t. WT via im2col-mr + single BLAS gemm (col^T @ g).
     col[M, fan_in] = im2col-mr(src);  dwt[fan_in, out_ch] = col^T @ g[M, out_ch].
     When dtype is f32 and the backend provides a conv-bwd-weights-im2col-f32
     hot-kernel, that single C call is used instead."
    (let* ((fan-in (* C KH KW))
           (M      (* N OH OW))
           (col    (or pre-col (allocate-typed-vector dtype (* M fan-in))))
           (be     (active-blas-backend))
           (hook   (and (eq? dtype 'f32) (blas-backend-conv-bwd-weights-im2col-f32 be))))
      (if hook
          (hook dwt-buf col src-data g-data fan-in out-ch M N C H W KH KW SH SW PH PW OH OW)
          (begin
            (execute-im2col-batched-mr col src-data N C H W KH KW SH SW PH PW OH OW dtype)
            (%conv-blas-gemm-strided be dtype fan-in out-ch M
                                     1.0 col fan-in 'trans
                                     g-data out-ch 'no-trans
                                     0.0 dwt-buf)))))

  (define (execute-conv-bwd-weights-nhwc-blas dwt-buf wt-shape g-data g-shape src-data
                                               N C H W KH KW SH SW PH PW OH OW out-ch dtype
                                               #!optional (pre-col #f))
    "Backward w.r.t. WT via im2col-nhwc-mr + single BLAS gemm (col^T @ g) (NHWC input).
    When dtype is f32 and the backend provides a conv-bwd-weights-nhwc-im2col-f32
    hot-kernel, that single C call is used instead."
    (let* ((fan-in (* C KH KW))
           (M      (* N OH OW))
           (col    (or pre-col (allocate-typed-vector dtype (* M fan-in))))
           (be     (active-blas-backend))
           (hook   (and (eq? dtype 'f32) (blas-backend-conv-bwd-weights-nhwc-im2col-f32 be))))
      (if hook
          (hook dwt-buf col src-data g-data fan-in out-ch M N C H W KH KW SH SW PH PW OH OW)
          (begin
            (execute-im2col-nhwc-mr col src-data N C H W KH KW SH SW PH PW OH OW dtype)
            (%conv-blas-gemm-strided be dtype fan-in out-ch M
                                     1.0 col fan-in 'trans
                                     g-data out-ch 'no-trans
                                     0.0 dwt-buf)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Reduction Morphism Execution
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (realize-reduction m)
    "Realize reduction morphism
    
    Performs accumulation over specified axes"
    
    (cases array-morphism m
      (reduction-morphism (morph-id op operand reduce-axes index-fn shape dtype batch-axis)
        
        ;; Realize source first
        (let ((src-realized (realize operand)))
          
          (cases array-morphism src-realized
            (concrete-array (src-data src-shape src-strides src-offset src-dtype _ _)
              
              (let* ((size (shape-size shape))
                     (result-data (allocate-typed-vector dtype size))
                     (reducer (reduction-index-fn-reducer index-fn))
                     (keepdims? (reduction-index-fn-keepdims? index-fn))
                     (src-size (shape-size src-shape)))
                
                ;; Execute reduction
                (execute-reduction-morphism op result-data shape
                                            src-data src-shape
                                            src-strides src-offset
                                            reduce-axes reducer 
                                            keepdims? dtype src-dtype)
                
                (concrete-array result-data shape 
                              (compute-strides shape) 0 dtype -1
                              batch-axis)))
            
            (else (error "Reduction source must be concrete")))))
      
      (else (error "Expected reduction-morphism" m))))

  (define (execute-reduction-morphism op output-buffer out-shape
                                      src-data src-shape
                                      src-strides src-offset
                                      reduce-axes reducer
                                      keepdims? out-dtype src-dtype)
    "Execute reduction accumulation.

    Args:
    op:           Reduction op symbol (sum, mean, max, min, prod).
    output-buffer: Pre-allocated output typed vector.
    out-shape:    Output shape vector.
    src-data:     Source typed vector.
    src-shape:    Source shape vector.
    src-strides:  Source strides vector (may differ from row-major).
    src-offset:   Source base offset into src-data.
    reduce-axes:  List of axes to reduce.
    reducer:      Binary accumulation function (already specialised for op).
    keepdims?:    Whether reduced axes are kept as size 1.
    out-dtype:    Output dtype symbol.
    src-dtype:    Source dtype symbol."

    (let* ((out-size (shape-size out-shape))
           (src-size (shape-size src-shape))
           (src-rank (vector-length src-shape)))

      ;; Fast path: 2D row-major source with a single axis reduced.
      ;; Avoids per-element linear-to-multi-index and multi-to-linear-index calls.
      (define (row-major-2d?)
        (and (= src-rank 2)
             (= src-offset 0)
             (= (vector-ref src-strides 1) 1)
             (= (vector-ref src-strides 0) (vector-ref src-shape 1))))

      (define (fast-reduce-2d-axis0!)
        ;; Reduce [M, N] over axis 0 -> [N]; op must be sum/mean/max/min.
        (let* ((M (vector-ref src-shape 0))
               (N (vector-ref src-shape 1))
               (init (case op ((sum mean) 0.0) ((max) -inf.0) ((min) +inf.0) (else 0.0))))
          (case src-dtype
            ((f32)
             (begin
               (do ((n 0 (+ n 1))) ((= n N)) (f32vector-set! output-buffer n init))
               (case op
                 ((sum mean)
                  (do ((m 0 (+ m 1))) ((= m M))
                    (let ((base (* m N)))
                      (do ((n 0 (+ n 1))) ((= n N))
                        (f32vector-set! output-buffer n
                                        (+ (f32vector-ref output-buffer n)
                                           (f32vector-ref src-data (+ base n))))))))
                 ((max)
                  (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m N)))
                    (do ((n 0 (+ n 1))) ((= n N))
                      (let ((v (f32vector-ref src-data (+ base n))))
                        (when (> v (f32vector-ref output-buffer n))
                          (f32vector-set! output-buffer n v)))))))
                 ((min)
                  (do ((m 0 (+ m 1))) ((= m M))
                    (let ((base (* m N)))
                      (do ((n 0 (+ n 1))) ((= n N))
                        (let ((v (f32vector-ref src-data (+ base n))))
                          (when (< v (f32vector-ref output-buffer n))
                            (f32vector-set! output-buffer n v))))))))
               (when (eq? op 'mean)
                 (do ((n 0 (+ n 1))) ((= n N))
                   (f32vector-set! output-buffer n (/ (f32vector-ref output-buffer n) M))))
             ))
            ((f64)
             (begin
               (do ((n 0 (+ n 1))) ((= n N)) (f64vector-set! output-buffer n init))
               (case op
                 ((sum mean)
                  (do ((m 0 (+ m 1))) ((= m M))
                    (let ((base (* m N)))
                      (do ((n 0 (+ n 1))) ((= n N))
                        (f64vector-set! output-buffer n
                                        (+ (f64vector-ref output-buffer n)
                                           (f64vector-ref src-data (+ base n))))))))
                 ((max)
                  (do ((m 0 (+ m 1))) ((= m M))
                    (let ((base (* m N)))
                      (do ((n 0 (+ n 1))) ((= n N))
                        (let ((v (f64vector-ref src-data (+ base n))))
                          (when (> v (f64vector-ref output-buffer n))
                            (f64vector-set! output-buffer n v)))))))
                 ((min)
                  (do ((m 0 (+ m 1))) ((= m M))
                    (let ((base (* m N)))
                      (do ((n 0 (+ n 1))) ((= n N))
                        (let ((v (f64vector-ref src-data (+ base n))))
                          (when (< v (f64vector-ref output-buffer n))
                            (f64vector-set! output-buffer n v))))))))
               (when (eq? op 'mean)
                 (do ((n 0 (+ n 1))) ((= n N))
                   (f64vector-set! output-buffer n (/ (f64vector-ref output-buffer n) M))))
               ))
            ))
        )
      
      (define (fast-reduce-2d-axis1!)
        ;; Reduce [M, N] over axis 1 -> scalar per row; op sum/mean/max/min.
        ;; Output: [M] if keepdims?=#f, [M,1] if keepdims?=#t (same flat layout).
        (let* ((M (vector-ref src-shape 0))
               (N (vector-ref src-shape 1)))
          (case src-dtype
            ((f32)
             (case op
               ((sum mean)
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m N))
                        (acc 0.0))
                    (do ((n 0 (+ n 1))) ((= n N))
                      (set! acc (+ acc (f32vector-ref src-data (+ base n)))))
                    (f32vector-set! output-buffer m
                      (if (eq? op 'mean) (/ acc N) acc)))))
               ((max)
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m N))
                        (acc -inf.0))
                    (do ((n 0 (+ n 1))) ((= n N))
                      (let ((v (f32vector-ref src-data (+ base n))))
                        (when (> v acc) (set! acc v))))
                    (f32vector-set! output-buffer m acc))))
               ((min)
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m N))
                        (acc +inf.0))
                    (do ((n 0 (+ n 1))) ((= n N))
                      (let ((v (f32vector-ref src-data (+ base n))))
                        (when (< v acc) (set! acc v))))
                    (f32vector-set! output-buffer m acc))))))
            ((f64)
             (case op
               ((sum mean)
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m N))
                        (acc 0.0))
                    (do ((n 0 (+ n 1))) ((= n N))
                      (set! acc (+ acc (f64vector-ref src-data (+ base n)))))
                    (f64vector-set! output-buffer m
                      (if (eq? op 'mean) (/ acc N) acc)))))
               ((max)
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m N))
                        (acc -inf.0))
                    (do ((n 0 (+ n 1))) ((= n N))
                      (let ((v (f64vector-ref src-data (+ base n))))
                        (when (> v acc) (set! acc v))))
                    (f64vector-set! output-buffer m acc))))
               ((min)
                (do ((m 0 (+ m 1))) ((= m M))
                  (let ((base (* m N))
                        (acc +inf.0))
                    (do ((n 0 (+ n 1))) ((= n N))
                      (let ((v (f64vector-ref src-data (+ base n))))
                        (when (< v acc) (set! acc v))))
                    (f64vector-set! output-buffer m acc))))
               ))
            ))
        )

      (define *1d-reduce-ones-cache* '())

      (define (ones-vector dtype N)
        (let ((key (cons dtype N)))
          (let loop ((cache *1d-reduce-ones-cache*))
            (cond ((null? cache)
                   (let ((v (allocate-typed-vector dtype N)))
                     (case dtype
                       ((f32) (do ((i 0 (+ i 1))) ((= i N)) (f32vector-set! v i 1.0)))
                       ((f64) (do ((i 0 (+ i 1))) ((= i N)) (f64vector-set! v i 1.0)))
                       (else  (do ((i 0 (+ i 1))) ((= i N)) (typed-vector-set! v dtype i 1.0))))
                     (set! *1d-reduce-ones-cache* (cons (cons key v) *1d-reduce-ones-cache*))
                     v))
                  ((and (eq? (car key) (caar cache)) (= (cdr key) (cdar cache)))
                   (cdar cache))
                  (else (loop (cdr cache)))))))

      (define (reduce-1d-sum-mean-via-dot! dtype N op)
        (let ((fn (cond ((and (eq? dtype 'f32) *active-backend*)
                         (blas-backend-dot-f32 *active-backend*))
                        ((and (eq? dtype 'f64) *active-backend*)
                         (blas-backend-dot-f64 *active-backend*))
                        (else #f))))
          (if fn
              (let ((val (fn N src-data (ones-vector dtype N))))
                (typed-vector-set! output-buffer dtype 0
                                   (if (eq? op 'mean) (/ val N) val)))
              ;; Fallback to tight loop when no BLAS backend/kernel is available.
              (case dtype
                ((f32)
                 (let ((acc 0.0))
                   (do ((i 0 (+ i 1))) ((= i N))
                     (set! acc (+ acc (f32vector-ref src-data i))))
                   (f32vector-set! output-buffer 0 (if (eq? op 'mean) (/ acc N) acc))))
                ((f64)
                 (let ((acc 0.0))
                   (do ((i 0 (+ i 1))) ((= i N))
                     (set! acc (+ acc (f64vector-ref src-data i))))
                   (f64vector-set! output-buffer 0 (if (eq? op 'mean) (/ acc N) acc))))))))

      (cond
        ;; Fast path: 1D row-major, reduce all axes (scalar result).
        ((and (= src-rank 1)
              (= src-offset 0)
              (= (vector-ref src-strides 0) 1)
              (null? (cdr reduce-axes))   ; single axis
              (= (car reduce-axes) 0)
              (memq op '(sum mean max min)))
          (let ((N (vector-ref src-shape 0)))
            (case src-dtype
              ((f32)
               (case op
                 ((sum mean)
                  (reduce-1d-sum-mean-via-dot! 'f32 N op))
                 ((max)
                  (let ((acc -inf.0))
                    (do ((i 0 (+ i 1))) ((= i N))
                      (let ((v (f32vector-ref src-data i)))
                        (when (> v acc) (set! acc v))))
                    (f32vector-set! output-buffer 0 acc)))
                 ((min)
                  (let ((acc +inf.0))
                    (do ((i 0 (+ i 1))) ((= i N))
                      (let ((v (f32vector-ref src-data i)))
                        (when (< v acc) (set! acc v))))
                    (f32vector-set! output-buffer 0 acc)))))
              ((f64)
               (case op
                 ((sum mean)
                  (reduce-1d-sum-mean-via-dot! 'f64 N op))
                 ((max)
                  (let ((acc -inf.0))
                    (do ((i 0 (+ i 1))) ((= i N))
                      (let ((v (f64vector-ref src-data i)))
                        (when (> v acc) (set! acc v))))
                    (f64vector-set! output-buffer 0 acc)))
                 ((min)
                  (let ((acc +inf.0))
                    (do ((i 0 (+ i 1))) ((= i N))
                      (let ((v (f64vector-ref src-data i)))
                        (when (< v acc) (set! acc v))))
                    (f64vector-set! output-buffer 0 acc)))))
              (else (error "execute-reduction-morphism: unsupported dtype for 1D fast path" src-dtype)))))

        ;; Fast path: 2D row-major, reduce axis 0
        ((and (row-major-2d?)
              (equal? reduce-axes '(0))
              (memq op '(sum mean max min)))
         (fast-reduce-2d-axis0!))

        ;; Fast path: 2D row-major, reduce axis 1
        ((and (row-major-2d?)
              (equal? reduce-axes '(1))
              (memq op '(sum mean max min)))
         (fast-reduce-2d-axis1!))

        (else
         ;; General path: initialise output then accumulate over all source positions.
         (let ((init-val (case op
                           ((sum mean) 0.0)
                           ((prod)     1.0)
                           ((max)      -inf.0)
                           ((min)      +inf.0)
                           (else       0.0))))
           (do ((i 0 (+ i 1)))
               ((= i out-size))
             (typed-vector-set! output-buffer out-dtype i init-val)))
         ;; Accumulate: iterate over every logical source position.
         (do ((src-i 0 (+ src-i 1)))
             ((= src-i src-size))
           (let* ((src-multi (vector->list (linear-to-multi-index src-i src-shape)))
                  (physical  (multi-to-linear-index (list->vector src-multi)
                                                    src-strides
                                                    src-offset))
                  (out-idx
                   (if keepdims?
                       (map (lambda (i val)
                              (if (member i reduce-axes) 0 val))
                            (iota src-rank) src-multi)
                       (fold-right
                        (lambda (i val acc)
                          (if (member i reduce-axes) acc (cons val acc)))
                        '()
                        (iota src-rank) src-multi)))
                  (out-linear (multi-to-linear-index
                               (list->vector out-idx)
                               (compute-strides out-shape)))
                  (src-val    (typed-vector-ref src-data    src-dtype physical))
                  (out-val    (typed-vector-ref output-buffer out-dtype out-linear))
                  (new-val    (reducer out-val src-val)))
             (typed-vector-set! output-buffer out-dtype out-linear new-val)))
         ;; Post-processing: divide accumulated sum by element count for mean.
         (when (eq? op 'mean)
           (let ((reduce-size
                  (fold * 1 (map (lambda (ax) (vector-ref src-shape ax))
                                 reduce-axes))))
             (do ((i 0 (+ i 1)))
                 ((= i out-size))
               (let ((val (typed-vector-ref output-buffer out-dtype i)))
                 (typed-vector-set! output-buffer out-dtype i
                                    (/ val reduce-size))))))
         ))
      ))


  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Composed Index Function Execution
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (execute-composed-morphism fn output-buffer shape operands dtype)
    "Execute composed index function: (f ∘ g)(i) = f(g(i))"
    
    (let* ((outer (composed-index-fn-outer fn))
           (inner (composed-index-fn-inner fn))
           (size (shape-size shape)))
      
      (do ((i 0 (+ i 1)))
          ((= i size))
        
        (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
               
               ;; Apply inner function
               (inner-idx (apply-index-fn inner out-idx))
               
               ;; Apply outer function
               (src-idx (apply-index-fn outer inner-idx))
               
               ;; Retrieve value
               (value (if (= (length operands) 1)
                         (retrieve-value (car operands) src-idx)
                         (error "Composed morphism with multiple operands not yet supported"))))
          
          (typed-vector-set! output-buffer dtype i value)))))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Procedure Index Function Execution
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (execute-procedure-index-fn fn output-buffer shape operands dtype)
    "Execute direct procedure index function"
    
    (let ((size (shape-size shape)))
      
      (do ((i 0 (+ i 1)))
          ((= i size))
        
        (let* ((out-idx (vector->list (linear-to-multi-index i shape)))
               
               ;; Apply procedure
               (result (fn out-idx))
               
               ;; Handle result
               (value
                (cond
                  ;; Padding markers
                  ((and (pair? result) (eq? (car result) 'constant))
                   (cadr result))
                  ((eq? result 'pad-zero)
                   0.0)
                  
                  ;; Direct value
                  ((number? result) result)
                  
                  ;; Index to retrieve
                  ((list? result)
                   (if (null? operands)
                       (error "Procedure returned index but no operands provided")
                       (retrieve-value (car operands) result)))
                  
                  (else
                   (error "Invalid procedure index function result" result)))))
          
          (typed-vector-set! output-buffer dtype i value)))))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Value Retrieval Helpers
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (retrieve-value morphism indices)
    "Retrieve value from morphism at given indices
    
    Args:
      morphism: Concrete array morphism
      indices: List of indices
    
    Returns:
      Scalar value"
    
    (unless (concrete-array? morphism)
      (error "Can only retrieve from concrete array" morphism))
    
    (cases array-morphism morphism
      (concrete-array (data shape strides offset dtype _ _)
        
        (let ((linear-idx (multi-to-linear-index 
                          (list->vector indices)
                          strides
                          offset)))
          (typed-vector-ref data dtype linear-idx)))
      
      (else (error "Expected concrete array"))))
  
  (define (retrieve-value-with-padding morphism indices padding-value)
    "Retrieve value with bounds checking and padding
    
    Returns padding-value if indices are out of bounds"
    
    (cases array-morphism morphism
      (concrete-array (data shape strides offset dtype _ _)
        
        ;; Check bounds
        (let ((in-bounds?
               (every (lambda (idx dim)
                        (and (>= idx 0) (< idx dim)))
                      indices
                      (vector->list shape))))
          
          (if in-bounds?
              (retrieve-value morphism indices)
              padding-value)))
      
      (else (error "Expected concrete array"))))
  
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Procedure Name Helper (for reduction detection)
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  
  (define (procedure-name proc)
    "Attempt to get name of procedure (for built-in operators)"
    (cond
      ((eq? proc +) '+)
      ((eq? proc *) '*)
      ((eq? proc max) 'max)
      ((eq? proc min) 'min)
      (else 'unknown)))
  
) ;; end module
