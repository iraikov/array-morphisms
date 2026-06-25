;;; array-morphisms-blas-compat.scm
;;; Phase 1: BLAS Compatibility Predicates
;;;
;;; Detects whether operations can be profitably routed to BLAS kernels.
;;; All functions are pure predicates with no side effects.
;;;
;;; Detection hierarchy:
;;;   contiguous-row-major?       -- layout check on concrete arrays
;;;   blas-compatible-matmul?     -- GEMM readiness for two matrices
;;;   blas-compatible-matvec?     -- GEMV readiness for matrix + vector
;;;   blas-compatible-dot?        -- DOT readiness for two vectors
;;;   blas-compatible-operation?  -- high-level morphism-expr check

(module array-morphisms-blas-compat

  (;; Layout predicates
   contiguous-row-major?
   contiguous-column-major?

   ;; Operation compatibility (work on concrete-array arguments)
   blas-compatible-matmul?
   blas-compatible-matvec?
   blas-compatible-dot?
   blas-compatible-vecvec?
   matmul-concrete?
   array->gemm-blas-params

   ;; High-level morphism-expr compatibility
   blas-compatible-operation?

   ;; Metadata extraction
   get-blas-function-name)

  (import scheme (chicken base))
  (import (only srfi-1 every))
  (import datatype)
  (import array-morphisms-core)

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Layout Predicates
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (contiguous-row-major? m)
    "True when concrete array m has standard row-major layout.

    A row-major contiguous array satisfies:
      offset  = 0
      strides = compute-strides(shape)   (no gaps, no padding)

    Abstract morphisms always return #f."
    (cases array-morphism m
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (and (= offset 0)
             (let* ((expected (compute-strides shape))
                    (rank     (vector-length strides)))
               (let loop ((i 0))
                 (cond
                   ((= i rank)                                  #t)
                   ((= (vector-ref strides i)
                       (vector-ref expected i))                 (loop (+ i 1)))
                   (else                                        #f))))))
      (else #f)))

  (define (contiguous-column-major? m)
    "True when concrete array m has column-major (Fortran) layout.

    Column-major satisfies:
      offset     = 0
      strides[0] = 1
      strides[i] = strides[i-1] * shape[i-1]

    Example: shape (3,2) -> strides (1, 3)."
    (cases array-morphism m
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (and (= offset 0)
             (let ((rank (vector-length strides)))
               (let loop ((i 0) (s 1))
                 (cond
                   ((= i rank) #t)
                   ((= (vector-ref strides i) s)
                    (loop (+ i 1) (* s (vector-ref shape i))))
                   (else #f))))))
      (else #f)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMM: Matrix-Matrix Compatibility
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (blas-compatible-matmul? m1 m2)
    "True when matrix-matrix multiply can use BLAS GEMM.

    Requirements:
      - Both are concrete arrays
      - Both are 2-D matrices
      - Same dtype, which must be f32 or f64
      - Both have contiguous row-major layout
      - Inner K dimension matches: shape(m1)[1] = shape(m2)[0]"
    (and (concrete-array? m1)
         (concrete-array? m2)
         (= 2 (morph-rank m1))
         (= 2 (morph-rank m2))
         (let ((d1 (get-morphism-dtype m1))
               (d2 (get-morphism-dtype m2)))
           (and (memq d1 '(f32 f64))
                (eq? d1 d2)))
         (contiguous-row-major? m1)
         (contiguous-row-major? m2)
         (let ((s1 (get-morphism-shape m1))
               (s2 (get-morphism-shape m2)))
           (= (vector-ref s1 1)
              (vector-ref s2 0)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMV: Matrix-Vector Compatibility
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (blas-compatible-matvec? mat vec)
    "True when matrix-vector multiply can use BLAS GEMV.

    Requirements:
      - mat is 2-D, vec is 1-D
      - Both concrete and contiguous row-major
      - Same dtype (f32 or f64)
      - N dimension matches: shape(mat)[1] = shape(vec)[0]"
    (and (concrete-array? mat)
         (concrete-array? vec)
         (= 2 (morph-rank mat))
         (= 1 (morph-rank vec))
         (let ((dm (get-morphism-dtype mat))
               (dv (get-morphism-dtype vec)))
           (and (memq dm '(f32 f64))
                (eq? dm dv)))
         (contiguous-row-major? mat)
         (contiguous-row-major? vec)
         (let ((sm (get-morphism-shape mat))
               (sv (get-morphism-shape vec)))
           (= (vector-ref sm 1)
              (vector-ref sv 0)))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; DOT: Vector Dot-Product Compatibility
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (blas-compatible-dot? v1 v2)
    "True when vector dot product can use BLAS DOT.

    Requirements:
      - Both 1-D concrete arrays
      - Same dtype (f32 or f64)
      - Both contiguous
      - Same length"
    (and (concrete-array? v1)
         (concrete-array? v2)
         (= 1 (morph-rank v1))
         (= 1 (morph-rank v2))
         (let ((d1 (get-morphism-dtype v1))
               (d2 (get-morphism-dtype v2)))
           (and (memq d1 '(f32 f64))
                (eq? d1 d2)))
         (contiguous-row-major? v1)
         (contiguous-row-major? v2)
         (equal? (get-morphism-shape v1)
                 (get-morphism-shape v2))))

  (define (blas-compatible-vecvec? op v1 v2)
    "True when a vector-vector operation can use BLAS.
    Currently delegates to blas-compatible-dot? (same requirements)."
    (blas-compatible-dot? v1 v2))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMM: Stride-Any Matrix-Matrix Compatibility
  ;;;
  ;;; Like blas-compatible-matmul? but without the row-major requirement.
  ;;; Used to route transposed/strided operands to the Scheme fallback kernel
  ;;; (execute-scheme-gemm), which is stride-aware.
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (matmul-concrete? m1 m2)
    "True when a Scheme-fallback matmul can be executed.

    Requirements:
      - Both are concrete arrays
      - Both are 2-D matrices
      - Same dtype, which must be f32 or f64
      - Inner K dimension matches: shape(m1)[1] = shape(m2)[0]

    No layout constraint: accepts row-major, column-major, or arbitrarily
    strided (e.g. transposed zero-copy view) arrays."
    (and (concrete-array? m1)
         (concrete-array? m2)
         (= 2 (morph-rank m1))
         (= 2 (morph-rank m2))
         (let ((d1 (get-morphism-dtype m1))
               (d2 (get-morphism-dtype m2)))
           (and (memq d1 '(f32 f64))
                (eq? d1 d2)))
         (= (vector-ref (get-morphism-shape m1) 1)
            (vector-ref (get-morphism-shape m2) 0))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; GEMM Parameter Extraction for Strided / Transposed Operands
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (array->gemm-blas-params A)
    "Extract BLAS GEMM parameters from a 2-D concrete array.

    Returns (data transa lda) when A can be expressed as a cBLAS GEMM
    operand using a Trans/NoTrans flag, or #f when the layout is not
    expressible this way (e.g. non-unit minor stride, non-zero offset).

    Accepted layouts (offset must be 0):
      Row-major:            strides = (C, 1), s1==1 AND s0==C
                            -> (data 'no-trans C)   lda = C
      Transposed row-major: strides = (1, R), s0==1 AND s1==R
                            -> (data 'trans    R)   lda = R

    The returned lda is always the physical leading dimension of the
    underlying row-major buffer, as required by cblas_dgemm/sgemm."
    (cases array-morphism A
      (concrete-array (data shape strides offset dtype alloc-id batch-axis)
        (and (= (vector-length shape) 2)
             (= offset 0)
             (let ((R  (vector-ref shape 0))
                   (C  (vector-ref shape 1))
                   (s0 (vector-ref strides 0))
                   (s1 (vector-ref strides 1)))
               (cond
                 ((and (= s1 1) (= s0 C))  (list data 'no-trans C))
                 ((and (= s0 1) (= s1 R))  (list data 'trans    R))
                 (else #f)))))
      (else #f)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; High-Level Morphism-Expr Compatibility Check
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (blas-compatible-operation? morphism)
    "Check whether a morphism-expr can be executed via BLAS.

    Returns (blas-op . operands) when compatible, #f otherwise.
    blas-op is one of: gemm, gemv, dot.

    Note: All operands must already be concrete.  Phase 3 realizes
    operands before calling this predicate."
    (cases array-morphism morphism
      (morphism-expr (morph-id op operands index-fn shape dtype metadata batch-axis)
        (cond

          ;; Matrix-matrix multiply -> GEMM (row-major: BLAS or Scheme)
          ((and (eq? op 'matmul)
                (= (length operands) 2)
                (blas-compatible-matmul? (car operands) (cadr operands)))
           (cons 'gemm operands))

          ;; Strided/transposed matmul -> gemm-strided (Scheme kernel only)
          ;; Catches non-row-major concrete operands (e.g. zero-copy transpose
          ;; views) that blas-compatible-matmul? rejects but execute-scheme-gemm
          ;; can handle correctly via its stride-aware triple loop.
          ((and (eq? op 'matmul)
                (= (length operands) 2)
                (matmul-concrete? (car operands) (cadr operands)))
           (cons 'gemm-strided operands))

          ;; Matrix-vector multiply -> GEMV
          ((and (eq? op 'matvec)
                (= (length operands) 2)
                (blas-compatible-matvec? (car operands) (cadr operands)))
           (cons 'gemv operands))

          ;; Vector dot product -> DOT
          ((and (eq? op 'dot)
                (= (length operands) 2)
                (blas-compatible-dot? (car operands) (cadr operands)))
           (cons 'dot operands))

          (else #f)))
      (else #f)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Function Name Mapping
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (get-blas-function-name operation dtype)
    "Return the Chicken blas egg procedure symbol for a given operation and dtype.

    The returned symbols correspond to the safe destructive (!) variants for
    in-place operations, and the safe pure variant for DOT (which returns a
    scalar and has no in-place form).  These are the names exported by the
    Chicken 5 'blas' egg.

    Returns symbol or #f for unknown combinations.

    Examples:
      (get-blas-function-name 'gemm 'f64) => dgemm!
      (get-blas-function-name 'dot  'f32) => sdot
      (get-blas-function-name 'axpy 'f64) => daxpy!"
    (case operation
      ((gemm) (if (eq? dtype 'f64) 'dgemm!  'sgemm!))
      ((gemv) (if (eq? dtype 'f64) 'dgemv!  'sgemv!))
      ((dot)  (if (eq? dtype 'f64) 'ddot    'sdot))
      ((axpy) (if (eq? dtype 'f64) 'daxpy!  'saxpy!))
      ((scal) (if (eq? dtype 'f64) 'dscal!  'sscal!))
      (else   #f)))

) ;; end module array-morphisms-blas-compat
