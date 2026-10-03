;;; segop-lower.scm
;;;
;;; Lowering of element-wise SSA bindings to segops, and the two built-in
;;; segop backends that turn them into replay executors.  Included into the
;;; array-morphisms-ssa module.
;;;
;;; An element-wise binding becomes a seg-map over the binding's shape.
;;; Each input is read through the NumPy broadcast map from the output's
;;; coordinates to its own, with the layout (strides and offset) of its
;;; trace-time array; the output is the row-major pool buffer.  The body
;;; names the binding's op when the op is in the scalar-op table and the
;;; binding's combiner is that op's combiner, and is (in 0) for a cast,
;;; whose conversion happens when the value is stored; otherwise it applies
;;; the combiner as an opaque procedure.  The combiner is kept as the segop's
;;; fallback.
;;;
;;; An executor is a procedure (executor deref out-buf): deref maps a
;;; replay-ref (the slot of an input operand) to its current array, and
;;; out-buf is the output's pool buffer.  The executor writes out-buf.
;;;
;;; The backends recognize the operand layouts below, tried in this order:
;;;
;;;   activation-unary   one input of the output's shape, row-major, body
;;;                      (op (in 0)) with op a registered activation op;
;;;                      uses the activation backend's kernel when present
;;;   flat-unary         one row-major input of the output's shape
;;;   activation-binary  two such inputs, body (op (in 0) (in 1)) with op a
;;;                      registered binary op
;;;   flat-binary        two row-major inputs of the output's shape
;;;   bias-broadcast     a row-major input of the output's shape and a
;;;                      row-major vector as long as the last dimension
;;;   flat-ternary       three row-major f32 or f64 inputs of the output's
;;;                      shape and dtype
;;;   flat-broadcast     two row-major f32 or f64 inputs, each of the
;;;                      output's shape, one value per row ([..., 1]), one
;;;                      value per column ([N]) or a single value
;;;
;;; The registry backend takes the patterns that use registered kernels:
;;; activation-unary, activation-binary, and bias-broadcast and
;;; flat-broadcast for the plain add, sub, mul and div bodies.  The scheme
;;; backend takes the others.  A binding that matches none of them is
;;; compiled to ri-index.

;;; ------------------------------------------------------------
;;; Binding bodies
;;; ------------------------------------------------------------

(define (input-body-args n) (map (lambda (k) `(in ,k)) (iota n)))

(define (binding-scalar-body b combiner)
  "The scalar body of element-wise binding b, whose combiner is combiner:
   the composed body of a fused binding, (in 0) for a cast, (op (in 0) ...)
   when op is a scalar-op table entry of the right arity whose combiner is
   combiner, and (opaque combiner (in 0) ...) otherwise."
  (let* ((meta (ssa-binding-meta b))
         (op   (ssa-binding-op b))
         (n    (length (ssa-binding-inputs b)))
         (bp   (assq 'body meta)))
    (cond (bp (cdr bp))
          ((and (eq? op 'cast) (= n 1)
                (eq? combiner (cast-combiner (ssa-binding-dtype b))))
           '(in 0))
          ((and (not (assq 'fused? meta))
                (symbol? op)
                (scalar-op? op)
                (= (scalar-op-arity op) n)
                (eq? combiner (scalar-op-combiner op)))
           (cons op (input-body-args n)))
          (else `(opaque ,combiner ,@(input-body-args n))))))

(define (compose-binding-bodies p-body p-nargs c-body)
  "The body of a producer fused into its consumer: the consumer's (in 0) is
   replaced by the producer's body, and its other inputs follow the
   producer's, as in compose-flat-combiners."
  (scalar-expr-substitute-inputs
   c-body
   (lambda (k) (if (= k 0) p-body `(in ,(+ p-nargs k -1))))))

;; The op of a body that applies one op directly to its inputs (in 0) ...
;; (in (n - 1)), or #f.
(define (single-op-body-op body n)
  (and (pair? body)
       (symbol? (car body))
       (not (eq? (car body) 'opaque))
       (equal? (cdr body) (input-body-args n))
       (car body)))

;;; ------------------------------------------------------------
;;; Lowering
;;; ------------------------------------------------------------

;; True when an operand of shape in-shape broadcasts to out-shape.
(define (broadcastable? in-shape out-shape)
  (let ((k (vector-length in-shape)) (r (vector-length out-shape)))
    (and (<= k r)
         (let loop ((j 0))
           (or (= j k)
               (let ((n (vector-ref in-shape j)))
                 (and (or (= n 1) (= n (vector-ref out-shape (+ j (- r k)))))
                      (loop (+ j 1)))))))))

(define (lower-elementwise-binding b combiner in-traces in-refs pool-idx shape strides dtype)
  "The seg-map of element-wise binding b with combiner combiner, whose
   inputs had the trace-time arrays in-traces and are found at in-refs, and
   whose output is pool slot pool-idx of the given shape, strides and dtype;
   #f when an input does not broadcast to the output shape."
  (let ((rank (vector-length shape)))
    (and (every (lambda (tr) (broadcastable? (trace-arr-shape tr) shape)) in-traces)
         (make-seg-map
          shape
          (map (lambda (tr ref)
                 (cases array-morphism tr
                   (concrete-array (_ s st off dt _ _)
                     (make-segop-operand ref dt s st off (broadcast-index-map s rank)))
                   (else (error "lower-elementwise-binding: input not concrete" tr))))
               in-traces in-refs)
          (list (make-segop-operand pool-idx dtype shape strides 0 (identity-index-map rank)))
          (list (binding-scalar-body b combiner))
          combiner))))

;;; ------------------------------------------------------------
;;; Operand patterns
;;; ------------------------------------------------------------

(define (operand-row-major? o)
  (and (= (segop-operand-offset o) 0)
       (equal? (segop-operand-strides o) (compute-strides (segop-operand-shape o)))))

;; Row-major with the shape of the space.
(define (operand-flat? o space)
  (and (operand-row-major? o) (equal? (segop-operand-shape o) space)))

(define (bias-operands? ins space)
  (and (= (length ins) 2)
       (> (vector-length space) 0)
       (operand-flat? (car ins) space)
       (operand-row-major? (cadr ins))
       (let ((bs (segop-operand-shape (cadr ins))))
         (and (= (vector-length bs) 1)
              (= (vector-ref bs 0) (vector-ref space (- (vector-length space) 1)))))))

;; The #(rsA csA rsB csB) steps of a flat-broadcast, or #f.
(define (operands-broadcast-steps ins space dtype)
  (and (= (length ins) 2)
       (> (vector-length space) 0)
       (memq dtype '(f32 f64))
       (every (lambda (o) (and (operand-row-major? o) (eq? (segop-operand-dtype o) dtype))) ins)
       (let ((a (broadcast-steps space (segop-operand-shape (car ins))))
             (b (broadcast-steps space (segop-operand-shape (cadr ins)))))
         (and a b (list->vector (append a b))))))

(define (ternary-operands? ins space dtype)
  (and (= (length ins) 3)
       (memq dtype '(f32 f64))
       (every (lambda (o) (and (operand-flat? o space) (eq? (segop-operand-dtype o) dtype))) ins)))

;;; ------------------------------------------------------------
;;; Executors
;;; ------------------------------------------------------------

(define-syntax with-array-data
  (syntax-rules ()
    ((_ who arr (data dt) body ...)
     (cases array-morphism arr
       (concrete-array (data _ _ _ dt _ _) body ...)
       (else (error who "operand not concrete" arr))))))

(define (activation-unary-executor op combiner in-A size dtype)
  (lambda (deref buf)
    (with-array-data 'activation-unary (deref in-A) (data src-dtype)
      (execute-activation-unary-compute op combiner data src-dtype buf size dtype))))

(define (flat-unary-executor combiner in-A size dtype)
  (lambda (deref buf)
    (with-array-data 'flat-unary (deref in-A) (data src-dtype)
      (execute-flat-unary-compute combiner data src-dtype buf size dtype))))

(define (activation-binary-executor op combiner in-A in-B size dtype)
  (lambda (deref buf)
    (with-array-data 'activation-binary (deref in-A) (data1 dt1)
      (with-array-data 'activation-binary (deref in-B) (data2 dt2)
        (execute-activation-binary-compute op combiner data1 dt1 data2 dt2 buf size dtype)))))

(define (flat-binary-executor combiner in-A in-B size dtype)
  (lambda (deref buf)
    (with-array-data 'flat-binary (deref in-A) (data1 dt1)
      (with-array-data 'flat-binary (deref in-B) (data2 dt2)
        (execute-flat-binary-compute combiner data1 dt1 data2 dt2 buf size dtype)))))

;; plain-op is add, sub, mul or div when the body is that op alone, which
;; allows the active backend's broadcast kernel; otherwise #f.
(define (bias-broadcast-executor plain-op combiner in-A in-B size N dtype)
  (lambda (deref buf)
    (with-array-data 'bias-broadcast (deref in-A) (data1 dt1)
      (with-array-data 'bias-broadcast (deref in-B) (data2 dt2)
        (let ((kernel (and plain-op
                           (memq dtype '(f32 f64))
                           (lookup-broadcast-kernel (active-activation-backend) plain-op dtype))))
          (if (and kernel (> N 0) (> size 0))
              (kernel (quotient size N) N data1 0 data2 2 buf)
              (execute-flat-bias-broadcast-compute combiner data1 data2 buf size N dtype)))))))

(define (flat-ternary-executor combiner in-A in-B in-C size dtype)
  (lambda (deref buf)
    (with-array-data 'flat-ternary (deref in-A) (a dta)
      (with-array-data 'flat-ternary (deref in-B) (b dtb)
        (with-array-data 'flat-ternary (deref in-C) (c dtc)
          (case dtype
            ((f32) (do ((i 0 (fx+ i 1))) ((fx= i size))
                     (f32vector-set! buf i (exact->inexact
                                            (combiner (f32vector-ref a i)
                                                      (f32vector-ref b i)
                                                      (f32vector-ref c i))))))
            (else (do ((i 0 (fx+ i 1))) ((fx= i size))
                    (f64vector-set! buf i (exact->inexact
                                           (combiner (f64vector-ref a i)
                                                     (f64vector-ref b i)
                                                     (f64vector-ref c i))))))))))))

;; op-kind is add, sub, mul, div or pow for a body that is that op alone,
;; and generic otherwise.  An empty output has nothing to compute.
(define (flat-broadcast-executor op-kind combiner in-A in-B space steps dtype)
  (let* ((rank (vector-length space))
         (N    (vector-ref space (- rank 1)))
         (size (shape-size space))
         (M    (if (> N 0) (quotient size N) 0)))
    (lambda (deref buf)
      (unless (= size 0)
        (with-array-data 'flat-broadcast (deref in-A) (data1 dt1)
          (with-array-data 'flat-broadcast (deref in-B) (data2 dt2)
            (let ((kernel (and (memq op-kind '(add sub mul div))
                               (lookup-broadcast-kernel (active-activation-backend) op-kind dtype))))
              (if kernel
                  (kernel M N data1 (broadcast-mode steps 0 N) data2 (broadcast-mode steps 2 N) buf)
                  (execute-flat-broadcast-compute combiner op-kind data1 data2
                                                  buf M N steps dtype)))))))))

;;; ------------------------------------------------------------
;;; Backends
;;; ------------------------------------------------------------

;; Calls (k space ins out body fallback) for a single-output seg-map with a
;; fallback, and returns #f for every other segop.
(define (with-single-output-seg-map sg k)
  (cases segop sg
    (seg-map (space ins outs bodies fallback)
      (and fallback (= (length outs) 1)
           (k space ins (car outs) (car bodies) fallback)))
    (else #f)))

(define (all-flat? ins space) (every (lambda (o) (operand-flat? o space)) ins))

(define (registry-select sg)
  (with-single-output-seg-map sg
    (lambda (space ins out body fallback)
      (let ((dtype (segop-operand-dtype out))
            (size  (shape-size space))
            (refs  (map segop-operand-slot ins)))
        (cond
         ((and (= (length ins) 1)
               (single-op-body-op body 1)
               (activation-op-registered? (single-op-body-op body 1))
               (all-flat? ins space))
          (cons 'segop/activation-unary
                (activation-unary-executor (single-op-body-op body 1) fallback
                                           (car refs) size dtype)))
         ((and (= (length ins) 2)
               (single-op-body-op body 2)
               (binary-op-registered? (single-op-body-op body 2))
               (all-flat? ins space))
          (cons 'segop/activation-binary
                (activation-binary-executor (single-op-body-op body 2) fallback
                                            (car refs) (cadr refs) size dtype)))
         ;; The broadcast patterns below come after flat-binary in the
         ;; scheme backend, so two flat inputs are left to it.
         ((and (= (length ins) 2) (all-flat? ins space)) #f)
         ((not (memq (single-op-body-op body 2) '(add sub mul div))) #f)
         ((bias-operands? ins space)
          (cons 'segop/bias-broadcast
                (bias-broadcast-executor (single-op-body-op body 2) fallback
                                         (car refs) (cadr refs) size
                                         (vector-ref space (- (vector-length space) 1)) dtype)))
         ((operands-broadcast-steps ins space dtype)
          => (lambda (steps)
               (cons 'segop/flat-broadcast
                     (flat-broadcast-executor (single-op-body-op body 2) fallback
                                              (car refs) (cadr refs) space steps dtype))))
         (else #f))))))

(define (scheme-select sg)
  (with-single-output-seg-map sg
    (lambda (space ins out body fallback)
      (let ((dtype (segop-operand-dtype out))
            (size  (shape-size space))
            (refs  (map segop-operand-slot ins)))
        (cond
         ((and (= (length ins) 1) (all-flat? ins space))
          (cons 'segop/flat-unary (flat-unary-executor fallback (car refs) size dtype)))
         ((and (= (length ins) 2) (all-flat? ins space))
          (cons 'segop/flat-binary
                (flat-binary-executor fallback (car refs) (cadr refs) size dtype)))
         ((bias-operands? ins space)
          (cons 'segop/bias-broadcast
                (bias-broadcast-executor #f fallback (car refs) (cadr refs) size
                                         (vector-ref space (- (vector-length space) 1)) dtype)))
         ((ternary-operands? ins space dtype)
          (cons 'segop/flat-ternary
                (flat-ternary-executor fallback (car refs) (cadr refs) (caddr refs) size dtype)))
         ((operands-broadcast-steps ins space dtype)
          => (lambda (steps)
               (let ((op (single-op-body-op body 2)))
                 (cons 'segop/flat-broadcast
                       (flat-broadcast-executor (if (memq op '(add sub mul div pow)) op 'generic)
                                                fallback (car refs) (cadr refs)
                                                space steps dtype)))))
         (else #f))))))

(register-segop-backend! (make-segop-backend 'registry registry-select) 'last)
(register-segop-backend! (make-segop-backend 'scheme scheme-select) 'last)

(define (compile-elementwise-segop b combiner in-traces in-refs pool-idx shape strides dtype)
  "The ri-segop instruction of element-wise binding b, or #f when the
   binding cannot be lowered or no backend accepts its segop."
  (let ((sg (lower-elementwise-binding b combiner in-traces in-refs pool-idx shape strides dtype)))
    (and sg
         (let ((sel (select-segop-executor sg)))
           (and sel
                (ri-segop pool-idx shape strides dtype sg (cdr sel) (car sel)))))))
