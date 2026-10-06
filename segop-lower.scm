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

;; True when an operand of shape in-shape broadcasts to out-shape: aligned
;; on the right, each operand extent is 1 or equals the output's, and any
;; extra leading dimensions of the operand have extent 1.
(define (broadcastable? in-shape out-shape)
  (let* ((k (vector-length in-shape)) (r (vector-length out-shape)) (extra (- k r)))
    (let loop ((j 0))
      (or (= j k)
          (let ((n (vector-ref in-shape j)))
            (and (if (< j extra)
                     (= n 1)
                     (or (= n 1) (= n (vector-ref out-shape (- j extra)))))
                 (loop (+ j 1))))))))

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

(define (registry-elementwise-select sg)
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

(define (scheme-elementwise-select sg)
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

;;; ------------------------------------------------------------
;;; Reductions
;;; ------------------------------------------------------------

;; The combine spec that describes how execute-reduction-morphism
;; accumulates reduction rop on path (see reduction-path) for a source of
;; dtype src-dtype and a result of dtype out-dtype.  The 1d sum may use the
;; BLAS dot product, whose summation order the backend chooses.  The fast
;; paths compare (max and min replace only by a greater or smaller value);
;; the general path applies the Scheme reducer and keeps the running value
;; in the output, so it is rounded to the output dtype after every step.
(define (reduction-combine-spec rop path src-dtype out-dtype)
  (let ((sum? (memq rop '(sum mean))))
    (case path
      ((1d)
       (if sum?
           (make-combine-spec 'add 0.0 'f64 'unspecified)
           (make-combine-spec rop (reduction-neutral rop 'f64) 'f64)))
      ((axis0)
       (if sum?
           (make-combine-spec 'add 0.0 src-dtype)
           (make-combine-spec rop (reduction-neutral rop src-dtype) src-dtype)))
      ((axis1)
       (if sum?
           (make-combine-spec 'add 0.0 'f64)
           (make-combine-spec rop (reduction-neutral rop 'f64) 'f64)))
      (else
       (make-combine-spec (case rop
                            ((sum mean) 'add)
                            ((prod)     'mul)
                            (else       (list 'opaque (get-reduction-function rop))))
                          (reduction-neutral rop out-dtype)
                          out-dtype)))))

(define (combine-specs-equal? a b)
  (and (equal? (combine-spec-op a) (combine-spec-op b))
       (eqv? (combine-spec-neutral a) (combine-spec-neutral b))
       (eq? (combine-spec-acc-dtype a) (combine-spec-acc-dtype b))
       (eq? (combine-spec-order a) (combine-spec-order b))))

(define (lower-reduction-binding rop axes keepdims? src-tr in-ref pool-idx shape strides dtype)
  "The seg-red of reduction rop over axes (with keepdims?) of the trace-time
   source array src-tr, found at in-ref at replay, writing pool slot
   pool-idx of the given shape, strides and dtype.  The space lists the
   kept axes, then the reduced axes in increasing order.  Its combine spec
   is the one of the loop execute-reduction-morphism takes for this layout."
  (cases array-morphism src-tr
    (concrete-array (_ s st off sdt _ _)
      (let* ((rank  (vector-length s))
             (red   (filter (lambda (k) (memv k axes)) (iota rank)))
             (par   (filter (lambda (k) (not (memv k axes))) (iota rank)))
             (pos   (map cons (append par red) (iota rank)))
             (dim-of (lambda (k) `(dim ,(cdr (assv k pos)))))
             (space (list->vector (map (lambda (k) (vector-ref s k)) (append par red))))
             (in-map (make-index-map (list->vector (map dim-of (iota rank))) #f 0.0 #f))
             (out-map (if keepdims?
                          (make-index-map (list->vector
                                           (map (lambda (k) (if (memv k red) 0 (dim-of k)))
                                                (iota rank)))
                                          #f 0.0 #f)
                          (identity-index-map (length par))))
             (count (fold (lambda (k acc) (* acc (vector-ref s k))) 1 red))
             (path  (reduction-path rop s st off sdt red)))
        (make-seg-red space (length red)
                      (list (make-segop-operand in-ref sdt s st off in-map))
                      (list (make-red-result
                             (make-segop-operand pool-idx dtype shape strides 0 out-map)
                             '(in 0)
                             (reduction-combine-spec rop path sdt dtype)
                             (and (eq? rop 'mean) `(/ (acc) ,count)))))))
    (else (error "lower-reduction-binding: source not concrete" src-tr))))

(define (standard-reduction sg)
  "The parameters of execute-reduction-morphism that compute seg-red sg, as
   a list (rop axes keepdims? reducer src-operand path), or #f when sg is not
   in the form lower-reduction-binding produces, combine spec included."
  (cases segop sg
    (seg-red (space nred ins results map-outs fallback)
      (and
       (= (length ins) 1) (= (length results) 1) (null? map-outs)
       (let* ((src  (car ins))
              (res  (car results))
              (out  (red-result-out res))
              (rank (vector-length space))
              (npar (- rank nred))
              (exprs (index-map-exprs (segop-operand-map src)))
              (dims (and exprs
                         (= (vector-length exprs) rank)
                         (every (lambda (e) (and (pair? e) (eq? (car e) 'dim))) (vector->list exprs))
                         (map cadr (vector->list exprs)))))
         (and
          dims
          (equal? (red-result-body res) '(in 0))
          ;; kept axes in order on dims 0..npar-1, reduced axes in order after
          (equal? (filter (lambda (d) (< d npar)) dims) (iota npar))
          (equal? (filter (lambda (d) (>= d npar)) dims) (iota nred npar))
          (let* ((red  (filter (lambda (k) (>= (list-ref dims k) npar)) (iota rank)))
                 (oexprs (index-map-exprs (segop-operand-map out)))
                 (keepdims?
                  (cond ((and oexprs (= (vector-length oexprs) npar)
                              (equal? (vector->list oexprs) (map (lambda (j) `(dim ,j)) (iota npar))))
                         #f)
                        ((and oexprs (= (vector-length oexprs) rank)
                              (equal? (vector->list oexprs)
                                      (map (lambda (k) (if (memv k red) 0 `(dim ,(list-ref dims k))))
                                           (iota rank))))
                         #t)
                        (else 'no)))
                 (spec (red-result-combine res))
                 (fin  (red-result-finalize res))
                 (count (fold (lambda (k acc) (* acc (vector-ref (segop-operand-shape src) k))) 1 red))
                 (cop  (combine-spec-op spec))
                 (rop  (cond ((and (eq? cop 'add) (not fin)) 'sum)
                             ((and (eq? cop 'add) (equal? fin `(/ (acc) ,count))) 'mean)
                             ((and (eq? cop 'mul) (not fin)) 'prod)
                             (fin #f)
                             ((memq cop '(max min)) cop)
                             ((and (pair? cop) (eq? (cadr cop) max)) 'max)
                             ((and (pair? cop) (eq? (cadr cop) min)) 'min)
                             (else #f))))
            (and (not (eq? keepdims? 'no))
                 rop
                 (let ((path (reduction-path rop (segop-operand-shape src) (segop-operand-strides src)
                                             (segop-operand-offset src) (segop-operand-dtype src) red)))
                   (and (combine-specs-equal?
                         spec (reduction-combine-spec rop path (segop-operand-dtype src)
                                                      (segop-operand-dtype out)))
                        (list rop red keepdims? (get-reduction-function rop) src path)))))))))
    (else #f)))

(define (reduction-executor rop axes keepdims? reducer src out)
  (let ((in-ref    (segop-operand-slot src))
        (src-dtype (segop-operand-dtype src))
        (out-shape (segop-operand-shape out))
        (out-dtype (segop-operand-dtype out)))
    (lambda (deref buf)
      (cases array-morphism (deref in-ref)
        (concrete-array (data s st off _ _ _)
          (execute-reduction-morphism rop buf out-shape data s st off
                                      axes reducer keepdims? out-dtype src-dtype))
        (else (error "reduction: source not concrete"))))))

;; (tag . executor) for a standard reduction whose path is one of paths.
(define (select-standard-reduction sg paths)
  (let ((r (standard-reduction sg)))
    (and r
         (memq (list-ref r 5) paths)
         (let ((out (red-result-out (car (cases segop sg
                                            (seg-red (space nred ins results map-outs fallback) results)
                                            (else '()))))))
           (cons (case (list-ref r 5)
                   ((1d) 'segop/reduce-1d)
                   ((axis0) 'segop/reduce-axis0)
                   ((axis1) 'segop/reduce-axis1)
                   (else 'segop/reduce-general))
                 (reduction-executor (car r) (cadr r) (caddr r) (cadddr r) (list-ref r 4) out))))))

;;; ------------------------------------------------------------
;;; Affine gathers and host index functions
;;; ------------------------------------------------------------

(define (lower-affine-binding fn src-tr in-ref pool-idx shape strides dtype)
  "The seg-map that materializes the affine index function fn applied to the
   trace-time source src-tr.  A reshape (identity-fn) iterates over the
   source's shape and writes the output buffer, viewed with that shape, in
   row-major order; other affine functions iterate over the output shape."
  (cases array-morphism src-tr
    (concrete-array (_ s st off sdt _ _)
      (let ((reshape? (cases affine-index-fn fn (identity-fn () #t) (else #f))))
        (if reshape?
            (make-seg-map s
                          (list (make-segop-operand in-ref sdt s st off
                                                    (identity-index-map (vector-length s))))
                          (list (make-segop-operand pool-idx dtype s (compute-strides s) 0
                                                    (identity-index-map (vector-length s))))
                          '((in 0)))
            (make-seg-map shape
                          (list (make-segop-operand in-ref sdt s st off
                                                    (affine-index-fn->index-map fn shape s)))
                          (list (make-segop-operand pool-idx dtype shape strides 0
                                                    (identity-index-map (vector-length shape))))
                          '((in 0))))))
    (else (error "lower-affine-binding: source not concrete" src-tr))))

;; A placeholder map for the operands of a seg-lib, whose access pattern the
;; library operation defines.
(define seg-lib-operand-map (opaque-index-map (lambda (coords) coords)))

(define (lower-host-index-fn fn in-traces in-refs pool-idx shape strides dtype)
  "A seg-lib host-index-fn that evaluates index function fn on the host
   with execute-index-fn."
  (make-seg-lib 'host-index-fn
                (map (lambda (tr ref)
                       (cases array-morphism tr
                         (concrete-array (_ s st off dt _ _)
                           (make-segop-operand ref dt s st off seg-lib-operand-map))
                         (else (error "lower-host-index-fn: input not concrete" tr))))
                     in-traces in-refs)
                (list (make-segop-operand pool-idx dtype shape strides 0 seg-lib-operand-map))
                `((index-fn . ,fn))))

(define (host-index-fn-select sg)
  (cases segop sg
    (seg-lib (name ins outs attrs)
      (and (eq? name 'host-index-fn)
           (let ((fn    (cdr (assq 'index-fn attrs)))
                 (refs  (map segop-operand-slot ins))
                 (shape (segop-operand-shape (car outs)))
                 (dtype (segop-operand-dtype (car outs))))
             (cons 'segop/host-index-fn
                   (lambda (deref buf)
                     (execute-index-fn fn buf shape (map deref refs) dtype))))))
    (else #f)))

;;; ------------------------------------------------------------
;;; Strided loops
;;; ------------------------------------------------------------

;; Calls (proc offs) at every point of space in row-major order, where offs
;; is a vector holding, for each linear form (base . coefs) of linears, the
;; flat offset at that point.  offs is updated in place; no storage is
;; allocated per point.
(define (for-each-strided-point space linears proc)
  (let* ((rank  (vector-length space))
         (n     (length linears))
         (coefs (list->vector (map cdr linears)))
         (offs  (list->vector (map car linears)))
         (idx   (make-vector rank 0)))
    (unless (let loop ((d 0)) (and (< d rank) (or (fx= 0 (vector-ref space d)) (loop (+ d 1)))))
      (if (fx= rank 0)
          (proc offs)
          (let ((last (fx- rank 1)))
            (let outer ()
              ;; innermost dimension
              (let ((ext (vector-ref space last)))
                (do ((i 0 (fx+ i 1))) ((fx= i ext))
                  (proc offs)
                  (do ((k 0 (fx+ k 1))) ((fx= k n))
                    (vector-set! offs k (fx+ (vector-ref offs k)
                                             (vector-ref (vector-ref coefs k) last)))))
                ;; rewind the innermost dimension
                (do ((k 0 (fx+ k 1))) ((fx= k n))
                  (vector-set! offs k (fx- (vector-ref offs k)
                                           (fx* ext (vector-ref (vector-ref coefs k) last))))))
              ;; carry into the outer dimensions
              (let carry ((d (fx- last 1)))
                (when (fx>= d 0)
                  (let ((v (fx+ 1 (vector-ref idx d))))
                    (do ((k 0 (fx+ k 1))) ((fx= k n))
                      (vector-set! offs k (fx+ (vector-ref offs k)
                                               (vector-ref (vector-ref coefs k) d))))
                    (if (fx< v (vector-ref space d))
                        (begin (vector-set! idx d v) (outer))
                        (begin
                          (vector-set! idx d 0)
                          (do ((k 0 (fx+ k 1))) ((fx= k n))
                            (vector-set! offs k (fx- (vector-ref offs k)
                                                     (fx* v (vector-ref (vector-ref coefs k) d)))))
                          (carry (fx- d 1)))))))))))))

;; The procedure storing a value v at offset i of an output of dtype dtype,
;; converting it as execute-compute-morphism does.
(define (strided-store dtype)
  (case dtype
    ((f32) (lambda (buf i v) (f32vector-set! buf i (exact->inexact v))))
    ((f64) (lambda (buf i v) (f64vector-set! buf i (exact->inexact v))))
    ((s32) (lambda (buf i v) (s32vector-set! buf i (inexact->exact (truncate v)))))
    ((s64) (lambda (buf i v) (s64vector-set! buf i (inexact->exact (truncate v)))))
    (else #f)))

(define (operand-data who arr)
  (cases array-morphism arr
    (concrete-array (data _ _ _ _ _ _) data)
    (else (error who "operand not concrete" arr))))

;; Executor of a single-output seg-map whose operands all have linear forms.
;; body-proc is the fallback combiner, or #f for the body (in 0).
(define (strided-map-executor space ins out body-proc)
  (let* ((rank   (vector-length space))
         (refs   (map segop-operand-slot ins))
         (dtypes (list->vector (map segop-operand-dtype ins)))
         (n      (length ins))
         (lins   (append (map (lambda (o) (segop-operand-linear o rank)) ins)
                         (list (segop-operand-linear out rank))))
         (store  (strided-store (segop-operand-dtype out)))
         (proc   (or body-proc (lambda (x) x))))
    (lambda (deref buf)
      (let ((datas (list->vector (map (lambda (r) (operand-data 'strided-map (deref r))) refs))))
        (define (arg offs k) (typed-vector-ref (vector-ref datas k) (vector-ref dtypes k)
                                               (vector-ref offs k)))
        (for-each-strided-point
         space lins
         (case n
           ((1) (lambda (offs) (store buf (vector-ref offs 1) (proc (arg offs 0)))))
           ((2) (lambda (offs) (store buf (vector-ref offs 2) (proc (arg offs 0) (arg offs 1)))))
           ((3) (lambda (offs) (store buf (vector-ref offs 3)
                                      (proc (arg offs 0) (arg offs 1) (arg offs 2)))))
           (else (lambda (offs)
                   (store buf (vector-ref offs n)
                          (apply proc (map (lambda (k) (arg offs k)) (iota n))))))))))))

;; Executor of a gather (body (in 0)) into a row-major output, through the
;; active backend's copy kernel when it has one for the dtype.
(define (gather-executor space in out)
  (let* ((rank   (vector-length space))
         (lin    (segop-operand-linear in rank))
         (dtype  (segop-operand-dtype out))
         (loop   (strided-map-executor space (list in) out #f))
         (ref    (segop-operand-slot in)))
    (lambda (deref buf)
      (let ((kernel (lookup-copy-kernel (active-activation-backend) dtype)))
        (if kernel
            (kernel (operand-data 'gather (deref ref)) (car lin) space (cdr lin) buf)
            (loop deref buf))))))

;; True when every operand of a single-output seg-map has a linear form.
(define (all-linear? ops rank)
  (every (lambda (o) (segop-operand-linear o rank)) ops))

(define (gather-body? body ins fallback)
  (and (equal? body '(in 0)) (= (length ins) 1) (not fallback)))

(define (registry-gather-select sg)
  (cases segop sg
    (seg-map (space ins outs bodies fallback)
      (let ((rank (vector-length space)))
        (and (= (length outs) 1)
             (gather-body? (car bodies) ins fallback)
             (<= rank 4)
             (memq (segop-operand-dtype (car outs)) '(f32 f64))
             (eq? (segop-operand-dtype (car ins)) (segop-operand-dtype (car outs)))
             (all-linear? (append ins outs) rank)
             (equal? (segop-operand-linear (car outs) rank) (cons 0 (compute-strides space)))
             (every (lambda (c) (>= c 0)) (vector->list (cdr (segop-operand-linear (car ins) rank))))
             (cons 'segop/gather (gather-executor space (car ins) (car outs))))))
    (else #f)))

(define (strided-select sg)
  (cases segop sg
    (seg-map (space ins outs bodies fallback)
      (let ((rank (vector-length space)))
        (and (= (length outs) 1)
             (strided-store (segop-operand-dtype (car outs)))
             (all-linear? (append ins outs) rank)
             (cond ((gather-body? (car bodies) ins fallback)
                    (cons 'segop/gather (strided-map-executor space ins (car outs) #f)))
                   (fallback
                    (cons 'segop/strided-map (strided-map-executor space ins (car outs) fallback)))
                   (else #f)))))
    (else #f)))

;; The last resort: evaluate a single-output seg-map or a single-result
;; seg-red with the reference interpreter.
(define (interpreted-select sg)
  (let ((outs (segop-outputs sg)))
    (and (memq (segop-kind sg) '(seg-map seg-red))
         (= (length outs) 1)
         (let ((refs (map segop-operand-slot (segop-inputs sg))))
           (cons 'segop/interpreted
                 (lambda (deref buf)
                   (segop-interpret sg (map (lambda (r) (operand-data 'interpreted (deref r))) refs)
                                    (list buf))))))))

;;; ------------------------------------------------------------
;;; Backend registration and instruction compilation
;;; ------------------------------------------------------------

(define (registry-select sg)
  (or (registry-elementwise-select sg)
      (select-standard-reduction sg '(axis0 axis1))
      (registry-gather-select sg)))

(define (scheme-select sg)
  (or (scheme-elementwise-select sg)
      (select-standard-reduction sg '(1d axis0 axis1 general))
      (strided-select sg)
      (host-index-fn-select sg)
      (interpreted-select sg)))

(register-segop-backend! (make-segop-backend 'registry registry-select) 'last)
(register-segop-backend! (make-segop-backend 'scheme scheme-select) 'last)

;; The ri-segop of sg, or #f when no backend accepts it.
(define (segop-instruction sg pool-idx shape strides dtype)
  (and sg
       (let ((sel (select-segop-executor sg)))
         (and sel (ri-segop pool-idx shape strides dtype sg (cdr sel) (car sel))))))

(define (compile-elementwise-segop b combiner in-traces in-refs pool-idx shape strides dtype)
  "The ri-segop instruction of element-wise binding b, or #f when the
   binding cannot be lowered or no backend accepts its segop."
  (segop-instruction
   (lower-elementwise-binding b combiner in-traces in-refs pool-idx shape strides dtype)
   pool-idx shape strides dtype))

(define (compile-reduction-segop rop axes keepdims? src-tr in-ref pool-idx shape strides dtype)
  "The ri-segop instruction of a reduction binding."
  (or (segop-instruction
       (lower-reduction-binding rop axes keepdims? src-tr in-ref pool-idx shape strides dtype)
       pool-idx shape strides dtype)
      (error "compile-reduction-segop: no backend accepts the reduction" rop axes)))

(define (compile-affine-segop fn src-tr in-ref pool-idx shape strides dtype)
  "The ri-segop instruction materializing affine index function fn, or #f."
  (segop-instruction (lower-affine-binding fn src-tr in-ref pool-idx shape strides dtype)
                     pool-idx shape strides dtype))

(define (compile-host-segop fn in-traces in-refs pool-idx shape strides dtype)
  "The ri-segop instruction evaluating index function fn on the host."
  (or (segop-instruction (lower-host-index-fn fn in-traces in-refs pool-idx shape strides dtype)
                         pool-idx shape strides dtype)
      (error "compile-host-segop: no backend accepts the index function" fn)))
