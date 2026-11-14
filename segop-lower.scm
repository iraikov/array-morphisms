;;; segop-lower.scm
;;;
;;; Lowering of SSA bindings to segops, and the two built-in segop
;;; backends that turn them into replay executors.  Included into the
;;; array-morphisms-ssa module.  Element-wise bindings become seg-maps,
;;; reductions and matmuls seg-reduces, affine views gathers, and the
;;; convolution ops seg-maps and seg-reduces with guarded maps; other index
;;; functions run on the host through a seg-lib.
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
;;; backend takes the others.  A segop that matches no pattern runs in the
;;; strided loop or, as a last resort, the reference interpreter.
;;;
;;; The registry backend also takes the GEMMs (segop/gemm, gemm-strided and
;;; gemm-epilogue, see Contractions below) and the axis-0/1 reductions.
;;; Both backends take the convolutions (see Convolutions below).
;;;
;;; The registry backend is a catalogue backend (see segop.scm): it runs
;;; natively the bodies for which the active activation backend holds a
;;; fused kernel (fused-kernel below), and it answers the fusion pass's
;;; native-capability queries from the same table.  The scheme backend is
;;; a host backend.

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
          ((and (symbol? op)
                (scalar-op? op)
                (= (scalar-op-arity op) n)
                (eq? combiner (scalar-op-combiner op)))
           (cons op (input-body-args n)))
          (else `(opaque ,combiner ,@(input-body-args n))))))

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
;;; Fused bindings
;;; ------------------------------------------------------------
;;;
;;; A fused binding (op fused) computes several bindings in one seg-map or
;;; one seg-reduce.  Its template holds the iteration space, one index map per
;;; input (from the space to the input's index, without a layout), the
;;; outputs, the binding's own result first, and nred, the number of
;;; trailing dimensions of the space that are reduced (0 for a seg-map).
;;; A seg-reduce lists its reduced results before its map outputs.
;;;
;;; Layouts are attached when the template is lowered against arrays: the
;;; trace-time arrays when a plan is compiled, the input arrays when the
;;; trace realizes the binding.  Every output is a row-major buffer.

(define-record fusion-template space in-maps outs nred)

;; map takes the space to the output's index, or is #f for an output
;; written in row-major order.  red is (combine-spec . finalize) for a
;; reduced result, (gemm . finalize) for the product of a GEMM (see
;; Contractions below), or #f.
(define-record fused-out name shape dtype body combiner map red)

(define (fused-out-index-map o)
  (or (fused-out-map o) (identity-index-map (vector-length (fused-out-shape o)))))

(define (lower-fused-binding tpl in-arrs in-slots out-slots)
  "The seg-map or seg-reduce of fusion template tpl reading the concrete
   arrays in-arrs (for their layouts), found at in-slots, and writing
   out-slots, one slot per output.  A seg-map with a single output keeps
   its compiled body as the fallback."
  (let* ((space (fusion-template-space tpl))
         (outs  (fusion-template-outs tpl))
         (nred  (fusion-template-nred tpl))
         (ins   (map (lambda (arr slot m)
                       (cases array-morphism arr
                         (concrete-array (_ s st off dt _ _) (make-segop-operand slot dt s st off m))
                         (else (error "lower-fused-binding: input not concrete" arr))))
                     in-arrs in-slots (fusion-template-in-maps tpl)))
         (out-op (lambda (o slot)
                   (let ((shape (fused-out-shape o)))
                     (make-segop-operand slot (fused-out-dtype o) shape (compute-strides shape) 0
                                         (fused-out-index-map o))))))
    (cond
     ((= nred 0)
        (make-seg-map space ins (map out-op outs out-slots) (map fused-out-body outs)
                      (and (null? (cdr outs)) (fused-out-combiner (car outs)))))
     ((gemm-fused-out? (car outs))
      (lower-gemm-template tpl in-arrs in-slots (car out-slots)))
     (else
        (let loop ((os outs) (slots out-slots) (results '()))
          (if (and (pair? os) (fused-out-red (car os)))
              (loop (cdr os) (cdr slots)
                    (cons (make-red-result (out-op (car os) (car slots)) (fused-out-body (car os))
                                           (car (fused-out-red (car os))) (cdr (fused-out-red (car os))))
                          results))
              (begin
                (when (any fused-out-red os)
                  (error "lower-fused-binding: reduced results must come before map outputs"))
                (make-seg-reduce space nred ins (reverse results)
                                 (map (lambda (o slot) (cons (out-op o slot) (fused-out-body o)))
                                      os slots)))))))))

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
      ((1d rows-1d)
       (if sum?
           (make-combine-spec 'add 0.0 'f64 'unspecified)
           (make-combine-spec rop (reduction-neutral rop 'f64) 'f64)))
      ((axis0)
       (if sum?
           (make-combine-spec 'add 0.0 src-dtype)
           (make-combine-spec rop (reduction-neutral rop src-dtype) src-dtype)))
      ((axis1 rows)
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

;; The path whose combine spec describes a reduction with frame rank
;; frame-rank: the path of reduction-path when the reduction runs as an
;; unframed one, rows when its rows run as the axis1 rows of one array, and
;; rows-1d when each row runs the 1d code.
(define (framed-path rop shape strides offset dtype red frame-rank)
  (let ((plan (framed-reduction-plan rop shape strides dtype red frame-rank)))
    (if (eq? plan 'direct)
        (reduction-path rop shape strides offset dtype red)
        (car plan))))

(define (combine-specs-equal? a b)
  (and (equal? (combine-spec-op a) (combine-spec-op b))
       (eqv? (combine-spec-neutral a) (combine-spec-neutral b))
       (eq? (combine-spec-acc-dtype a) (combine-spec-acc-dtype b))
       (eq? (combine-spec-order a) (combine-spec-order b))))

(define (lower-reduction-binding rop axes keepdims? src-tr in-ref pool-idx shape strides dtype
                                 #!optional (frame-rank 0))
  "The seg-reduce of reduction rop over axes (with keepdims?) of the trace-time
   source array src-tr, found at in-ref at replay, writing pool slot
   pool-idx of the given shape, strides and dtype.  The space lists the
   kept axes, then the reduced axes in increasing order.  Its combine spec
   is the one of the loop execute-framed-reduction takes for this layout
   and frame rank.  Two cases over a row-major source are lowered as
   reductions of a 2-D view of it, which accumulate in the same order and
   dtype as the loop for the source itself and run on the fast paths:
     - a framed reduction whose rows run as the rows of one array, as the
       axis-1 reduction of the [R, N] view;
     - a sum or mean over the leading axes of a source of rank 3 or more,
       as the axis-0 reduction of the [P, rest] view, P the size of the
       reduced axes and rest that of the others (1 when every axis is
       reduced)."
  (cases array-morphism src-tr
    (concrete-array (data s st off sdt _ _)
     (let* ((plan  (framed-reduction-plan rop s st sdt (sort-axes axes) frame-rank))
            (flat? (and (= off 0) (equal? st (compute-strides s))))
            (nlead (and (= frame-rank 0) (memq rop '(sum mean)) (memq sdt '(f32 f64))
                        (eq? sdt dtype) (>= (vector-length s) 3) flat?
                        (equal? (sort-axes axes) (iota (length axes)))
                        (length axes))))
      (cond
       ((and (pair? plan) (eq? (car plan) 'rows) flat?)
        (let ((R (cadr plan)) (N (caddr plan)))
          (lower-reduction-binding rop '(1) #f
                                   (concrete-array data (vector R N) (vector N 1) 0 sdt -1 -1)
                                   in-ref pool-idx (vector R) (vector 1) dtype)))
       ((and nlead (> nlead 0))
        (let* ((dims (vector->list s))
               (P    (fold * 1 (take dims nlead)))
               (rest (fold * 1 (drop dims nlead))))
          (lower-reduction-binding rop '(0) #f
                                   (concrete-array data (vector P rest) (vector rest 1) 0 sdt -1 -1)
                                   in-ref pool-idx (vector rest) (vector 1) dtype)))
       (else
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
             (path  (framed-path rop s st off sdt red frame-rank)))
        (make-seg-reduce space (length red)
                         (list (make-segop-operand in-ref sdt s st off in-map))
                         (list (make-red-result
                                (make-segop-operand pool-idx dtype shape strides 0 out-map)
                                '(in 0)
                                (reduction-combine-spec rop path sdt dtype)
                                (and (eq? rop 'mean) `(/ (reduced) ,count))))))))))
    (else (error "lower-reduction-binding: source not concrete" src-tr))))

;; The axes in increasing order.
(define (sort-axes axes)
  (filter (lambda (k) (memv k axes)) (iota (+ 1 (fold max -1 axes)))))

(define (standard-reduction sg)
  "The parameters of execute-reduction-morphism that compute seg-reduce sg, as
   a list (rop axes keepdims? reducer src-operand path), or #f when sg is not
   in the form lower-reduction-binding produces, combine spec included."
  (cases segop sg
    (seg-reduce (space nred ins results map-outs fallback)
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
                             ((and (eq? cop 'add) (equal? fin `(/ (reduced) ,count))) 'mean)
                             ((and (eq? cop 'mul) (not fin)) 'prod)
                             (fin #f)
                             ((memq cop '(max min)) cop)
                             ((and (pair? cop) (eq? (cadr cop) max)) 'max)
                             ((and (pair? cop) (eq? (cadr cop) min)) 'min)
                             (else #f))))
            (and (not (eq? keepdims? 'no))
                 rop
                 (let* ((sdt  (segop-operand-dtype src))
                        (odt  (segop-operand-dtype out))
                        (path (reduction-path rop (segop-operand-shape src) (segop-operand-strides src)
                                              (segop-operand-offset src) sdt red))
                        (spec-of (lambda (p) (reduction-combine-spec rop p sdt odt)))
                        (rows? (and (= nred 1) (= (car red) (- rank 1)) (> rank 1)
                                    (memq sdt '(f32 f64)) (eq? sdt odt)
                                    (operand-row-major? src))))
                   (cond
                    ((combine-specs-equal? spec (spec-of path))
                     (list rop red keepdims? (get-reduction-function rop) src path))
                    ((and rows? (memq rop '(sum mean))
                          (combine-specs-equal? spec (spec-of 'rows-1d)))
                     (list rop red keepdims? (get-reduction-function rop) src 'rows-1d))
                    (else #f)))))))))
    (else #f)))

;; The executor of a standard reduction.  The source's layout is the one
;; of the segop operand, which describes the replayed array or a view of
;; it with another shape.
(define (reduction-executor rop axes keepdims? reducer src out)
  (let ((in-ref    (segop-operand-slot src))
        (src-dtype (segop-operand-dtype src))
        (s         (segop-operand-shape src))
        (st        (segop-operand-strides src))
        (off       (segop-operand-offset src))
        (out-shape (segop-operand-shape out))
        (out-dtype (segop-operand-dtype out)))
    (lambda (deref buf)
      (cases array-morphism (deref in-ref)
        (concrete-array (data _ _ _ _ _ _)
          (execute-reduction-morphism rop buf out-shape data s st off
                                      axes reducer keepdims? out-dtype src-dtype))
        (else (error "reduction: source not concrete"))))))

;; The executor of a reduction over the rows of src (path rows or rows-1d).
(define (rows-reduction-executor kind rop reducer src out)
  (let ((in-ref    (segop-operand-slot src))
        (src-dtype (segop-operand-dtype src))
        (out-dtype (segop-operand-dtype out)))
    (lambda (deref buf)
      (cases array-morphism (deref in-ref)
        (concrete-array (data s st off _ _ _)
          (execute-rows-reduction kind rop buf data s st off reducer out-dtype src-dtype))
        (else (error "reduction: source not concrete"))))))

;; (tag . executor) for a standard reduction whose path is one of paths.
(define (select-standard-reduction sg paths)
  (let ((r (standard-reduction sg)))
    (and r
         (memq (list-ref r 5) paths)
         (let ((out (red-result-out (car (cases segop sg
                                            (seg-reduce (space nred ins results map-outs fallback) results)
                                            (else '()))))))
           (case (list-ref r 5)
             ((rows-1d)
              (cons 'segop/reduce-rows-1d
                    (rows-reduction-executor 'rows-1d (car r) (cadddr r) (list-ref r 4) out)))
             (else
              (cons (case (list-ref r 5)
                      ((1d) 'segop/reduce-1d)
                      ((axis0) 'segop/reduce-axis0)
                      ((axis1) 'segop/reduce-axis1)
                      (else 'segop/reduce-general))
                    (reduction-executor (car r) (cadr r) (caddr r) (cadddr r) (list-ref r 4)
                                        out))))))))

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

;; A seg-map with several outputs runs with its bodies compiled together,
;; so that a value its outputs share is computed once per point, when
;; every operand has a linear form.  Each output is stored with the
;; rounding of its dtype, as segop-interpret stores it.
(define (strided-multi-map-executor space ins outs bodies)
  (let* ((rank   (vector-length space))
         (n      (length ins))
         (m      (length outs))
         (refs   (map segop-operand-slot ins))
         (dtypes (list->vector (map segop-operand-dtype ins)))
         (odts   (list->vector (map segop-operand-dtype outs)))
         (lins   (map (lambda (o) (segop-operand-linear o rank)) (append ins outs)))
         (proc   (scalar-exprs->procedure bodies n)))
    (lambda (deref buf #!optional (extra-bufs '()))
      (let ((datas (list->vector (map (lambda (r) (operand-data 'strided-multi-map (deref r))) refs)))
            (bufs  (list->vector (cons buf extra-bufs))))
        (for-each-strided-point
         space lins
         (lambda (offs)
           (let ((vals (apply proc (map (lambda (k) (typed-vector-ref (vector-ref datas k) (vector-ref dtypes k)
                                                                      (vector-ref offs k)))
                                        (iota n)))))
             (do ((j 0 (+ j 1))) ((= j m))
               (typed-vector-set! (vector-ref bufs j) (vector-ref odts j) (vector-ref offs (+ n j))
                                  (vector-ref vals j))))))))))

(define (strided-multi-select sg)
  (cases segop sg
    (seg-map (space ins outs bodies fallback)
      (and (> (length outs) 1)
           (all-linear? (append ins outs) (vector-length space))
           (every (lambda (o) (memq (segop-operand-dtype o) '(f32 f64 s32 s64 u32 u64))) (append ins outs))
           (cons 'segop/strided-multi-map (strided-multi-map-executor space ins outs bodies))))
    (else #f)))

;;; ------------------------------------------------------------
;;; Strided reductions
;;; ------------------------------------------------------------
;;;
;;; A seg-reduce that is not a standard reduction, such as a fused one, runs
;;; on the host with compiled bodies when every operand has a linear form.
;;; It visits the points in the order segop-interpret does and rounds as it
;;; does, so the two agree bit for bit.  A map output whose body is the
;;; same expression as a result's body reuses the result's value.

(define (strided-red-executor space nred ins results map-outs)
  (let* ((rank    (vector-length space))
         (npar    (- rank nred))
         (n       (length ins))
         (refs    (map segop-operand-slot ins))
         (dtypes  (list->vector (map segop-operand-dtype ins)))
         (in-lins (map (lambda (o) (segop-operand-linear o rank)) ins))
         (red-space (subvector space npar rank))
         (par-space (subvector space 0 npar))
         ;; The linear forms of the inputs and map outputs over the reduced
         ;; dimensions, and over the parallel ones (for the offsets).
         (red-coefs (map (lambda (l) (subvector (cdr l) npar rank))
                         (append in-lins (map (lambda (mo) (segop-operand-linear (car mo) rank))
                                              map-outs))))
         (base-lins (append in-lins (map (lambda (mo) (segop-operand-linear (car mo) rank)) map-outs)
                            (map (lambda (r) (segop-operand-linear (red-result-out r) rank)) results)))
         (par-lins  (map (lambda (l) (cons (car l) (subvector (cdr l) 0 npar))) base-lins))
         (n-mo      (length map-outs))
         (procs     (list->vector
                     (map (lambda (r) (scalar-expr->procedure (red-result-body r) n)) results)))
         (specs     (list->vector (map red-result-combine results)))
         (fins      (list->vector
                     (map (lambda (r)
                            (and (red-result-finalize r)
                                 (scalar-expr->procedure (red-result-finalize r) n reduced?: #t)))
                          results)))
         (out-dtypes (list->vector (map (lambda (r) (segop-operand-dtype (red-result-out r))) results)))
         ;; For each map output, (j . #f) when it shares result j's body,
         ;; or (#f . its compiled body).
         (mo-src    (list->vector
                     (map (lambda (mo)
                            (let ((j (list-index (lambda (r) (eq? (red-result-body r) (cdr mo))) results)))
                              (if j (cons j #f) (cons #f (scalar-expr->procedure (cdr mo) n)))))
                          map-outs)))
         (mo-dtypes (list->vector (map (lambda (mo) (segop-operand-dtype (car mo))) map-outs)))
         (n-res     (length results)))
    (lambda (deref buf #!optional (extra-bufs '()))
      (let* ((datas (list->vector (map (lambda (r) (operand-data 'strided-red (deref r))) refs)))
             (bufs  (list->vector (cons buf extra-bufs)))
             (accs  (make-vector n-res #f))
             (blks  (make-vector n-res #f))
             (cnts  (make-vector n-res 0))
             (vals  (make-vector n-res #f)))
        (define (args offs) (map (lambda (k) (typed-vector-ref (vector-ref datas k) (vector-ref dtypes k)
                                                               (vector-ref offs k)))
                                 (iota n)))
        (for-each-strided-point
         par-space par-lins
         (lambda (poffs)
           (do ((j 0 (+ j 1))) ((= j n-res))
             (let ((spec (vector-ref specs j)))
               (vector-set! accs j (round-to-dtype (combine-spec-neutral spec)
                                                   (combine-spec-acc-dtype spec)))
               (vector-set! blks j (vector-ref accs j))
               (vector-set! cnts j 0)))
           (for-each-strided-point
            red-space
            (map (lambda (k c) (cons (vector-ref poffs k) c)) (iota (+ n n-mo)) red-coefs)
            (lambda (offs)
              (let ((xs (args offs)))
                (do ((j 0 (+ j 1))) ((= j n-res))
                  (let* ((spec (vector-ref specs j))
                         (v    (apply (vector-ref procs j) xs)))
                    (vector-set! vals j v)
                    (if (eq? (combine-spec-order spec) 'blocked)
                        (begin
                          (vector-set! blks j (combine-step spec (vector-ref blks j) v))
                          (vector-set! cnts j (+ 1 (vector-ref cnts j)))
                          (when (= (vector-ref cnts j) (combine-spec-block-size spec))
                            (vector-set! accs j (combine-step spec (vector-ref accs j) (vector-ref blks j)))
                            (vector-set! blks j (round-to-dtype (combine-spec-neutral spec)
                                                                (combine-spec-acc-dtype spec)))
                            (vector-set! cnts j 0)))
                        (vector-set! accs j (combine-step spec (vector-ref accs j) v)))))
                (do ((m 0 (+ m 1))) ((= m n-mo))
                  (let* ((src (vector-ref mo-src m))
                         (v   (if (car src) (vector-ref vals (car src)) (apply (cdr src) xs))))
                    (typed-vector-set! (vector-ref bufs (+ n-res m)) (vector-ref mo-dtypes m)
                                       (vector-ref offs (+ n m)) v))))))
           (do ((j 0 (+ j 1))) ((= j n-res))
             (let ((spec (vector-ref specs j)))
               (when (and (eq? (combine-spec-order spec) 'blocked) (> (vector-ref cnts j) 0))
                 (vector-set! accs j (combine-step spec (vector-ref accs j) (vector-ref blks j))))
               (let* ((acc (vector-ref accs j))
                      (fin (vector-ref fins j))
                      (v   (if fin (apply fin (append (args poffs) (list acc))) acc)))
                 (typed-vector-set! (vector-ref bufs j) (vector-ref out-dtypes j)
                                    (vector-ref poffs (+ n n-mo j)) v))))))))))

(define (strided-red-select sg)
  (cases segop sg
    (seg-reduce (space nred ins results map-outs fallback)
      (let ((rank (vector-length space)))
        (and (all-linear? (append ins (map red-result-out results) (map car map-outs)) rank)
             (every (lambda (o) (memq (segop-operand-dtype o) '(f32 f64 s32 s64 u32 u64)))
                    (append ins (map red-result-out results) (map car map-outs)))
             (cons 'segop/strided-red (strided-red-executor space nred ins results map-outs)))))
    (else #f)))

;;; ------------------------------------------------------------
;;; Scans
;;; ------------------------------------------------------------
;;;
;;; A scan along axis a of a source of rank R becomes a seg-scan whose
;;; space lists the other axes in increasing order, then axis a.  The
;;; source is read, and the row-major output written, through the map
;;; that sends each space dimension back to its axis; a reversed scan
;;; reads and writes the scanned axis at n - 1 - r, so its maps have a
;;; negative coefficient there.  The body is (in 0); an init array is input
;;; 1, read at the parallel point, and is the seed.  The combine spec has
;;; the result's dtype as accumulator dtype and the sequential order, which
;;; is the loop of execute-scan-morphism.

(define (scan-combine-op op)
  (case op ((sum) 'add) ((prod) 'mul) ((max) 'max) ((min) 'min)
    (else (error "scan: unknown op" op))))

(define (lower-scan-binding meta in-traces in-refs pool-idx shape strides dtype)
  "The seg-scan of a scan binding with meta meta (scan-op, axis,
   exclusive?, reverse?, swapped?), whose source (and init, when there is
   one) had the trace-time arrays in-traces and are found at in-refs,
   writing pool slot pool-idx of the given shape, strides and dtype."
  (let* ((get  (lambda (k) (cdr (assq k meta))))
         (op   (get 'scan-op))
         (ax   (get 'axis))
         (rev? (get 'reverse?))
         (rank (vector-length shape))
         (n    (vector-ref shape ax))
         (npar (- rank 1))
         (par  (filter (lambda (k) (not (= k ax))) (iota rank)))
         (space (list->vector (append (map (lambda (k) (vector-ref shape k)) par) (list n))))
         (scan-expr (if rev? `(+ ,(- n 1) (* -1 (dim ,npar))) `(dim ,npar)))
         (axis-map (make-index-map
                    (list->vector (map (lambda (k)
                                         (if (= k ax) scan-expr `(dim ,(list-index (lambda (j) (= j k)) par))))
                                       (iota rank)))
                    #f 0.0 #f))
         (operand (lambda (tr ref m)
                    (cases array-morphism tr
                      (concrete-array (_ s st off dt _ _) (make-segop-operand ref dt s st off m))
                      (else (error "lower-scan-binding: input not concrete" tr)))))
         (src  (operand (car in-traces) (car in-refs) axis-map))
         (init (and (pair? (cdr in-traces))
                    (operand (cadr in-traces) (cadr in-refs) (identity-index-map npar))))
         (spec (make-combine-spec (scan-combine-op op) (reduction-neutral op dtype) dtype)))
    (make-seg-scan space
                   (if init (list src init) (list src))
                   (list (make-scan-result (make-segop-operand pool-idx dtype shape strides 0 axis-map)
                                           '(in 0) spec (and init '(in 1))
                                           (get 'exclusive?) (get 'swapped?))))))

(define (compile-scan-segop meta in-traces in-refs pool-idx shape strides dtype)
  "The ri-segop instruction of a scan binding (see lower-scan-binding)."
  (or (segop-instruction (lower-scan-binding meta in-traces in-refs pool-idx shape strides dtype)
                         pool-idx shape strides dtype)
      (error "compile-scan-segop: no backend accepts the scan" meta)))

;; The step (acc x) -> value of combine op cop, before rounding, the
;; arguments trading places when swapped?.  It is the step of combine-step.
(define (scan-step cop swapped?)
  (let ((f (case cop
             ((add) scalar-add)
             ((mul) *)
             ((max) (lambda (acc x) (if (> x acc) x acc)))
             ((min) (lambda (acc x) (if (< x acc) x acc)))
             (else (lambda (acc x) ((cadr cop) acc x))))))
    (if swapped? (lambda (acc x) (f x acc)) f)))

;; The parts of a seg-scan the executors below share: the parallel space,
;; the linear forms of every operand over it (inputs, then the output), the
;; step along the scanned dimension of each, and the scan length.
(define (scan-layout space ops)
  (let* ((rank (vector-length space))
         (npar (- rank 1))
         (lins (map (lambda (o) (segop-operand-linear o rank)) ops)))
    (values (subvector space 0 npar)
            (map (lambda (l) (cons (car l) (subvector (cdr l) 0 npar))) lins)
            (list->vector (map (lambda (l) (vector-ref (cdr l) npar)) lins))
            (vector-ref space npar))))

;; Executor of a seg-scan with one result, body (in 0), an optional seed
;; (in 1) and a table combine op, over f32 or f64 operands of one dtype.
(define (scan-executor space ins res)
  (let*-values (((out)  (scan-result-out res))
                ((spec) (scan-result-combine res))
                ((dt)   (segop-operand-dtype out))
                ((par-space lins steps n) (scan-layout space (append ins (list out))))
                ((seed?) (and (scan-result-seed res) #t))
                ((excl?) (scan-result-exclusive? res))
                ((step)  (scan-step (combine-spec-op spec) (scan-result-swapped? res)))
                ((neutral) (round-to-dtype (combine-spec-neutral spec) dt))
                ((x-ref) (car (map segop-operand-slot ins)))
                ((s-ref) (and seed? (cadr (map segop-operand-slot ins))))
                ((xs)    (vector-ref steps 0))
                ((os)    (vector-ref steps (if seed? 2 1)))
                ((oi)    (if seed? 2 1)))
    (define-syntax scan-lines
      (syntax-rules ()
        ((_ vref vset! round x-data s-data buf)
         (for-each-strided-point
          par-space lins
          (lambda (offs)
            (let ((xb (vector-ref offs 0)) (ob (vector-ref offs oi)))
              (let loop ((r 0) (xo xb) (oo ob)
                         (acc (and seed? (vref s-data (vector-ref offs 1)))))
                (when (fx< r n)
                  (let ((x (vref x-data xo)))
                    (if excl?
                        (begin (vset! buf oo (or acc neutral))
                               (loop (fx+ r 1) (fx+ xo xs) (fx+ oo os)
                                     (if acc (round (step acc x)) x)))
                        (let ((v (if acc (round (step acc x)) x)))
                          (vset! buf oo v)
                          (loop (fx+ r 1) (fx+ xo xs) (fx+ oo os) v))))))))))))
    (lambda (deref buf #!optional (extra-bufs '()))
      (let ((x-data (operand-data 'scan (deref x-ref)))
            (s-data (and seed? (operand-data 'scan (deref s-ref)))))
        (if (eq? dt 'f32)
            (let ((cell (make-f32vector 1)))
              (define (round32 v) (f32vector-set! cell 0 v) (f32vector-ref cell 0))
              (scan-lines f32vector-ref f32vector-set! round32 x-data s-data buf))
            (scan-lines f64vector-ref f64vector-set! values x-data s-data buf))))))

(define (scan-select sg)
  (cases segop sg
    (seg-scan (space ins results fallback)
      (and (= (length results) 1)
           (let* ((res  (car results))
                  (out  (scan-result-out res))
                  (dt   (segop-operand-dtype out))
                  (spec (scan-result-combine res))
                  (rank (vector-length space)))
             (and (memq dt '(f32 f64))
                  (memq (combine-spec-op spec) '(add mul max min))
                  (eq? (combine-spec-acc-dtype spec) dt)
                  (memq (combine-spec-order spec) '(sequential unspecified))
                  (equal? (scan-result-body res) '(in 0))
                  (if (scan-result-seed res)
                      (and (= (length ins) 2) (equal? (scan-result-seed res) '(in 1)))
                      (= (length ins) 1))
                  (every (lambda (o) (eq? (segop-operand-dtype o) dt)) ins)
                  (all-linear? (append ins (list out)) rank)
                  (cons 'segop/scan (scan-executor space ins res))))))
    (else #f)))

;; Executor of any seg-scan in the sequential order whose operands all have
;; linear forms: bodies and seeds are compiled procedures.
(define (strided-scan-executor space ins results)
  (let*-values (((n-in) (length ins))
                ((outs) (map scan-result-out results))
                ((par-space lins steps n) (scan-layout space (append ins outs)))
                ((refs) (map segop-operand-slot ins))
                ((dtypes) (list->vector (map segop-operand-dtype ins)))
                ((res-v) (list->vector results))
                ((n-res) (length results))
                ((bodies) (list->vector (map (lambda (r) (scalar-expr->procedure (scan-result-body r) n-in))
                                             results)))
                ((seeds) (list->vector (map (lambda (r) (and (scan-result-seed r)
                                                             (scalar-expr->procedure (scan-result-seed r) n-in)))
                                            results)))
                ((steps-of) (list->vector
                             (map (lambda (r) (scan-step (combine-spec-op (scan-result-combine r))
                                                         (scan-result-swapped? r)))
                                  results)))
                ((odts) (list->vector (map segop-operand-dtype outs))))
    (lambda (deref buf #!optional (extra-bufs '()))
      (let* ((datas (list->vector (map (lambda (r) (operand-data 'strided-scan (deref r))) refs)))
             (bufs  (list->vector (cons buf extra-bufs)))
             (offs* (make-vector (+ n-in n-res) 0))
             (accs  (make-vector n-res #f)))
        (define (args offs)
          (map (lambda (k) (typed-vector-ref (vector-ref datas k) (vector-ref dtypes k) (vector-ref offs k)))
               (iota n-in)))
        (for-each-strided-point
         par-space lins
         (lambda (poffs)
           (do ((k 0 (+ k 1))) ((= k (+ n-in n-res))) (vector-set! offs* k (vector-ref poffs k)))
           (do ((j 0 (+ j 1))) ((= j n-res))
             (let* ((seed (vector-ref seeds j))
                    (dt (combine-spec-acc-dtype (scan-result-combine (vector-ref res-v j)))))
               (vector-set! accs j (and seed (round-to-dtype (apply seed (args poffs)) dt)))))
           (do ((r 0 (+ r 1))) ((= r n))
             (let ((xs (args offs*)))
               (do ((j 0 (+ j 1))) ((= j n-res))
                 (let* ((res  (vector-ref res-v j))
                        (spec (scan-result-combine res))
                        (dt   (combine-spec-acc-dtype spec))
                        (acc  (vector-ref accs j))
                        (v    (apply (vector-ref bodies j) xs))
                        (next (round-to-dtype (if acc ((vector-ref steps-of j) acc v) v) dt))
                        (o    (vector-ref offs* (+ n-in j))))
                   (typed-vector-set! (vector-ref bufs j) (vector-ref odts j) o
                                      (if (scan-result-exclusive? res)
                                          (or acc (round-to-dtype (combine-spec-neutral spec) dt))
                                          next))
                   (vector-set! accs j next))))
             (do ((k 0 (+ k 1))) ((= k (+ n-in n-res)))
               (vector-set! offs* k (+ (vector-ref offs* k) (vector-ref steps k)))))))))))

(define (strided-scan-select sg)
  (cases segop sg
    (seg-scan (space ins results fallback)
      (let ((rank (vector-length space)))
        (and (all-linear? (append ins (map scan-result-out results)) rank)
             (every (lambda (r) (memq (combine-spec-order (scan-result-combine r)) '(sequential unspecified)))
                    results)
             (every (lambda (o) (memq (segop-operand-dtype o) '(f32 f64 s32 s64 u32 u64)))
                    (append ins (map scan-result-out results)))
             (cons 'segop/strided-scan (strided-scan-executor space ins results)))))
    (else #f)))

;; The last resort: evaluate a seg-map, a seg-reduce or a seg-scan with the
;; reference interpreter.  The executor of a segop with several outputs
;; takes the buffers of the outputs after the first as a list.
(define (interpreted-select sg)
  (and (memq (segop-kind sg) '(seg-map seg-reduce seg-scan))
       (let ((refs (map segop-operand-slot (segop-inputs sg))))
         (cons 'segop/interpreted
               (lambda (deref buf #!optional (extra-bufs '()))
                 (segop-interpret sg (map (lambda (r) (operand-data 'interpreted (deref r))) refs)
                                  (cons buf extra-bufs)))))))

;;; ------------------------------------------------------------
;;; Contractions
;;; ------------------------------------------------------------
;;;
;;; A matmul A[M,K] x B[K,N] becomes a seg-reduce over the space (M N K) with
;;; K reduced.  A is read through #((dim 0) (dim 2)) and B through
;;; #((dim 2) (dim 1)), each with the layout of its trace-time array, so a
;;; transposed view is described by its strides.  The body is
;;; (mul (in 0) (in 1)) and the output is the row-major pool buffer.
;;;
;;; The combine spec follows the gemm-plan the product runs with.  A plan
;;; that calls a backend kernel sums in the order the backend chooses; a
;;; plan without one runs the Scheme loop, which sums over increasing k in
;;; double precision and rounds once when storing.
;;;
;;; A GEMM can absorb the element-wise bindings that consume its product
;;; (the epilogue): the finalize is the epilogue's body with its input 0
;;; replaced by (cast dtype (reduced)), the product as stored in the output,
;;; and its other inputs appended to the seg-reduce's inputs.  The seg-reduce's
;;; fallback is the epilogue's combiner, which the executor applies to the
;;; stored product in place.  The fusion pass makes such seg-reduces from
;;; fusion templates whose result has red (gemm . finalize).

(define gemm-A-exprs '#((dim 0) (dim 2)))
(define gemm-B-exprs '#((dim 2) (dim 1)))
(define gemm-out-exprs '#((dim 0) (dim 1)))
(define gemm-body '(mul (in 0) (in 1)))

;; A concrete array with the layout of operand o and no elements, for the
;; blas-exec procedures that decide from shapes, strides and dtypes alone.
(define (operand-layout-array o)
  (let ((dt (segop-operand-dtype o)))
    (concrete-array (allocate-typed-vector dt 0) (segop-operand-shape o)
                    (segop-operand-strides o) (segop-operand-offset o) dt -1 -1)))

;; gemm when the operands are row-major and contiguous (the plain GEMM
;; kernel), gemm-strided for other 2-D operands of one dtype, f32 or f64,
;; and #f when blas-exec cannot multiply them.
(define (gemm-plan-kind A B)
  (let ((a (operand-layout-array A)) (b (operand-layout-array B)))
    (cond ((blas-compatible-matmul? a b) 'gemm)
          ((matmul-concrete? a b) 'gemm-strided)
          (else #f))))

(define (prepare-gemm-plan kind A B)
  ((if (eq? kind 'gemm) prepare-blas-gemm/into prepare-blas-gemm-strided/into)
   (operand-layout-array A) (operand-layout-array B)))

(define (gemm-combine-spec plan dtype)
  (if (gemm-plan-uses-kernel? plan)
      (make-combine-spec 'add 0.0 dtype 'unspecified)
      (make-combine-spec 'add 0.0 'f64)))

(define (operand-exprs-are? o exprs)
  (equal? (index-map-exprs (segop-operand-map o)) exprs))

(define (lower-matmul-binding A-tr B-tr A-ref B-ref pool-idx shape strides dtype
                              #!optional (epilogue #f))
  "The seg-reduce of the matmul of the trace-time arrays A-tr and B-tr, found
   at A-ref and B-ref, writing pool slot pool-idx of the given shape,
   strides and dtype; #f when blas-exec cannot multiply them.  epilogue is
   #f or the single-output seg-map of an element-wise binding whose input 0
   is the product; the seg-reduce then computes that binding's output, in the
   epilogue's output slot."
  (let ((operand (lambda (tr ref exprs)
                   (cases array-morphism tr
                     (concrete-array (_ s st off dt _ _)
                       (make-segop-operand ref dt s st off (make-index-map exprs #f 0.0 #f)))
                     (else (error "lower-matmul-binding: operand not concrete" tr))))))
    (let* ((A    (operand A-tr A-ref gemm-A-exprs))
           (B    (operand B-tr B-ref gemm-B-exprs))
           (kind (gemm-plan-kind A B)))
      (and kind
           (eq? (segop-operand-dtype A) dtype)
           (let* ((sA    (segop-operand-shape A))
                  (sB    (segop-operand-shape B))
                  (space (vector (vector-ref sA 0) (vector-ref sB 1) (vector-ref sA 1)))
                  (plan  (prepare-gemm-plan kind A B))
                  (spec  (gemm-combine-spec plan dtype))
                  (out   (lambda (slot)
                           (make-segop-operand slot dtype shape strides 0
                                               (make-index-map gemm-out-exprs #f 0.0 #f)))))
             (if epilogue
                 (cases segop epilogue
                   (seg-map (e-space e-ins e-outs e-bodies e-fallback)
                     (make-seg-reduce space 1 (append (list A B) (cdr e-ins))
                                      (list (make-red-result
                                             (out (segop-operand-slot (car e-outs)))
                                             gemm-body spec
                                             (scalar-expr-substitute-inputs
                                              (car e-bodies)
                                              (lambda (k) (if (= k 0) `(cast ,dtype (reduced)) `(in ,(+ k 1)))))))
                                      '() e-fallback))
                   (else #f))
                 (make-seg-reduce space 1 (list A B)
                                  (list (make-red-result (out pool-idx) gemm-body spec #f)))))))))

(define (standard-gemm sg)
  "The GEMM that computes seg-reduce sg, as a list
   (A B out kind plan epilogue op), or #f when sg is not in the form
   lower-matmul-binding produces, combine spec included.  kind is gemm or
   gemm-strided and plan the gemm-plan for the operands; epilogue is #f,
   (unary . combiner), (bias bias-operand . combiner) for a row bias read
   along the columns, or (map operands . combiner) for any other inputs.
   op is the op of an epilogue that applies one op to the product and its
   other inputs, in that order, and #f otherwise."
  (cases segop sg
    (seg-reduce (space nred ins results map-outs fallback)
      (and
       (= (vector-length space) 3) (= nred 1) (= (length results) 1) (null? map-outs)
       (>= (length ins) 2)
       (let* ((A     (car ins))
              (B     (cadr ins))
              (res   (car results))
              (out   (red-result-out res))
              (dtype (segop-operand-dtype out))
              (fin   (red-result-finalize res))
              (M     (vector-ref space 0))
              (N     (vector-ref space 1))
              (K     (vector-ref space 2)))
         (and
          (equal? (red-result-body res) gemm-body)
          (operand-exprs-are? A gemm-A-exprs) (equal? (segop-operand-shape A) (vector M K))
          (operand-exprs-are? B gemm-B-exprs) (equal? (segop-operand-shape B) (vector K N))
          (operand-exprs-are? out gemm-out-exprs) (operand-flat? out (vector M N))
          (eq? (segop-operand-dtype A) dtype)
          (let ((epilogue
                 (cond ((not fin) (and (= (length ins) 2) 'none))
                       ((not fallback) #f)
                       ((= (length ins) 2) (cons 'unary fallback))
                       ((and (= (length ins) 3)
                             (let ((bias (caddr ins)))
                               (and (operand-exprs-are? bias '#((dim 1)))
                                    (operand-row-major? bias)
                                    (equal? (segop-operand-shape bias) (vector N))
                                    (eq? (segop-operand-dtype bias) dtype))))
                        (cons 'bias (cons (caddr ins) fallback)))
                       ((every (lambda (o) (segop-operand-linear o 2)) (cddr ins))
                        (cons 'map (cons (cddr ins) fallback)))
                       (else #f)))
                (kind (gemm-plan-kind A B)))
            (and epilogue kind
                 (let ((plan (prepare-gemm-plan kind A B)))
                   (and (combine-specs-equal? (red-result-combine res) (gemm-combine-spec plan dtype))
                        (list A B out kind plan (if (eq? epilogue 'none) #f epilogue)
                              (and fin (single-op-body-op (gemm-finalize->epilogue fin dtype)
                                                          (- (length ins) 1))))))))))))
    (else #f)))

;; The executor that applies combiner in place to the GEMM product in the
;; row-major buffer of out: each element x becomes (combiner x y ...), with
;; y ... the elements of the operands ins at its index.
(define (gemm-map-epilogue-executor out ins combiner)
  (let* ((space  (segop-operand-shape out))
         (dtype  (segop-operand-dtype out))
         (refs   (map segop-operand-slot ins))
         (dtypes (list->vector (map segop-operand-dtype ins)))
         (n      (length ins))
         (lins   (append (map (lambda (o) (segop-operand-linear o 2)) ins)
                         (list (segop-operand-linear out 2))))
         (store  (strided-store dtype)))
    (lambda (deref buf)
      (let ((datas (list->vector (map (lambda (r) (operand-data 'gemm-epilogue (deref r))) refs))))
        (define (arg offs k) (typed-vector-ref (vector-ref datas k) (vector-ref dtypes k)
                                               (vector-ref offs k)))
        (for-each-strided-point
         space lins
         (if (= n 1)
             (lambda (offs)
               (let ((i (vector-ref offs 1)))
                 (store buf i (combiner (typed-vector-ref buf dtype i) (arg offs 0)))))
             (lambda (offs)
               (let ((i (vector-ref offs n)))
                 (store buf i (apply combiner (typed-vector-ref buf dtype i)
                                     (map (lambda (k) (arg offs k)) (iota n))))))))))))

;; The registry routes GEMMs to the active BLAS backend through prepared
;; gemm-plans; a fused epilogue then runs in place on the stored product.
;; An epilogue that is one op on the product, or on the product and a row
;; bias, runs as the unfused op would: through the active activation
;; backend's unary or broadcast kernel for the op when it has one, and
;; through the flat unary and bias loops otherwise.  Other epilogues are
;; left to the backends after the registry: the scheme backend runs them
;; through a strided loop on the host (scheme-gemm-select), and a backend
;; placed between the two may run them as compiled code.
(define (registry-gemm-select sg)
  (let ((g (standard-gemm sg)))
    (and g
         (let* ((A-ref    (segop-operand-slot (list-ref g 0)))
                (B-ref    (segop-operand-slot (list-ref g 1)))
                (out      (list-ref g 2))
                (plan     (list-ref g 4))
                (epilogue (list-ref g 5))
                (op       (list-ref g 6))
                (dtype    (segop-operand-dtype out))
                (size     (shape-size (segop-operand-shape out)))
                (gemm!    (lambda (deref buf)
                            (execute-gemm-plan/into! plan (deref A-ref) (deref B-ref) buf))))
           (cond
            ((not epilogue)
             (cons (if (eq? (list-ref g 3) 'gemm) 'segop/gemm 'segop/gemm-strided) gemm!))
            ((eq? (car epilogue) 'map) #f)
            ((eq? (car epilogue) 'unary)
             (let ((combiner (cdr epilogue))
                   (act-op   (and op (activation-op-registered? op) op)))
               (cons 'segop/gemm-epilogue
                     (lambda (deref buf)
                       (gemm! deref buf)
                       (if act-op
                           (execute-activation-unary-compute act-op combiner buf dtype buf size dtype)
                           (execute-flat-unary-compute combiner buf dtype buf size dtype))))))
            (else
             (let ((bias-ref (segop-operand-slot (cadr epilogue)))
                   (combiner (cddr epilogue))
                   (plain-op (and (memq op '(add sub mul div)) op))
                   (N        (vector-ref (segop-operand-shape out) 1)))
               (cons 'segop/gemm-epilogue
                     (lambda (deref buf)
                       (gemm! deref buf)
                       (with-array-data 'gemm-epilogue (deref bias-ref) (bias-data bias-dtype)
                         (let ((kernel (and plain-op
                                            (lookup-broadcast-kernel (active-activation-backend)
                                                                     plain-op dtype))))
                           (if (and kernel (> N 0) (> size 0))
                               (kernel (quotient size N) N buf 0 bias-data 2 buf)
                               (execute-flat-bias-broadcast-compute combiner buf bias-data buf
                                                                    size N dtype)))))))))))))

;; The scheme backend runs a GEMM whose epilogue is not one of the
;; registry's forms through the prepared gemm-plan, and then the epilogue
;; in place over the product, in a strided loop on the host.
(define (scheme-gemm-select sg)
  (let ((g (standard-gemm sg)))
    (and g
         (let ((epilogue (list-ref g 5)))
           (and epilogue
                (eq? (car epilogue) 'map)
                (let* ((A-ref (segop-operand-slot (list-ref g 0)))
                       (B-ref (segop-operand-slot (list-ref g 1)))
                       (out   (list-ref g 2))
                       (plan  (list-ref g 4))
                       (host  (gemm-map-epilogue-executor out (cadr epilogue) (cddr epilogue))))
                  (cons 'segop/gemm-epilogue-host
                        (lambda (deref buf)
                          (execute-gemm-plan/into! plan (deref A-ref) (deref B-ref) buf)
                          (host deref buf)))))))))

;;; ------------------------------------------------------------
;;; Batched contractions
;;; ------------------------------------------------------------
;;;
;;; A batched matmul with frame F (rank f) becomes a seg-reduce over the space
;;; F ++ (M N K) with K reduced.  A framed operand is read through the frame
;;; dimensions followed by the 2-D GEMM map, (dim f) (dim f+2) for A and
;;; (dim f+2) (dim f+1) for B; an operand shared by every cell is read
;;; through the 2-D map alone.  The body is (mul (in 0) (in 1)), the output
;;; is the row-major pool buffer, and the combine spec follows the gemm-plan
;;; of one cell, as for a 2-D matmul.

;; The index expressions of the batched GEMM operands for frame rank f.
(define (frame-dim-exprs f) (map (lambda (i) `(dim ,i)) (iota f)))
(define (batched-A-exprs f shared?)
  (list->vector (append (if shared? '() (frame-dim-exprs f)) `((dim ,f) (dim ,(+ f 2))))))
(define (batched-B-exprs f shared?)
  (list->vector (append (if shared? '() (frame-dim-exprs f)) `((dim ,(+ f 2)) (dim ,(+ f 1))))))
(define (batched-out-exprs f)
  (list->vector (frame-dim-exprs (+ f 2))))

(define (lower-batch-matmul-binding A-tr B-tr A-ref B-ref pool-idx shape strides dtype)
  "The seg-reduce of the batched matmul of the trace-time arrays A-tr and B-tr,
   found at A-ref and B-ref, writing pool slot pool-idx of the given shape,
   strides and dtype; #f when blas-exec cannot multiply them."
  (and (batch-matmul-concrete? A-tr B-tr)
       (eq? (get-morphism-dtype A-tr) dtype)
       (let* ((ra (vector-length (get-morphism-shape A-tr)))
              (rb (vector-length (get-morphism-shape B-tr)))
              (f  (- (vector-length shape) 2))
              (operand
               (lambda (tr ref exprs)
                 (cases array-morphism tr
                   (concrete-array (_ s st off dt _ _)
                     (make-segop-operand ref dt s st off (make-index-map exprs #f 0.0 #f)))
                   (else (error "lower-batch-matmul-binding: operand not concrete" tr)))))
              (A    (operand A-tr A-ref (batched-A-exprs f (= ra 2))))
              (B    (operand B-tr B-ref (batched-B-exprs f (= rb 2))))
              (plan (prepare-batched-gemm (operand-layout-array A) (operand-layout-array B)))
              (sA   (get-morphism-shape A-tr))
              (space (list->vector
                      (append (vector->list (subvector shape 0 f))
                              (list (vector-ref sA (- ra 2))
                                    (vector-ref shape (+ f 1))
                                    (vector-ref sA (- ra 1))))))
              (out  (make-segop-operand pool-idx dtype shape strides 0
                                        (make-index-map (batched-out-exprs f) #f 0.0 #f))))
         (and plan
              (make-seg-reduce space 1 (list A B)
                               (list (make-red-result out gemm-body
                                                      (gemm-combine-spec (batched-gemm-plan-cell plan)
                                                                         dtype)
                                                      #f)))))))

(define (standard-batched-gemm sg)
  "The batched GEMM that computes seg-reduce sg, as a list (A B out plan),
   or #f when sg is not in the form lower-batch-matmul-binding produces,
   combine spec included.  plan is the batched-gemm-plan for the operands."
  (cases segop sg
    (seg-reduce (space nred ins results map-outs fallback)
      (and
       (>= (vector-length space) 4) (= nred 1) (= (length results) 1) (null? map-outs)
       (= (length ins) 2)
       (let* ((f     (- (vector-length space) 3))
              (A     (car ins))
              (B     (cadr ins))
              (res   (car results))
              (out   (red-result-out res))
              (dtype (segop-operand-dtype out))
              (frame (vector->list (subvector space 0 f)))
              (M     (vector-ref space f))
              (N     (vector-ref space (+ f 1)))
              (K     (vector-ref space (+ f 2)))
              (a-shared? (= (vector-length (segop-operand-shape A)) 2))
              (b-shared? (= (vector-length (segop-operand-shape B)) 2)))
         (and
          (not (red-result-finalize res))
          (not (and a-shared? b-shared?))
          (equal? (red-result-body res) gemm-body)
          (operand-exprs-are? A (batched-A-exprs f a-shared?))
          (equal? (segop-operand-shape A)
                  (list->vector (append (if a-shared? '() frame) (list M K))))
          (operand-exprs-are? B (batched-B-exprs f b-shared?))
          (equal? (segop-operand-shape B)
                  (list->vector (append (if b-shared? '() frame) (list K N))))
          (operand-exprs-are? out (batched-out-exprs f))
          (operand-flat? out (list->vector (append frame (list M N))))
          (eq? (segop-operand-dtype A) dtype)
          (let ((plan (prepare-batched-gemm (operand-layout-array A) (operand-layout-array B))))
            (and plan
                 (combine-specs-equal? (red-result-combine res)
                                       (gemm-combine-spec (batched-gemm-plan-cell plan) dtype))
                 (list A B out plan)))))))
    (else #f)))

;; Batched GEMMs run every cell through one prepared batched-gemm-plan.
(define (batched-gemm-select sg)
  (let ((g (standard-batched-gemm sg)))
    (and g
         (let ((A-ref (segop-operand-slot (car g)))
               (B-ref (segop-operand-slot (cadr g)))
               (plan  (cadddr g)))
           (cons 'segop/gemm-batched
                 (lambda (deref buf)
                   (execute-batched-gemm-plan/into! plan (deref A-ref) (deref B-ref) buf)))))))

;; True when fused output o is the product of a GEMM.
(define (gemm-fused-out? o)
  (let ((red (fused-out-red o))) (and red (eq? (car red) 'gemm))))

;; The epilogue body of a GEMM finalize fin of dtype dtype: the product as
;; stored, (cast dtype (reduced)) or (reduced), becomes (in 0), and the
;; finalize's input j, j >= 2, becomes (in j-1).
(define (gemm-finalize->epilogue fin dtype)
  (let walk ((e fin))
    (cond ((not (pair? e)) e)
          ((equal? e '(reduced)) '(in 0))
          ((and (eq? (car e) 'cast) (eq? (cadr e) dtype) (equal? (caddr e) '(reduced))) '(in 0))
          ((eq? (car e) 'in) `(in ,(- (cadr e) 1)))
          (else (map walk e)))))

;; True when body e applies one op, or one opaque procedure, to the inputs
;; (in 0) ... (in (n - 1)) in order.
(define (single-call-body? e n)
  (or (and (single-op-body-op e n) #t)
      (and (pair? e) (eq? (car e) 'opaque) (equal? (cddr e) (input-body-args n)))))

;; The procedure computing epilogue body e of n inputs: the combiner of
;; the op, or the opaque procedure, that e applies to its inputs in order;
;; otherwise e compiled.
(define (epilogue-combiner e n)
  (let ((op (single-op-body-op e n)))
    (cond ((and op (scalar-op? op) (= (scalar-op-arity op) n)) (scalar-op-combiner op))
          ((and (pair? e) (eq? (car e) 'opaque) (equal? (cddr e) (input-body-args n))) (cadr e))
          (else (scalar-expr->procedure e n)))))

(define (lower-gemm-template tpl in-arrs in-slots out-slot)
  "The seg-reduce of fusion template tpl, whose result is a GEMM product
   with its epilogue as finalize: the product of the first two arrays of
   in-arrs, with the epilogue applied to it in place, written to pool slot
   out-slot.  The other arrays are the epilogue's inputs, read through
   tpl's maps.  #f when blas-exec cannot multiply the first two arrays."
  (let* ((o      (car (fusion-template-outs tpl)))
         (shape  (fused-out-shape o))
         (dtype  (fused-out-dtype o))
         (fin    (cdr (fused-out-red o)))
         (out    (make-segop-operand out-slot dtype shape (compute-strides shape) 0
                                     (identity-index-map 2)))
         (extras (map (lambda (arr slot m)
                        (cases array-morphism arr
                          (concrete-array (_ s st off dt _ _) (make-segop-operand slot dt s st off m))
                          (else (error "lower-gemm-template: input not concrete" arr))))
                      (cddr in-arrs) (cddr in-slots) (cddr (fusion-template-in-maps tpl)))))
    (lower-matmul-binding (car in-arrs) (cadr in-arrs) (car in-slots) (cadr in-slots)
                          out-slot shape (compute-strides shape) dtype
                          (and fin
                               (let ((e (gemm-finalize->epilogue fin dtype)))
                                 (make-seg-map shape (cons out extras) (list out) (list e)
                                               (epilogue-combiner e (+ 1 (length extras)))))))))

;; True when segop template tpl describes a GEMM with an epilogue.
(define (gemm-template? tpl)
  (let ((outs (segop-template-outs tpl)))
    (and (pair? outs) (eq? (template-output-combine (car outs)) 'gemm))))

;; True when the registry runs GEMM template tpl natively: its epilogue is
;; one op applied to the product, or to the product and a row bias of the
;; output's dtype, f32 or f64, read along the columns.  These run as flat
;; kernels on the stored product.
(define (gemm-template-native? tpl)
  (let* ((ins (segop-template-ins tpl))
         (res (car (segop-template-outs tpl)))
         (dt  (template-output-dtype res))
         (fin (template-output-finalize res))
         (extras (cddr ins)))
    (and (null? (cdr (segop-template-outs tpl)))
         (fused-kernel-dtype? dt)
         fin
         (<= (length extras) 1)
         (single-call-body? (gemm-finalize->epilogue fin dt) (+ 1 (length extras)))
         (every (lambda (b) (and (template-input-flat? b)
                                 (eq? (template-input-dtype b) dt)
                                 (equal? (index-map-exprs (template-input-map b)) '#((dim 1)))))
                extras))))

;;; ------------------------------------------------------------
;;; Fused kernels
;;; ------------------------------------------------------------
;;;
;;; A seg-map whose body has a fused kernel in the active activation
;;; backend (see activation-backend-add-fused-kernel!) runs as that kernel
;;; (tag segop/fused-kernel) when it has one output and one to three
;;; inputs, all flat and of the output's dtype, f32 or f64.  Kernels are
;;; keyed by canonical bodies; the inputs are passed in the order of the
;;; canonical body.  When the active backend holds no such kernel at the
;;; time the step runs, the segop's fallback computes it on the host.

;; The canonical body of body over n inputs, and the permutation taking
;; canonical inputs to body's.
(define (canonical-body+perm body n)
  (call-with-values (lambda () (canonical-scalar-body body n)) cons))

(define (fused-kernel-dtype? dt) (memq dt '(f32 f64)))

;; True when index map m reads, at each point of space, the operand element
;; with the same coordinates (a dimension of extent 1 may be read at 0).
(define (space-identity-map? m space)
  (let ((exprs (index-map-exprs m)))
    (and exprs
         (not (index-map-guard m))
         (= (vector-length exprs) (vector-length space))
         (let loop ((d 0))
           (or (= d (vector-length space))
               (let ((e (vector-ref exprs d)))
                 (and (or (equal? e `(dim ,d))
                          (and (eqv? e 0) (= (vector-ref space d) 1)))
                      (loop (+ d 1)))))))))

(define (registry-native? tpl)
  "The native-capability query of the registry backend: true when the
   segop template tpl has one output and one to three flat inputs read at
   the iteration point, all of the output's dtype, f32 or f64, and the
   active activation backend holds a fused kernel for its canonical body;
   when tpl is a GEMM whose epilogue runs as a flat kernel (see
   gemm-template-native?); or when tpl is a column sum that a fused
   column-sum kernel of the active activation backend computes (see
   colsum-template?)."
  (cond ((= (segop-template-nred tpl) 0) (registry-map-native? tpl))
        ((gemm-template? tpl) (gemm-template-native? tpl))
        (else (colsum-template? tpl))))

(define (registry-map-native? tpl)
  (let ((ins (segop-template-ins tpl)) (outs (segop-template-outs tpl))
        (space (segop-template-space tpl)))
    (and (= (length outs) 1)
         (<= 1 (length ins) 3)
         (let ((dt (template-output-dtype (car outs))))
           (and (fused-kernel-dtype? dt)
                (every (lambda (i) (and (template-input-flat? i)
                                        (eq? (template-input-dtype i) dt)
                                        (space-identity-map? (template-input-map i) space)))
                       ins)
                (lookup-fused-kernel (active-activation-backend)
                                     (car (canonical-body+perm (template-output-body (car outs))
                                                               (length ins)))
                                     dt)
                #t)))))

(define (fused-kernel-executor body perm fallback refs size dtype)
  (let ((crefs (map (lambda (k) (list-ref refs k)) perm)))
    (lambda (deref buf)
      (let ((kernel (lookup-fused-kernel (active-activation-backend) body dtype)))
        (if kernel
            (kernel size (map (lambda (r) (operand-data 'fused-kernel (deref r))) crefs) buf)
            (let ((datas (map (lambda (r) (operand-data 'fused-kernel (deref r))) refs)))
              (case (length datas)
                ((1) (execute-flat-unary-compute fallback (car datas) dtype buf size dtype))
                ((2) (execute-flat-binary-compute fallback (car datas) dtype (cadr datas) dtype
                                                  buf size dtype))
                (else ((flat-ternary-executor fallback 0 1 2 size dtype)
                       (lambda (k) (concrete-array (list-ref datas k) (vector size)
                                                   (vector 1) 0 dtype -1 -1))
                       buf)))))))))

(define (registry-fused-select sg)
  (with-single-output-seg-map sg
    (lambda (space ins out body fallback)
      (let ((dtype (segop-operand-dtype out)) (n (length ins)))
        (and (fused-kernel-dtype? dtype)
             (<= 1 n 3)
             (operand-flat? out space)
             (every (lambda (o) (and (operand-flat? o space) (eq? (segop-operand-dtype o) dtype)
                                     (space-identity-map? (segop-operand-map o) space)))
                    ins)
             (let ((cp (canonical-body+perm body n)))
               (and (lookup-fused-kernel (active-activation-backend) (car cp) dtype)
                    (cons 'segop/fused-kernel
                          (fused-kernel-executor (car cp) (cdr cp) fallback
                                                 (map segop-operand-slot ins)
                                                 (shape-size space) dtype)))))))))

;;; ------------------------------------------------------------
;;; Fused column sums
;;; ------------------------------------------------------------
;;;
;;; A seg-reduce that sums a body over the rows of rows x cols arrays and
;;; also stores the body's values runs as a column-sum kernel of the
;;; active activation backend (tag segop/colsum-kernel), when the backend
;;; holds one for the canonical body.  Its space is (cols rows), with the
;;; rows reduced; the inputs and the stored values are row-major arrays of
;;; one dtype, f32 or f64.  The kernel adds the rows in order, as the
;;; unfused axis-0 sum does.  If the kernel is missing when the step runs,
;;; the strided reduction executor computes the segop instead.

(define colsum-in-exprs '#((dim 1) (dim 0)))

(define (colsum-out-exprs? exprs)
  (member exprs '(#((dim 0)) #(0 (dim 0)))))

(define (colsum-spec? spec dtype)
  (and (eq? (combine-spec-op spec) 'add)
       (eq? (combine-spec-order spec) 'sequential)
       (eq? (combine-spec-acc-dtype spec) dtype)
       (eqv? (combine-spec-neutral spec) 0.0)))

(define (colsum-template? tpl)
  (let ((space (segop-template-space tpl))
        (ins   (segop-template-ins tpl))
        (outs  (segop-template-outs tpl)))
    (and (= (segop-template-nred tpl) 1)
         (= (vector-length space) 2)
         (= (length outs) 2)
         (<= 1 (length ins) 3)
         (let* ((res (car outs)) (mo (cadr outs)) (dt (template-output-dtype res)))
           (and (fused-kernel-dtype? dt)
                (template-output-combine res)
                (colsum-spec? (template-output-combine res) dt)
                (not (template-output-finalize res))
                (template-output-map res)
                (colsum-out-exprs? (index-map-exprs (template-output-map res)))
                (not (template-output-combine mo))
                (eq? (template-output-dtype mo) dt)
                (eq? (template-output-body mo) (template-output-body res))
                (template-output-map mo)
                (equal? (index-map-exprs (template-output-map mo)) colsum-in-exprs)
                (every (lambda (i) (and (template-input-flat? i)
                                        (eq? (template-input-dtype i) dt)
                                        (equal? (index-map-exprs (template-input-map i)) colsum-in-exprs)))
                       ins)
                (lookup-colsum-kernel (active-activation-backend)
                                      (car (canonical-body+perm (template-output-body res) (length ins)))
                                      dt)
                #t)))))

(define (registry-colsum-select sg)
  (cases segop sg
    (seg-reduce (space nred ins results map-outs fallback)
      (and (= nred 1)
           (= (vector-length space) 2)
           (= (length results) 1)
           (= (length map-outs) 1)
           (<= 1 (length ins) 3)
           (let* ((res   (car results))
                  (out   (red-result-out res))
                  (mo    (caar map-outs))
                  (body  (red-result-body res))
                  (dt    (segop-operand-dtype out))
                  (cols  (vector-ref space 0))
                  (rows  (vector-ref space 1))
                  (full  (vector rows cols)))
             (and (fused-kernel-dtype? dt)
                  (eq? (cdar map-outs) body)
                  (colsum-spec? (red-result-combine res) dt)
                  (not (red-result-finalize res))
                  (operand-row-major? out)
                  (= (shape-size (segop-operand-shape out)) cols)
                  (colsum-out-exprs? (index-map-exprs (segop-operand-map out)))
                  (eq? (segop-operand-dtype mo) dt)
                  (operand-flat? mo full)
                  (equal? (index-map-exprs (segop-operand-map mo)) colsum-in-exprs)
                  (every (lambda (o) (and (operand-flat? o full) (eq? (segop-operand-dtype o) dt)
                                          (equal? (index-map-exprs (segop-operand-map o)) colsum-in-exprs)))
                         ins)
                  (let* ((cp    (canonical-body+perm body (length ins)))
                         (refs  (map segop-operand-slot ins))
                         (crefs (map (lambda (k) (list-ref refs k)) (cdr cp)))
                         (host  (strided-red-executor space nred ins results map-outs)))
                    (and (lookup-colsum-kernel (active-activation-backend) (car cp) dt)
                         (cons 'segop/colsum-kernel
                               (lambda (deref buf #!optional (extra-bufs '()))
                                 (let ((kernel (lookup-colsum-kernel (active-activation-backend)
                                                                     (car cp) dt)))
                                   (if kernel
                                       (kernel rows cols
                                               (map (lambda (r) (operand-data 'colsum-kernel (deref r)))
                                                    crefs)
                                               (car extra-bufs) buf)
                                       (host deref buf extra-bufs)))))))))))
    (else #f)))

;;; ------------------------------------------------------------
;;; Convolutions
;;; ------------------------------------------------------------
;;;
;;; The convolution ops are segops over spaces with one dimension per loop
;;; index, so that each index map is a short expression:
;;;
;;;   im2col            seg-map over (N C KH KW OH OW):
;;;                     col[n, fan, oh*OW + ow] = x[n, c, ih, iw], or 0
;;;                     where (ih, iw) is outside the image
;;;   col2im            seg-reduce over (N C H W ; KH KW):
;;;                     x[n, c, h, w] = sum of col[n, fan, oh*OW + ow]
;;;   conv-fwd          seg-reduce over (N OH OW CO ; C KH KW):
;;;                     y[row, co] = b[co] + sum of x[n, c, ih, iw] * wt[fan, co]
;;;   conv-bwd-weights  seg-reduce over (C KH KW CO ; N OH OW):
;;;                     dwt[fan, co] = sum of g[row, co] * x[n, c, ih, iw]
;;;   conv-bwd-data     seg-reduce over (N C H W ; KH KW CO):
;;;                     dx[n, c, h, w] = sum of g[row, co] * wt[fan, co]
;;;
;;; with fan = c*KH*KW + kh*KW + kw, row = n*OH*OW + oh*OW + ow,
;;; ih = oh*SH + kh - PH and iw = ow*SW + kw - PW.  The image x is
;;; [N C H W] (layout nchw), [N H W C] (nhwc), or [C H W] (rank 3, nchw,
;;; with N = 1).  col2im and conv-bwd-data gather: at (h, w) they read the
;;; windows that cover it, those where h + PH - kh is a multiple of SH with
;;; the quotient oh in [0, OH), and likewise for w; the other terms read 0.
;;;
;;; col2im adds its terms in increasing (kh, kw), as its executors do.  The
;;; contractions have the unspecified order, since BLAS orders their sums.
;;;
;;; conv-select recognizes these segops by reading the parameters back from
;;; their maps and rebuilding the maps from them, and runs them with the
;;; convolution executors of realization.scm.  The contractions take a
;;; scratch buffer for the image columns that the convolutions of the plan
;;; share (see current-conv-scratch).

(define-record conv-geom kind layout rank N C H W KH KW SH SW PH PW OH OW CO)

(define (conv-out-size n k s p) (+ 1 (quotient (+ n (* 2 p) (- k)) s)))

;; ih for output dimension o and kernel dimension k.
(define (conv-window o k S P) `(+ (* ,S (dim ,o)) (+ (dim ,k) ,(- P))))

;; h + PH - kh, for image dimension d and kernel dimension k.
(define (conv-cover d k P) `(+ (dim ,d) (+ (* -1 (dim ,k)) ,P)))

;; The output index oh whose window covers d at kernel index k, and the
;; guard entries that hold when there is one.
(define (conv-gather d k S P) `(floordiv ,(conv-cover d k P) ,S))
(define (conv-gather-guard d k S P O)
  (list (list `(mod ,(conv-cover d k P) ,S) 0 1)
        (list (conv-gather d k S P) 0 O)))

(define (conv-fan g c kh kw)
  `(+ (* ,(* (conv-geom-KH g) (conv-geom-KW g)) ,c) (+ (* ,(conv-geom-KW g) ,kh) ,kw)))
(define (conv-row g n oh ow)
  `(+ (* ,(* (conv-geom-OH g) (conv-geom-OW g)) ,n) (+ (* ,(conv-geom-OW g) ,oh) ,ow)))
(define (conv-col g n fan oh ow)
  (let ((ohw `(+ (* ,(conv-geom-OW g) ,oh) ,ow)))
    (if (= (conv-geom-rank g) 4) (vector n fan ohw) (vector fan ohw))))

;; The image index of (n, c, h, w) in g's layout.
(define (conv-image g n c h w)
  (cond ((= (conv-geom-rank g) 3) (vector c h w))
        ((eq? (conv-geom-layout g) 'nhwc) (vector n h w c))
        (else (vector n c h w))))

(define (conv-image-shape g)
  (let ((N (conv-geom-N g)) (C (conv-geom-C g)) (H (conv-geom-H g)) (W (conv-geom-W g)))
    (cond ((= (conv-geom-rank g) 3) (vector C H W))
          ((eq? (conv-geom-layout g) 'nhwc) (vector N H W C))
          (else (vector N C H W)))))

;; The image read at output dimensions oh, ow and kernel dimensions kh, kw,
;; with 0 outside the image.
(define (conv-image-read g n c oh kh ow kw)
  (let ((ih (conv-window oh kh (conv-geom-SH g) (conv-geom-PH g)))
        (iw (conv-window ow kw (conv-geom-SW g) (conv-geom-PW g))))
    (make-index-map (conv-image g `(dim ,n) `(dim ,c) ih iw)
                    (list (list ih 0 (conv-geom-H g)) (list iw 0 (conv-geom-W g)))
                    0.0 #f)))

(define (dims-map . ds) (make-index-map (list->vector (map (lambda (d) `(dim ,d)) ds)) #f 0.0 #f))
(define (exprs-map . es) (make-index-map (list->vector es) #f 0.0 #f))

;; The structure of the segop of convolution g of the given dtype, as a
;; list (space nred in-maps out-map body combine finalize).  For im2col,
;; nred is 0 and combine #f.
(define (conv-skeleton g dtype)
  (let* ((N (conv-geom-N g)) (C (conv-geom-C g)) (H (conv-geom-H g)) (W (conv-geom-W g))
         (KH (conv-geom-KH g)) (KW (conv-geom-KW g)) (SH (conv-geom-SH g)) (SW (conv-geom-SW g))
         (PH (conv-geom-PH g)) (PW (conv-geom-PW g)) (OH (conv-geom-OH g)) (OW (conv-geom-OW g))
         (CO (conv-geom-CO g))
         (contraction (make-combine-spec 'add 0.0 dtype 'unspecified))
         (image-out (make-index-map (conv-image g '(dim 0) '(dim 1) '(dim 2) '(dim 3)) #f 0.0 #f))
         ;; The gather over image dimensions 2, 3 and kernel dimensions 4, 5.
         (gh (conv-gather 2 4 SH PH)) (gw (conv-gather 3 5 SW PW))
         (gather-guard (append (conv-gather-guard 2 4 SH PH OH) (conv-gather-guard 3 5 SW PW OW))))
    (case (conv-geom-kind g)
      ((im2col)
       (list (vector N C KH KW OH OW) 0
             (list (conv-image-read g 0 1 4 2 5 3))
             (make-index-map (conv-col g '(dim 0) (conv-fan g '(dim 1) '(dim 2) '(dim 3))
                                       '(dim 4) '(dim 5))
                             #f 0.0 #f)
             '(in 0) #f #f))
      ((col2im)
       (list (vector N C H W KH KW) 2
             (list (make-index-map (conv-col g '(dim 0) (conv-fan g '(dim 1) '(dim 4) '(dim 5)) gh gw)
                                   gather-guard 0.0 #f))
             image-out
             '(in 0) (make-combine-spec 'add 0.0 dtype) #f))
      ((conv-fwd)
       (list (vector N OH OW CO C KH KW) 3
             (list (conv-image-read g 0 4 1 5 2 6)
                   (exprs-map (conv-fan g '(dim 4) '(dim 5) '(dim 6)) '(dim 3))
                   (dims-map 3))
             (exprs-map (conv-row g '(dim 0) '(dim 1) '(dim 2)) '(dim 3))
             '(mul (in 0) (in 1)) contraction '(add (reduced) (in 2))))
      ((conv-bwd-weights)
       (list (vector C KH KW CO N OH OW) 3
             (list (exprs-map (conv-row g '(dim 4) '(dim 5) '(dim 6)) '(dim 3))
                   (conv-image-read g 4 0 5 1 6 2))
             (exprs-map (conv-fan g '(dim 0) '(dim 1) '(dim 2)) '(dim 3))
             '(mul (in 0) (in 1)) contraction #f))
      ((conv-bwd-data)
       (list (vector N C H W KH KW CO) 3
             (list (make-index-map (vector (conv-row g '(dim 0) gh gw) '(dim 6)) gather-guard 0.0 #f)
                   (exprs-map (conv-fan g '(dim 1) '(dim 4) '(dim 5)) '(dim 6)))
             image-out
             '(mul (in 0) (in 1)) contraction #f))
      (else (error "conv-skeleton: unknown kind" (conv-geom-kind g))))))

;; The segop of convolution g over operands ins (operand records whose
;; maps are replaced by the skeleton's) writing out.
(define (conv-segop g dtype ins out)
  (let* ((sk     (conv-skeleton g dtype))
         (with-map (lambda (o m)
                     (make-segop-operand (segop-operand-slot o) (segop-operand-dtype o)
                                         (segop-operand-shape o) (segop-operand-strides o)
                                         (segop-operand-offset o) m)))
         (ins    (map with-map ins (list-ref sk 2)))
         (out    (with-map out (list-ref sk 3))))
    (if (= (list-ref sk 1) 0)
        (make-seg-map (list-ref sk 0) ins (list out) (list (list-ref sk 4)))
        (make-seg-reduce (list-ref sk 0) (list-ref sk 1) ins
                         (list (make-red-result out (list-ref sk 4) (list-ref sk 5) (list-ref sk 6)))))))

;; The geometry of a convolution binding with op op and meta meta, whose
;; inputs had the trace-time shapes in-shapes and whose output has shape
;; out-shape.
(define (conv-binding-geom op meta in-shapes out-shape)
  (let* ((pair  (lambda (key) (cdr (assq key meta))))
         (ks (pair 'kernel-size)) (st (pair 'stride)) (pd (pair 'padding))
         (so (let ((p (assq 'spatial-output meta))) (and p (cdr p))))
         (layout (let ((p (assq 'layout meta))) (if (and p (eq? (cdr p) 'nhwc)) 'nhwc 'nchw)))
         (kind (case op
                 ((im2col) 'im2col) ((col2im) 'col2im) ((conv2d-fwd) 'conv-fwd)
                 ((conv2d-bwd-data) 'conv-bwd-data) ((conv2d-bwd-weights) 'conv-bwd-weights)))
         (image (case kind
                  ((im2col conv-fwd) (car in-shapes))
                  ((conv-bwd-weights) (cadr in-shapes))
                  (else out-shape)))
         (rank (vector-length image))
         (nhwc? (and (= rank 4) (eq? layout 'nhwc)))
         (dim (lambda (nchw4 nhwc4 nchw3)
                (vector-ref image (cond (nhwc? nhwc4) ((= rank 4) nchw4) (else nchw3)))))
         (H (dim 2 1 1)) (W (dim 3 2 2))
         (KH (car ks)) (KW (cadr ks)) (SH (car st)) (SW (cadr st)) (PH (car pd)) (PW (cadr pd))
         (CO (case kind
               ((conv-fwd) (vector-ref out-shape 1))
               ((conv-bwd-data conv-bwd-weights) (vector-ref (car in-shapes) 1))
               (else 0))))
    (make-conv-geom kind (if nhwc? 'nhwc 'nchw) rank
                    (if (= rank 4) (vector-ref image 0) 1) (dim 1 3 0) H W KH KW SH SW PH PW
                    (if so (car so) (conv-out-size H KH SH PH))
                    (if so (cadr so) (conv-out-size W KW SW PW))
                    CO)))

(define (lower-conv-binding op meta in-traces in-refs pool-idx shape strides dtype)
  "The segop of a convolution binding (im2col, col2im, conv2d-fwd,
   conv2d-bwd-data or conv2d-bwd-weights) with meta meta, whose inputs had
   the trace-time arrays in-traces and are found at in-refs, writing pool
   slot pool-idx of the given shape, strides and dtype."
  (let* ((operand (lambda (tr ref)
                    (cases array-morphism tr
                      (concrete-array (_ s st off dt _ _)
                        (make-segop-operand ref dt s st off seg-lib-operand-map))
                      (else (error "lower-conv-binding: input not concrete" op tr)))))
         (ins (map operand in-traces in-refs))
         (g   (conv-binding-geom op meta (map segop-operand-shape ins) shape)))
    (conv-segop g dtype ins (make-segop-operand pool-idx dtype shape strides 0 seg-lib-operand-map))))

;; The parameters of window expression e, (S o k P) for oh*S + kh - P
;; with o and k the dimensions of oh and kh, or #f.
(define (parse-conv-window e)
  (match e
    (('+ ('* (? integer? S) ('dim o)) ('+ ('dim k) (? integer? negP))) (list S o k (- negP)))
    (_ #f)))

;; The parameters (S P O) of a gather guard entry pair, or #f.
(define (parse-conv-gather-guard g1 g2)
  (match (list g1 g2)
    (((('mod ('+ ('dim _) ('+ ('* -1 ('dim _)) (? integer? P))) (? integer? S)) 0 1)
      (_ 0 (? integer? O)))
     (list S P O))
    (_ #f)))

;; The geometry of a convolution of kind kind that segop sg could be, read
;; from its maps and shapes, or #f.  conv-segop-geom checks that sg is that
;; convolution.
(define (read-conv-geom kind sg)
  (let* ((space (segop-space sg))
         (ins   (segop-inputs sg))
         (out   (car (segop-outputs sg)))
         (ext   (lambda (d) (vector-ref space d)))
         (image-geom
          ;; From an image operand read through windows.
          (lambda (o CO)
            (let* ((shape (segop-operand-shape o))
                   (rank  (vector-length shape))
                   (exprs (index-map-exprs (segop-operand-map o)))
                   (try   (lambda (layout ih iw)
                            (let ((h (parse-conv-window (vector-ref exprs ih)))
                                  (w (parse-conv-window (vector-ref exprs iw))))
                              (and h w
                                   (let* ((nhwc? (eq? layout 'nhwc))
                                          (dim (lambda (nchw4 nhwc4 nchw3)
                                                 (vector-ref shape (cond (nhwc? nhwc4) ((= rank 4) nchw4)
                                                                         (else nchw3))))))
                                     (make-conv-geom kind layout rank
                                                     (if (= rank 4) (vector-ref shape 0) 1)
                                                     (dim 1 3 0) (dim 2 1 1) (dim 3 2 2)
                                                     (ext (cadr (cdr h))) (ext (cadr (cdr w)))
                                                     (car h) (car w) (cadddr h) (cadddr w)
                                                     (ext (cadr h)) (ext (cadr w)) CO)))))))
              (and exprs (memv rank '(3 4))
                   (if (= rank 3)
                       (try 'nchw 1 2)
                       (or (try 'nchw 2 3) (try 'nhwc 1 2)))))))
         (gather-geom
          ;; From a gather operand and the image output.
          (lambda (o CO)
            (let* ((guard (index-map-guard (segop-operand-map o)))
                   (shape (segop-operand-shape out))
                   (rank  (vector-length shape))
                   (exprs (index-map-exprs (segop-operand-map out)))
                   (layout (cond ((equal? exprs '#((dim 0) (dim 1) (dim 2) (dim 3))) 'nchw)
                                 ((equal? exprs '#((dim 0) (dim 2) (dim 3) (dim 1))) 'nhwc)
                                 ((equal? exprs '#((dim 1) (dim 2) (dim 3))) 'nchw)
                                 (else #f)))
                   (h (and guard (= (length guard) 4)
                           (parse-conv-gather-guard (list-ref guard 0) (list-ref guard 1))))
                   (w (and h (parse-conv-gather-guard (list-ref guard 2) (list-ref guard 3)))))
              (and layout w
                   (make-conv-geom kind layout rank (ext 0) (ext 1) (ext 2) (ext 3) (ext 4) (ext 5)
                                   (car h) (car w) (cadr h) (cadr w) (caddr h) (caddr w) CO))))))
    (and (= (vector-length space) (if (memq kind '(im2col col2im)) 6 7))
         (case kind
           ((im2col) (and (= (length ins) 1) (image-geom (car ins) 0)))
           ((conv-fwd) (and (= (length ins) 3) (image-geom (car ins) (ext 3))))
           ((conv-bwd-weights) (and (= (length ins) 2) (image-geom (cadr ins) (ext 3))))
           ((col2im) (and (= (length ins) 1) (gather-geom (car ins) 0)))
           ((conv-bwd-data) (and (= (length ins) 2) (gather-geom (car ins) (ext 6))))
           (else #f)))))

(define (index-map-equal? a b)
  (and (not (index-map-proc a)) (not (index-map-proc b))
       (equal? (index-map-exprs a) (index-map-exprs b))
       (equal? (index-map-guard a) (index-map-guard b))
       (equal? (index-map-fill a) (index-map-fill b))))

(define (combine-spec-equal? a b)
  (or (and (not a) (not b))
      (and a b
           (eq? (combine-spec-op a) (combine-spec-op b))
           (eqv? (combine-spec-neutral a) (combine-spec-neutral b))
           (eq? (combine-spec-acc-dtype a) (combine-spec-acc-dtype b))
           (eq? (combine-spec-order a) (combine-spec-order b)))))

;; True when sg is exactly the segop conv-segop builds for geometry g.
(define (conv-segop-matches? sg g)
  (let* ((out   (car (segop-outputs sg)))
         (dtype (segop-operand-dtype out))
         (sk    (conv-skeleton g dtype)))
    (and (memq dtype '(f32 f64))
         (every (lambda (o) (eq? (segop-operand-dtype o) dtype)) (segop-inputs sg))
         (equal? (segop-space sg) (list-ref sk 0))
         (= (length (segop-inputs sg)) (length (list-ref sk 2)))
         (every index-map-equal? (map segop-operand-map (segop-inputs sg)) (list-ref sk 2))
         (index-map-equal? (segop-operand-map out) (list-ref sk 3))
         (cases segop sg
           (seg-map (space ins outs bodies fallback)
             (and (= (list-ref sk 1) 0) (= (length outs) 1)
                  (equal? bodies (list (list-ref sk 4)))))
           (seg-reduce (space nred ins results map-outs fallback)
             (and (= nred (list-ref sk 1)) (= (length results) 1) (null? map-outs)
                  (let ((r (car results)))
                    (and (equal? (red-result-body r) (list-ref sk 4))
                         (combine-spec-equal? (red-result-combine r) (list-ref sk 5))
                         (equal? (red-result-finalize r) (list-ref sk 6))))))
           (else #f)))))

(define (conv-segop-geom sg)
  "The geometry of the convolution segop sg, or #f when sg is not one."
  (let ((kinds (cases segop sg
                 (seg-map (space ins outs bodies fallback) '(im2col))
                 (seg-reduce (space nred ins results map-outs fallback)
                   (case nred ((2) '(col2im)) ((3) '(conv-fwd conv-bwd-weights conv-bwd-data)) (else '())))
                 (else '()))))
    (any (lambda (kind)
           (let ((g (read-conv-geom kind sg)))
             (and g (conv-segop-matches? sg g) g)))
         kinds)))

;; The number of elements of the image-column scratch buffer that
;; convolution g needs, 0 for im2col and col2im.
(define (conv-scratch-size g)
  (if (memq (conv-geom-kind g) '(conv-fwd conv-bwd-data conv-bwd-weights))
      (* (conv-geom-N g) (conv-geom-OH g) (conv-geom-OW g)
         (conv-geom-C g) (conv-geom-KH g) (conv-geom-KW g))
      0))

;; Scratch buffers for the image columns that the convolutions of a plan
;; share, one per dtype.  The convolutions ask for a size while the plan is
;; compiled; a plan finalizer allocates each buffer at the largest size
;; asked for, and the buffer is looked up each time a convolution runs.
;; The instructions of a plan run one at a time, and each convolution
;; writes the whole buffer before reading it, so they can share it.
;; entries is a list of mutable vectors #(dtype size buffer).
(define-record conv-scratch entries)

(define (new-conv-scratch) (make-conv-scratch '()))

;; The scratch of the plan being compiled, made with its finalizer at the
;; first request, or #f outside plan compilation.  A convolution compiled
;; outside a plan gets a buffer of its own.
(define (current-conv-scratch)
  (and (segop-plan-context)
       (or (segop-plan-property 'conv-scratch)
           (let ((s (new-conv-scratch)))
             (segop-plan-property-set! 'conv-scratch s)
             (segop-plan-add-finalizer! (lambda () (conv-scratch-allocate! s) #f))
             s))))

;; The entry of s for dtype, now at least n elements long.
(define (conv-scratch-request! s dtype n)
  (let ((e (or (find (lambda (e) (eq? (vector-ref e 0) dtype)) (conv-scratch-entries s))
               (let ((e (vector dtype 0 #f)))
                 (conv-scratch-entries-set! s (cons e (conv-scratch-entries s)))
                 e))))
    (vector-set! e 1 (max n (vector-ref e 1)))
    e))

(define (conv-scratch-allocate! s)
  (for-each (lambda (e) (vector-set! e 2 (allocate-typed-vector (vector-ref e 0) (vector-ref e 1))))
            (conv-scratch-entries s)))

(define (conv-scratch-bytes s)
  (fold (lambda (e sum) (+ sum (* (if (eq? (vector-ref e 0) 'f32) 4 8) (vector-ref e 1))))
        0 (conv-scratch-entries s)))

;; A procedure of no arguments returning a scratch buffer of dtype with at
;; least n elements.
(define (conv-scratch-getter dtype n)
  (let ((s (current-conv-scratch)))
    (if s
        (let ((e (conv-scratch-request! s dtype n)))
          (lambda () (vector-ref e 2)))
        (let ((buf (allocate-typed-vector dtype n)))
          (lambda () buf)))))

(define (conv-select sg)
  (let ((g (conv-segop-geom sg)))
    (and g
         (let* ((ins    (segop-inputs sg))
                (refs   (map segop-operand-slot ins))
                (out    (car (segop-outputs sg)))
                (dtype  (segop-operand-dtype out))
                (shape  (segop-operand-shape out))
                (N (conv-geom-N g)) (C (conv-geom-C g)) (H (conv-geom-H g)) (W (conv-geom-W g))
                (KH (conv-geom-KH g)) (KW (conv-geom-KW g)) (SH (conv-geom-SH g)) (SW (conv-geom-SW g))
                (PH (conv-geom-PH g)) (PW (conv-geom-PW g)) (OH (conv-geom-OH g)) (OW (conv-geom-OW g))
                (CO (conv-geom-CO g))
                (nhwc? (eq? (conv-geom-layout g) 'nhwc))
                (data  (lambda (deref k) (concrete-data/row-major (deref (list-ref refs k)))))
                (blas? (lambda () (and (blas-enabled?) (blas-available?))))
                (scratch (and (> (conv-scratch-size g) 0)
                              (conv-scratch-getter dtype (conv-scratch-size g)))))
           (case (conv-geom-kind g)
             ((im2col)
              (let ((exec (cond ((= (conv-geom-rank g) 3)
                                 (lambda (buf src)
                                   (execute-im2col-unbatched buf src C H W KH KW SH SW PH PW OH OW dtype)))
                                (nhwc? execute-im2col-nhwc-classic)
                                (else execute-im2col-batched))))
                (cons 'segop/im2col
                      (if (= (conv-geom-rank g) 3)
                          (lambda (deref buf) (exec buf (data deref 0)))
                          (lambda (deref buf)
                            (exec buf (data deref 0) N C H W KH KW SH SW PH PW OH OW dtype))))))
             ((col2im)
              (let ((exec (cond ((= (conv-geom-rank g) 3) execute-col2im-unbatched)
                                (nhwc? execute-col2im-nhwc-classic)
                                (else execute-col2im-batched)))
                    (col-shape (segop-operand-shape (car ins))))
                (cons 'segop/col2im
                      (lambda (deref buf)
                        (exec buf shape (data deref 0) col-shape KH KW SH SW PH PW dtype)))))
             ((conv-fwd)
              (let ((fast (if nhwc? execute-conv-fwd-nhwc-blas execute-conv-fwd-blas))
                    (slow (if nhwc? execute-conv-fwd-nhwc execute-conv-fwd-nchw)))
                (cons 'segop/conv-fwd
                      (lambda (deref buf)
                        (let ((src (data deref 0)) (wt (data deref 1)) (b (data deref 2)))
                          (if (blas?)
                              (fast buf src wt b N C H W KH KW SH SW PH PW OH OW CO dtype (scratch))
                              (slow buf src wt b N C H W KH KW SH SW PH PW OH OW CO dtype)))))))
             ((conv-bwd-data)
              (let ((fast (if nhwc? execute-conv-bwd-data-nhwc-blas execute-conv-bwd-data-blas))
                    (slow (if nhwc? execute-conv-bwd-data-nhwc execute-conv-bwd-data-nchw))
                    (g-shape (segop-operand-shape (car ins))))
                (cons 'segop/conv-bwd-data
                      (lambda (deref buf)
                        (let ((gd (data deref 0)) (wt (data deref 1)))
                          (if (blas?)
                              (fast buf shape gd g-shape wt N C H W KH KW SH SW PH PW OH OW CO dtype (scratch))
                              (slow buf shape gd g-shape wt N C H W KH KW SH SW PH PW OH OW CO dtype)))))))
             ((conv-bwd-weights)
              (let ((fast (if nhwc? execute-conv-bwd-weights-nhwc-blas execute-conv-bwd-weights-blas))
                    (slow (if nhwc? execute-conv-bwd-weights-nhwc execute-conv-bwd-weights-nchw))
                    (g-shape (segop-operand-shape (car ins))))
                (cons 'segop/conv-bwd-weights
                      (lambda (deref buf)
                        (let ((gd (data deref 0)) (src (data deref 1)))
                          (if (blas?)
                              (fast buf shape gd g-shape src N C H W KH KW SH SW PH PW OH OW CO dtype (scratch))
                              (slow buf shape gd g-shape src N C H W KH KW SH SW PH PW OH OW CO dtype)))))))
             (else #f))))))

;;; ------------------------------------------------------------
;;; Backend registration and instruction compilation
;;; ------------------------------------------------------------

(define (registry-select sg)
  (or (registry-gemm-select sg)
      (batched-gemm-select sg)
      (conv-select sg)
      (registry-fused-select sg)
      (registry-colsum-select sg)
      (registry-elementwise-select sg)
      (select-standard-reduction sg '(axis0 axis1))
      (registry-gather-select sg)
      (scan-select sg)))

(define (scheme-select sg)
  (or (scheme-gemm-select sg)
      (batched-gemm-select sg)
      (conv-select sg)
      (scheme-elementwise-select sg)
      (select-standard-reduction sg '(1d axis0 axis1 general rows-1d))
      (strided-select sg)
      (strided-multi-select sg)
      (strided-red-select sg)
      (scan-select sg)
      (strided-scan-select sg)
      (host-index-fn-select sg)
      (interpreted-select sg)))

(register-segop-backend! (make-segop-backend 'registry registry-select
                                             kind: 'catalogue native?: registry-native?)
                         'last)
(register-segop-backend! (make-segop-backend 'scheme scheme-select) 'last)

;; The ri-segop of sg, or #f when no backend accepts it.  extra-outs
;; describes the outputs after the first (see ri-segop).
(define (segop-instruction sg pool-idx shape strides dtype #!optional (extra-outs '()))
  (and sg
       (let ((sel (select-segop-executor sg)))
         (and sel (ri-segop pool-idx shape strides dtype sg (cdr sel) (car sel) extra-outs)))))

(define (compile-fused-segop tpl in-traces in-refs pool-idx extra-outs shape strides dtype)
  "The ri-segop instruction of a fused binding with template tpl, whose
   inputs had the trace-time arrays in-traces and are found at in-refs.
   The first output goes to pool slot pool-idx, the others as extra-outs
   describes."
  (or (segop-instruction
       (lower-fused-binding tpl in-traces in-refs
                            (cons pool-idx (map (lambda (e) (vector-ref e 1)) extra-outs)))
       pool-idx shape strides dtype extra-outs)
      (error "compile-fused-segop: no backend accepts the fused segop")))

(define (compile-elementwise-segop b combiner in-traces in-refs pool-idx shape strides dtype)
  "The ri-segop instruction of element-wise binding b, or #f when the
   binding cannot be lowered or no backend accepts its segop."
  (segop-instruction
   (lower-elementwise-binding b combiner in-traces in-refs pool-idx shape strides dtype)
   pool-idx shape strides dtype))

(define (compile-reduction-segop rop axes keepdims? src-tr in-ref pool-idx shape strides dtype
                                 #!optional (frame-rank 0))
  "The ri-segop instruction of a reduction binding with frame rank
   frame-rank (see morph-reduce)."
  (or (segop-instruction
       (lower-reduction-binding rop axes keepdims? src-tr in-ref pool-idx shape strides dtype
                                frame-rank)
       pool-idx shape strides dtype)
      (error "compile-reduction-segop: no backend accepts the reduction" rop axes)))

(define (compile-affine-segop fn src-tr in-ref pool-idx shape strides dtype)
  "The ri-segop instruction materializing affine index function fn, or #f."
  (segop-instruction (lower-affine-binding fn src-tr in-ref pool-idx shape strides dtype)
                     pool-idx shape strides dtype))

(define (compile-matmul-segop A-tr B-tr A-ref B-ref pool-idx shape strides dtype
                             #!optional (epilogue #f))
  "The ri-segop instruction of a matmul binding, with the element-wise
   binding whose seg-map is epilogue fused into it when epilogue is given;
   #f when the matmul cannot be lowered or no backend accepts its segop."
  (segop-instruction
   (lower-matmul-binding A-tr B-tr A-ref B-ref pool-idx shape strides dtype epilogue)
   pool-idx shape strides dtype))

(define (compile-batch-matmul-segop A-tr B-tr A-ref B-ref pool-idx shape strides dtype)
  "The ri-segop instruction of a batch-matmul binding; #f when the product
   cannot be lowered or no backend accepts its segop."
  (segop-instruction
   (lower-batch-matmul-binding A-tr B-tr A-ref B-ref pool-idx shape strides dtype)
   pool-idx shape strides dtype))

(define (compile-conv-segop op meta in-traces in-refs pool-idx shape strides dtype)
  "The ri-segop instruction of a convolution binding with op op and meta
   meta (see lower-conv-binding)."
  (or (segop-instruction (lower-conv-binding op meta in-traces in-refs pool-idx shape strides dtype)
                         pool-idx shape strides dtype)
      (error "compile-conv-segop: no backend accepts the convolution" op)))

(define (compile-host-segop fn in-traces in-refs pool-idx shape strides dtype)
  "The ri-segop instruction evaluating index function fn on the host."
  (or (segop-instruction (lower-host-index-fn fn in-traces in-refs pool-idx shape strides dtype)
                         pool-idx shape strides dtype)
      (error "compile-host-segop: no backend accepts the index function" fn)))
