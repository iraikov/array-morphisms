;;; lift.scm
;;; Frame lifting: apply a function to every cell of a frame as one
;;; computation.
;;;
;;; (morph-lift fn cell-ranks arg ...) calls fn once, on placeholder arrays
;;; with the cell shapes of the arguments, and rebuilds the morphism fn
;;; returns with every node that depends on an argument widened by the
;;; argument's frame.  The result is an ordinary morphism: an element-wise
;;; body stays one element-wise node, a reduction becomes one reduction
;;; with the frame as extra kept axes, and a matrix product becomes one
;;; batched matrix product, so the tree has as many nodes for a frame of
;;; any size.  Each cell of the result has the values fn gives for that
;;; cell on its own.
;;;
;;; The same tracing turns a procedure of two morphisms, the step of a
;;; scan, into a combine expression (scan-combine-expression), so that
;;; (morph-scan/fn fn m axis ...) scans with it as one node.

(module array-morphisms-lift

  (morph-lift
   lift-unsupported?
   lift-unsupported-op
   scan-combine-expression
   morph-scan/fn)

  (import scheme (chicken base) (chicken condition))
  (import (only srfi-1 iota every any fold take drop make-list list-index))
  (import srfi-69)
  (import datatype)
  (import array-morphisms-core)
  (import array-morphisms-basic-ops)
  (import array-morphisms-structural-ops)
  (import array-morphisms-blas-exec)
  (import (only array-morphisms-scalar-ops scalar-op? scalar-op-arity scalar-op-combiner))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Conditions
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;; morph-lift signals a lift-unsupported condition (also an exn) when fn
  ;; uses an op that has no widening rule, or uses one in a way the rule
  ;; does not cover.
  (define (lift-unsupported op why)
    (abort (make-composite-condition
            (make-property-condition 'exn 'location 'morph-lift
                                     'message (string-append "cannot lift op: " why)
                                     'arguments (list op))
            (make-property-condition 'lift-unsupported 'op op))))

  (define lift-unsupported? (condition-predicate 'lift-unsupported))

  (define (lift-unsupported-op c)
    "The op named by lift-unsupported condition c."
    (get-condition-property c 'lift-unsupported 'op))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Shapes
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (shape-of m) (vector->list (get-morphism-shape m)))
  (define (rank-of m) (vector-length (get-morphism-shape m)))
  (define (product lst) (fold * 1 lst))

  ;; A widened value: a morphism whose first fr axes are a frame (a prefix
  ;; of the principal frame), followed by the cell axes.  fr = 0 marks a
  ;; value that does not depend on any argument.
  (define (wv m fr) (cons m fr))
  (define (wv-m v) (car v))
  (define (wv-fr v) (cdr v))
  (define (wv-frame v) (take (shape-of (wv-m v)) (wv-fr v)))
  (define (wv-cell v) (drop (shape-of (wv-m v)) (wv-fr v)))

  ;; v seen with frame rank fr2 >= its own and cell rank cr >= its own, by
  ;; inserting size-1 axes after its frame and in front of its cell.  A
  ;; value without a frame is left as it is: suffix broadcasting already
  ;; places it.
  (define (pad-axes v fr2 cr)
    (let* ((fr (wv-fr v)) (cell (wv-cell v)))
      (if (or (= fr 0) (and (= fr fr2) (= (length cell) cr)))
          (wv-m v)
          (morph-reshape (wv-m v)
                         (list->vector (append (wv-frame v)
                                               (make-list (- fr2 fr) 1)
                                               (make-list (- cr (length cell)) 1)
                                               cell))))))

  ;; v repeated along the first fr2 axes of frame, so that its frame rank
  ;; becomes fr2.  The repetition multiplies by ones, which leaves every
  ;; value, -0.0 and NaN included, as it is.
  (define (expand-frame v frame fr2)
    (if (= (wv-fr v) fr2)
        (wv-m v)
        (let* ((m     (wv-m v))
               (dt    (get-morphism-dtype m))
               (cell  (wv-cell v))
               (one   (if (memq dt '(f32 f64)) 1.0 1))
               (oshape (append (take frame fr2) (make-list (length cell) 1)))
               (ones  (morph-from-list (make-list (product oshape) one)
                                       (list->vector oshape) dt)))
          (morph* (pad-axes v fr2 (length cell)) ones))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Widening rules
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define unary-ops
    '(sqrt exp log sin cos tan negate abs floor ceiling relu sigmoid tanh))
  (define binary-ops '(add sub mul div pow))
  (define comparison-ops '(gt lt eq ge le))

  (define (meta-ref meta key)
    (let ((p (assq key meta)))
      (if p (cdr p) (error "morph-lift: missing metadata" key meta))))

  ;; The widened node for morphism-expr node with op, metadata, cell
  ;; shape and widened operands ws (at least one of which has a frame).
  ;; frame is the principal frame.
  (define (widen-expr op node meta shape dtype ws frame)
    (let* ((frs   (map wv-fr ws))
           (fr    (fold max 0 frs))
           (frame-prefix (take frame fr)))
      (define (aligned)
        (let ((cr (fold max 0 (map (lambda (v) (length (wv-cell v))) ws))))
          (map (lambda (v) (pad-axes v fr cr)) ws)))
      (define (expanded) (map (lambda (v) (expand-frame v frame fr)) ws))
      (define (one) (car ws))
      (cond
       ((memq op unary-ops)
        (wv (create-unary-morphism op (wv-m (one))) fr))
       ((memq op binary-ops)
        (wv (apply create-binary-morphism op (aligned)) fr))
       ((memq op comparison-ops)
        (wv (apply create-comparison-morphism op (aligned)) fr))
       (else
        (case op
          ((cast) (wv (morph-cast (wv-m (one)) dtype) fr))
          ((map)  (wv (morph-map (meta-ref meta 'function) (wv-m (one))) fr))
          ((transpose)
           (wv (morph-transpose (wv-m (one))
                                (append (iota fr) (map (lambda (p) (+ p fr))
                                                       (meta-ref meta 'permutation))))
               fr))
          ((reshape)
           (wv (morph-reshape (wv-m (one))
                              (list->vector (append frame-prefix (vector->list shape))))
               fr))
          ((slice)
           (wv (morph-slice (wv-m (one))
                            (append (make-list fr 0) (meta-ref meta 'start))
                            (append frame-prefix (meta-ref meta 'end))
                            (append (make-list fr 1) (meta-ref meta 'step)))
               fr))
          ((pad)
           (wv (morph-pad (wv-m (one))
                          (append (make-list fr '(0 0)) (meta-ref meta 'padding))
                          (meta-ref meta 'mode) (meta-ref meta 'value))
               fr))
          ((stack)
           (wv (morph-stack (expanded) (+ fr (meta-ref meta 'axis))) fr))
          ((concat)
           (wv (morph-concat (expanded) (+ fr (meta-ref meta 'axis))) fr))
          ((matmul batch-matmul)
           (let ((a (car ws)) (b (cadr ws)))
             (wv (condition-case
                     (if (and (> (wv-fr a) 0) (> (wv-fr b) 0))
                         (apply morph-batch-matmul (expanded))
                         (morph-batch-matmul (wv-m a) (wv-m b)))
                   ((exn) (lift-unsupported op "operand frames do not combine")))
                 fr)))
          ((attention)
           (let ((qkv (expanded)))
             (unless (<= (rank-of (car qkv)) 4)
               (lift-unsupported op "rank above 4"))
             (wv (attention-morphism (car qkv) (cadr qkv) (caddr qkv) (meta-ref meta 'scale))
                 fr)))
          ((scan)
           (let* ((e (expanded)) (p (lambda (k) (meta-ref meta k)))
                  (k (let ((c (assq 'aux-count meta))) (if c (cdr c) 0))))
             (wv (morph-scan (p 'scan-op) (car e) (+ fr (p 'axis))
                             exclusive?: (p 'exclusive?) reverse?: (p 'reverse?)
                             swapped?: (p 'swapped?)
                             neutral: (let ((n (assq 'neutral meta))) (and n (cdr n)))
                             init: (and (= (length e) (+ 2 k)) (cadr e))
                             aux: (list-tail e (- (length e) k)))
                 fr)))
          ((im2col) (widen-im2col node meta ws fr frame-prefix))
          ((col2im) (widen-col2im node meta shape ws fr frame-prefix))
          (else (lift-unsupported op "no widening rule")))))))

  ;; im2col of a framed input: the frame is folded into the image axis N
  ;; (or becomes it for a single image) by a reshape before and after.
  (define (widen-im2col node meta ws fr frame-prefix)
    (let* ((x     (car ws))
           (cell  (wv-cell x))
           (n     (product frame-prefix))
           (batch (if (= (length cell) 4) (car cell) 1))
           (img   (if (= (length cell) 4) (cdr cell) cell))
           (layout (let ((p (assq 'layout meta))) (if p (cdr p) 'nchw)))
           (x4    (morph-reshape (wv-m x) (list->vector (cons (* n batch) img))))
           (col   (im2col-morph x4 (meta-ref meta 'kernel-size) (meta-ref meta 'stride)
                                (meta-ref meta 'padding) layout: layout))
           (cs    (cdr (shape-of col))))
      (wv (morph-reshape col (list->vector (append frame-prefix
                                                   (if (= (length cell) 4) (list batch) '())
                                                   cs)))
          fr)))

  ;; col2im of a framed column array, folded into the image axis likewise.
  (define (widen-col2im node meta shape ws fr frame-prefix)
    (let* ((c     (car ws))
           (cell  (wv-cell c))
           (n     (product frame-prefix))
           (target (vector->list shape))
           (batched? (= (length target) 4))
           (_ (unless (= (length cell) (if batched? 3 2))
                (lift-unsupported 'col2im "column layout")))
           (batch (if batched? (car cell) 1))
           (img   (if batched? (cdr target) target))
           (layout (let ((p (assq 'layout meta))) (if p (cdr p) 'nchw)))
           (c3    (morph-reshape (wv-m c) (list->vector (cons (* n batch)
                                                              (if batched? (cdr cell) cell)))))
           (x     (col2im-morph c3 (list->vector (cons (* n batch) img))
                                (meta-ref meta 'kernel-size) (meta-ref meta 'stride)
                                (meta-ref meta 'padding) layout: layout)))
      (wv (morph-reshape x (list->vector (append frame-prefix target))) fr)))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; morph-lift
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (morph-lift fn cell-ranks . args)
    "Apply fn to every cell of the frame of the arguments as one
    computation.

    cell-ranks is the rank of the cells fn takes, one per argument, or one
    for all.  The frame of argument k is its leading rank_k - cell-rank_k
    axes.  The principal frame is the longest frame; every other frame must
    be a prefix of it, and an argument with a shorter frame is used
    unchanged along the frame axes it lacks.  The result has the shape of
    the principal frame followed by the shape fn returns, and each of its
    cells holds what fn gives for the corresponding cells of the
    arguments.

    fn is called once, on placeholder arrays with the cell shapes and
    dtypes, whose elements must not be read: fn may only build morphisms
    from them.  Nodes of fn's result that depend on no argument are used
    as they are and computed once.  Signals a lift-unsupported condition
    when fn uses an op that cannot be widened."
    (let* ((n      (length args))
           (ranks  (cond ((and (integer? cell-ranks) (exact? cell-ranks)) (make-list n cell-ranks))
                         ((and (list? cell-ranks) (= (length cell-ranks) n)) cell-ranks)
                         (else (error "morph-lift: one cell rank per argument, or one for all"
                                      cell-ranks n))))
           (frs    (map (lambda (a r)
                          (unless (array-morphism? a)
                            (error "morph-lift: argument is not a morphism" a))
                          (let ((fr (- (rank-of a) r)))
                            (unless (and (>= r 0) (>= fr 0))
                              (error "morph-lift: cell rank out of range" r (rank-of a)))
                            fr))
                        args ranks))
           (frames (map (lambda (a fr) (take (shape-of a) fr)) args frs))
           (frame  (fold (lambda (f best) (if (> (length f) (length best)) f best)) '() frames)))
      (unless (every (lambda (f) (equal? f (take frame (length f)))) frames)
        (error "morph-lift: every frame must be a prefix of the longest" frames))
      (if (null? frame)
          (apply fn args)
          (let* ((holders (map (lambda (a fr)
                                 (let ((cell (list->vector (drop (shape-of a) fr)))
                                       (dt   (get-morphism-dtype a)))
                                   (concrete-array (allocate-typed-vector dt 0) cell
                                                   (compute-strides cell) 0 dt -1 -1)))
                               args frs))
                 (traced  (apply fn holders))
                 (memo    (make-hash-table string=? string-hash)))
            (define (walk m)
              (cases array-morphism m
                (concrete-array (_ _ _ _ _ _ _)
                  (let ((k (list-index (lambda (h) (eq? h m)) holders)))
                    (if k (wv (list-ref args k) (list-ref frs k)) (wv m 0))))
                (morphism-expr (id op operands _ shape dtype meta _)
                  (memoized id
                    (lambda ()
                      (let ((ws (map walk operands)))
                        (if (every (lambda (v) (= (wv-fr v) 0)) ws)
                            (wv m 0)
                            (widen-expr op m meta shape dtype ws frame))))))
                (reduction-morphism (id op operand axes rfn shape dtype _)
                  (memoized id
                    (lambda ()
                      (let ((w (walk operand)))
                        (if (= (wv-fr w) 0)
                            (wv m 0)
                            (let ((fr (wv-fr w)))
                              (wv (morph-reduce op (wv-m w) (map (lambda (a) (+ a fr)) axes)
                                                (reduction-index-fn-keepdims? rfn)
                                                (+ fr (reduction-index-fn-frame-rank rfn)))
                                  fr)))))))))
            (define (memoized id thunk)
              (let ((key (symbol->string id)))
                (or (hash-table-ref/default memo key #f)
                    (let ((v (thunk))) (hash-table-set! memo key v) v))))
            (unless (array-morphism? traced)
              (error "morph-lift: fn must return a morphism" traced))
            (expand-frame (walk traced) frame (length frame))))))

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;;; Scans with a procedure as combine op
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  (define (scan-combine-expression fn dtype)
    "The combine expression (lambda (a b) e) of fn, a procedure of two
    morphisms (the value so far and the next element) that builds its
    result from them with element-wise table ops (add, mul, exp, ...) and
    one-element constant arrays, every node of dtype dtype.  fn is called
    once, on placeholder arrays of one element.  Each intermediate node is
    rounded as storing it in an array of dtype would round it (a cast in
    e), so that a scan with the expression gives the values of applying fn
    step by step.  Signals a lift-unsupported condition naming the first
    node that does not fit."
    (let* ((holders (map (lambda (k) (concrete-array (allocate-typed-vector dtype 0) (vector 1)
                                                     (vector 1) 0 dtype -1 -1))
                         '(0 1)))
           (traced (apply fn holders)))
      (define (stored e) (if (eq? dtype 'f64) e `(cast ,dtype ,e)))
      (define (walk m root?)
        (cases array-morphism m
          (concrete-array (data shape strides offset dt _ _)
            (cond ((eq? m (car holders)) 'a)
                  ((eq? m (cadr holders)) 'b)
                  ((and (= (shape-size shape) 1) (eq? dt dtype))
                   (typed-vector-ref data dt offset))
                  (else (lift-unsupported 'scan "a captured array that is not a scalar of the scan's dtype"))))
          (morphism-expr (id op operands ifn shape dtype2 meta _)
            (unless (and (symbol? op) (scalar-op? op)
                         (= (scalar-op-arity op) (length operands))
                         (compute-index-fn? ifn)
                         (eq? (compute-index-fn-combiner ifn) (scalar-op-combiner op)))
              (lift-unsupported op "not an element-wise table op in a scan step"))
            (unless (eq? dtype2 dtype)
              (lift-unsupported op "a scan step node of another dtype"))
            (let ((e (cons op (map (lambda (o) (walk o #f)) operands))))
              (if root? e (stored e))))
          (else (lift-unsupported 'scan "a reduction in a scan step"))))
      (unless (array-morphism? traced)
        (error "scan-combine-expression: fn must return a morphism" traced))
      `(lambda (a b) ,(walk traced #t))))

  (define (morph-scan/fn fn m axis #!key (exclusive? #f) (reverse? #f) (init #f) (neutral #f)
                         (swapped? #f))
    "Scan m along axis with the procedure fn of two morphisms as combine op:
    (morph-scan (scan-combine-expression fn dtype) m axis ...), dtype being
    m's, or the standard op sum or prod when fn adds or multiplies its two
    arguments.  Each output is fn applied to the previous output and the next
    element, with the rounding of fn's element-wise ops; the keywords are
    those of morph-scan.  Signals a lift-unsupported condition when fn is
    not made of element-wise table ops (see scan-combine-expression)."
    (let* ((e  (scan-combine-expression fn (get-morphism-dtype m)))
           (op (and (not neutral) (standard-scan-op e))))
      (if op
          (morph-scan (car op) m axis exclusive?: exclusive? reverse?: reverse? init: init
                      swapped?: (if swapped? (not (cdr op)) (cdr op)))
          (morph-scan e m axis exclusive?: exclusive? reverse?: reverse? init: init neutral: neutral
                      swapped?: swapped?))))

  ;; (op . swapped?) when combine expression e is the step of the standard
  ;; scan op sum or prod, (add a b) or (mul a b), or of one with its
  ;; operands trading places; #f otherwise.  The standard op's step is the
  ;; combiner of the same table op, so the values are the same, and the scan
  ;; then has the gradient rule and the kernels of the standard op.
  (define (standard-scan-op e)
    (let ((body (caddr e)))
      (and (pair? body) (= (length body) 3)
           (memq (car body) '(add mul))
           (let ((op (if (eq? (car body) 'add) 'sum 'prod)))
             (cond ((equal? (cdr body) '(a b)) (cons op #f))
                   ((equal? (cdr body) '(b a)) (cons op #t))
                   (else #f))))))

) ;; end module array-morphisms-lift
