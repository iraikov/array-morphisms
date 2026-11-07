;;; segop.scm
;;;
;;; Segmented operations (segops): a small intermediate representation for
;;; array computations, between SSA bindings and their execution.  A segop
;;; describes a computation by
;;;
;;;   * an iteration space, a vector of extents;
;;;   * operands, each an array read or written through an index map from
;;;     iteration coordinates to the array's multi-index;
;;;   * a scalar body, a symbolic expression computing one output element
;;;     from the input elements at the same iteration point;
;;;   * for reductions, a combine spec giving the combining operator, its
;;;     neutral element, the accumulator precision and the summation order.
;;;
;;; The bodies refer to the ops of the scalar-op table by name, so a code
;;; generator can read them.  A segop also keeps the Scheme combiner it was
;;; lowered from, which host execution uses.
;;;
;;; The module provides the data types, operations on index maps and
;;; scalar expressions, a reference interpreter (slow, meant for tests),
;;; and a registry of segop backends that turn segops into executors.
;;;
;;; Index expressions
;;;
;;;   ie ::= n | (dim d) | (+ ie ie) | (* n ie) | (floordiv ie n) | (mod ie n)
;;;
;;; n is an exact integer and d an iteration dimension.  floordiv and mod
;;; round towards minus infinity.
;;;
;;; Scalar expressions (segop bodies)
;;;
;;;   e ::= number | v                  literal, let*-bound variable
;;;       | (in k)                      element of input k at the current point
;;;       | (param k)                   runtime scalar parameter k
;;;       | (idx d)                     iteration coordinate d, an integer
;;;       | (reduced)                   the reduced value, in a finalize body
;;;       | (op e ...)                  a primitive (see scalar-primitives) or
;;;                                     an op of the scalar-op table
;;;       | (if e e e) | (let* ((v e) ...) e)
;;;       | (cast dtype e)              e rounded as storing it in an array of
;;;                                     dtype would round it
;;;       | (opaque proc e ...)         a Scheme procedure, host only
;;;
;;; Bodies compute in double precision; a result is rounded when it is
;;; stored to a single-precision output, or where a cast says so.

(module array-morphisms-segop

  (;; Index expressions
   index-expr?
   eval-index-expr
   simplify-index-expr
   index-expr-dims

   ;; Index maps
   make-index-map
   index-map?
   index-map-exprs index-map-guard index-map-fill index-map-proc
   index-map-rank
   index-map-opaque?
   identity-index-map
   broadcast-index-map
   permutation-index-map
   affine-index-fn->index-map
   reshape-index-map
   opaque-index-map
   offset-index-map
   apply-index-map
   index-map-dims
   index-map->linear
   compose-index-maps
   linear-injective?

   ;; Operands
   make-segop-operand
   segop-operand?
   segop-operand-slot segop-operand-dtype segop-operand-shape
   segop-operand-strides segop-operand-offset segop-operand-map
   segop-operand-linear

   ;; Scalar expressions
   scalar-expr-valid?
   eval-scalar-expr
   scalar-expr-substitute-inputs
   scalar-expr-substitute-reduced
   scalar-expr-inputs
   scalar-expr-opaque?
   scalar-expr->procedure
   scalar-exprs->procedure
   canonical-scalar-body

   ;; Reductions
   make-combine-spec
   combine-spec?
   combine-spec-op combine-spec-neutral combine-spec-acc-dtype
   combine-spec-order combine-spec-block-size
   make-red-result
   red-result?
   red-result-out red-result-body red-result-combine red-result-finalize
   combine-step round-to-dtype

   ;; Segops
   segop segop?
   seg-map seg-red seg-lib
   make-seg-map make-seg-red make-seg-lib
   segop-kind
   segop-space
   segop-inputs
   segop-outputs
   segop-fallback
   segop-interpret

   ;; Query templates
   make-segop-template
   segop-template?
   segop-template-space segop-template-ins segop-template-outs segop-template-nred
   make-template-input
   template-input?
   template-input-dtype template-input-map template-input-flat?
   make-template-output
   template-output?
   template-output-dtype template-output-body
   template-output-combine template-output-finalize template-output-map

   ;; Backends
   make-segop-backend
   segop-backend?
   segop-backend-name
   segop-backend-select
   segop-backend-kind
   segop-backend-native?
   segop-native?
   register-segop-backend!
   unregister-segop-backend!
   segop-backends
   select-segop-executor

   ;; Plan compilation
   make-segop-plan-context
   segop-plan-context?
   segop-plan-context
   segop-plan-add-finalizer!
   segop-plan-property
   segop-plan-property-set!
   segop-plan-finalize!
   make-segop-executor-cell
   segop-executor-cell?
   segop-executor-cell-ref
   segop-executor-cell-set!
   segop-executor-cell-executor)

  (import scheme (chicken base) (chicken format)
          (only (scheme base) exact-integer? vector-map vector-copy vector-append make-parameter)
          (only srfi-1 every any iota filter delete-duplicates fold list-index)
          (only srfi-4 make-f32vector f32vector-ref f32vector-set!)
          (only srfi-69 make-hash-table hash-table-ref/default hash-table-set!)
          datatype
          array-morphisms-core
          array-morphisms-scalar-ops)


;;;; ============================================================
;;;; Index expressions
;;;; ============================================================

(define (index-expr? ie)
  "True when ie is a well-formed index expression."
  (cond ((exact-integer? ie) #t)
        ((not (and (pair? ie) (list? ie))) #f)
        (else
         (case (car ie)
           ((dim) (and (= (length ie) 2) (exact-integer? (cadr ie)) (>= (cadr ie) 0)))
           ((+) (and (= (length ie) 3) (index-expr? (cadr ie)) (index-expr? (caddr ie))))
           ((*) (and (= (length ie) 3) (exact-integer? (cadr ie)) (index-expr? (caddr ie))))
           ((floordiv mod)
            (and (= (length ie) 3) (index-expr? (cadr ie))
                 (exact-integer? (caddr ie)) (> (caddr ie) 0)))
           (else #f)))))

(define (floordiv a n) (quotient (- a (modulo a n)) n))

(define (eval-index-expr ie coords)
  "Value of index expression ie at the iteration point coords, a vector of
   exact integers."
  (cond ((exact-integer? ie) ie)
        (else
         (case (car ie)
           ((dim) (vector-ref coords (cadr ie)))
           ((+) (+ (eval-index-expr (cadr ie) coords) (eval-index-expr (caddr ie) coords)))
           ((*) (* (cadr ie) (eval-index-expr (caddr ie) coords)))
           ((floordiv) (floordiv (eval-index-expr (cadr ie) coords) (caddr ie)))
           ((mod) (modulo (eval-index-expr (cadr ie) coords) (caddr ie)))
           (else (error 'eval-index-expr "malformed index expression" ie))))))

(define (index-expr-dims ie)
  "Sorted list of the iteration dimensions ie refers to."
  (define (walk ie acc)
    (cond ((exact-integer? ie) acc)
          ((eq? (car ie) 'dim) (if (memv (cadr ie) acc) acc (cons (cadr ie) acc)))
          ((eq? (car ie) '*) (walk (caddr ie) acc))
          ((eq? (car ie) '+) (walk (caddr ie) (walk (cadr ie) acc)))
          (else (walk (cadr ie) acc))))
  (sort-integers (walk ie '())))

(define (sort-integers lst)
  (let insert-all ((lst lst) (out '()))
    (if (null? lst)
        out
        (insert-all (cdr lst)
                    (let ins ((x (car lst)) (out out))
                      (cond ((null? out) (list x))
                            ((< x (car out)) (cons x out))
                            (else (cons (car out) (ins x (cdr out))))))))))

;; The affine form of ie: a pair (c . terms), terms an alist d -> coefficient
;; with no zero coefficients, such that ie = c + sum coef * (dim d); or #f
;; when ie contains a floordiv or mod that does not reduce to a constant.
(define (ie->affine ie)
  (define (add a b)
    (cons (+ (car a) (car b))
          (fold (lambda (t acc)
                  (let ((p (assv (car t) acc)))
                    (if p
                        (let ((c (+ (cdr p) (cdr t))))
                          (if (zero? c)
                              (filter (lambda (u) (not (eqv? (car u) (car t)))) acc)
                              (map (lambda (u) (if (eqv? (car u) (car t)) (cons (car u) c) u)) acc)))
                        (cons t acc))))
                (cdr a) (cdr b))))
  (define (scale k a)
    (if (zero? k)
        (cons 0 '())
        (cons (* k (car a)) (map (lambda (t) (cons (car t) (* k (cdr t)))) (cdr a)))))
  (cond ((exact-integer? ie) (cons ie '()))
        (else
         (case (car ie)
           ((dim) (cons 0 (list (cons (cadr ie) 1))))
           ((+) (let ((a (ie->affine (cadr ie))) (b (ie->affine (caddr ie))))
                  (and a b (add a b))))
           ((*) (let ((a (ie->affine (caddr ie)))) (and a (scale (cadr ie) a))))
           ((floordiv mod)
            (let ((a (ie->affine (cadr ie))) (n (caddr ie)))
              (cond ((and a (null? (cdr a)))
                     (cons (if (eq? (car ie) 'floordiv) (floordiv (car a) n) (modulo (car a) n)) '()))
                    ((and a (= n 1) (eq? (car ie) 'floordiv)) a)
                    ((and a (= n 1) (eq? (car ie) 'mod)) (cons 0 '()))
                    (else #f))))
           (else (error 'ie->affine "malformed index expression" ie))))))

(define (affine->ie a)
  (let* ((terms (sort-terms (cdr a)))
         (term-ies (map (lambda (t)
                          (if (= (cdr t) 1) `(dim ,(car t)) `(* ,(cdr t) (dim ,(car t)))))
                        terms)))
    (cond ((null? term-ies) (car a))
          (else
           (let ((sum (fold (lambda (t acc) `(+ ,acc ,t)) (car term-ies) (cdr term-ies))))
             (if (zero? (car a)) sum `(+ ,sum ,(car a))))))))

(define (sort-terms terms)
  (map (lambda (d) (assv d terms)) (sort-integers (map car terms))))

(define (simplify-index-expr ie)
  "An equivalent, simpler form of ie.  An affine expression becomes a sum of
   (* coefficient (dim d)) terms in increasing d plus a constant; floordiv
   and mod with constant arguments are folded."
  (let ((a (ie->affine ie)))
    (cond (a (affine->ie a))
          ((memq (car ie) '(floordiv mod))
           `(,(car ie) ,(simplify-index-expr (cadr ie)) ,(caddr ie)))
          ((eq? (car ie) '+)
           `(+ ,(simplify-index-expr (cadr ie)) ,(simplify-index-expr (caddr ie))))
          ((eq? (car ie) '*)
           (if (= (cadr ie) 1)
               (simplify-index-expr (caddr ie))
               `(* ,(cadr ie) ,(simplify-index-expr (caddr ie)))))
          (else ie))))

;; ie with every (dim d) replaced by (vector-ref subst d).
(define (substitute-dims ie subst)
  (cond ((exact-integer? ie) ie)
        ((eq? (car ie) 'dim)
         (if (< (cadr ie) (vector-length subst))
             (vector-ref subst (cadr ie))
             (error 'compose-index-maps "dimension out of range" ie)))
        ((eq? (car ie) '*) `(* ,(cadr ie) ,(substitute-dims (caddr ie) subst)))
        ((eq? (car ie) '+) `(+ ,(substitute-dims (cadr ie) subst) ,(substitute-dims (caddr ie) subst)))
        (else `(,(car ie) ,(substitute-dims (cadr ie) subst) ,(caddr ie)))))


;;;; ============================================================
;;;; Index maps
;;;; ============================================================

;; exprs: vector of index expressions, one per operand dimension, or #f
;;        for an opaque map.
;; guard: #f, or a list of (ie lo hi); the read is valid when
;;        lo <= ie < hi holds for every entry, and gives fill otherwise.
;; fill:  value read where the guard fails.
;; proc:  for an opaque map, a procedure from the coordinate vector to the
;;        operand's index vector (or #f for a padded read); otherwise #f.
(define-record index-map exprs guard fill proc)

(define (index-map-opaque? m)
  "True when m is an opaque map (a Scheme procedure)."
  (and (index-map-proc m) #t))

(define (index-map-rank m)
  "Rank of the operand m indexes, or #f for an opaque map."
  (and (index-map-exprs m) (vector-length (index-map-exprs m))))

(define (identity-index-map rank)
  "The map reading the operand at the iteration point itself."
  (make-index-map (list->vector (map (lambda (d) `(dim ,d)) (iota rank))) #f 0.0 #f))

(define (broadcast-index-map in-shape out-rank)
  "The map of NumPy broadcasting from an iteration space of rank out-rank to
   an operand of shape in-shape: dimensions are aligned on the right, and an
   operand dimension of extent 1 is read at index 0.  An operand of higher
   rank than the space is allowed when its extra leading dimensions have
   extent 1; they too are read at index 0."
  (let* ((k (vector-length in-shape))
         (extra (- k out-rank)))
    (when (> extra 0)
      (let loop ((j 0))
        (when (< j extra)
          (unless (= (vector-ref in-shape j) 1)
            (error 'broadcast-index-map "operand rank exceeds the space rank" in-shape out-rank))
          (loop (+ j 1)))))
    (make-index-map
     (list->vector
      (map (lambda (j)
             (if (= (vector-ref in-shape j) 1) 0 `(dim ,(+ j (- out-rank k)))))
           (iota k)))
     #f 0.0 #f)))

(define (permutation-index-map perm)
  "The map whose operand index k is iteration coordinate (list-ref perm k),
   the index function of a transpose."
  (make-index-map (list->vector (map (lambda (p) `(dim ,p)) perm)) #f 0.0 #f))

(define (offset-index-map rank offsets)
  "The identity map shifted by offsets, a list with one integer per
   dimension: operand index k is coordinate k plus (list-ref offsets k)."
  (make-index-map (list->vector (map (lambda (d o) (simplify-index-expr `(+ (dim ,d) ,o)))
                                     (iota rank) offsets))
                  #f 0.0 #f))

(define (affine-index-fn->index-map fn out-shape in-shape)
  "The index map of the affine index function fn (identity-fn,
   permutation-fn, diagonal-fn or general-fn) from a space of shape
   out-shape to an operand of shape in-shape.  identity-fn reinterprets the
   operand's elements in row-major order: it is the identity map when the
   shapes are equal and the reshape map otherwise."
  (define (affine-row coefs bias)
    (simplify-index-expr
     (fold (lambda (c d acc) (if (zero? c) acc `(+ ,acc (* ,c (dim ,d)))))
           bias coefs (iota (length coefs)))))
  (cases affine-index-fn fn
    (identity-fn ()
      (if (equal? out-shape in-shape)
          (identity-index-map (vector-length out-shape))
          (reshape-index-map out-shape in-shape)))
    (permutation-fn (perm) (permutation-index-map perm))
    (diagonal-fn (diag bias)
      (make-index-map
       (list->vector (map (lambda (d s b) (simplify-index-expr `(+ (* ,s (dim ,d)) ,b)))
                          (iota (length diag)) diag bias))
       #f 0.0 #f))
    (general-fn (matrix bias)
      (make-index-map
       (list->vector (map (lambda (row k) (affine-row row (if bias (list-ref bias k) 0)))
                          matrix (iota (length matrix))))
       #f 0.0 #f))))

(define (reshape-index-map out-shape in-shape)
  "The map of a reshape: iteration point i of a space of shape out-shape
   reads the element of an operand of shape in-shape at the same row-major
   position.  The operand index is formed with floordiv and mod, so the map
   is not affine in general."
  (unless (= (shape-size out-shape) (shape-size in-shape))
    (error 'reshape-index-map "shapes have different sizes" out-shape in-shape))
  (let* ((out-strides (compute-strides out-shape))
         (in-strides  (compute-strides in-shape))
         (linear (simplify-index-expr
                  (fold (lambda (d acc) `(+ ,acc (* ,(vector-ref out-strides d) (dim ,d))))
                        0 (iota (vector-length out-shape))))))
    (make-index-map
     (list->vector
      (map (lambda (k)
             (let ((q (simplify-index-expr `(floordiv ,linear ,(vector-ref in-strides k)))))
               (if (= k 0) q (simplify-index-expr `(mod ,q ,(vector-ref in-shape k))))))
           (iota (vector-length in-shape))))
     #f 0.0 #f)))

(define (opaque-index-map proc)
  "A map given by proc, a procedure from the coordinate vector to the
   operand's index vector, or to #f for a read that gives 0.0.  Opaque maps
   can only be executed on the host."
  (make-index-map #f #f 0.0 proc))

(define (guard-holds? guard coords)
  (every (lambda (g)
           (let ((v (eval-index-expr (car g) coords)))
             (and (<= (cadr g) v) (< v (caddr g)))))
         guard))

(define (apply-index-map m coords)
  "The operand index vector m gives at the iteration point coords, or #f
   when the map's guard fails there."
  (cond ((index-map-proc m) ((index-map-proc m) coords))
        ((and (index-map-guard m) (not (guard-holds? (index-map-guard m) coords))) #f)
        (else (vector-map (lambda (ie) (eval-index-expr ie coords)) (index-map-exprs m)))))

(define (index-map-dims m)
  "Sorted list of the iteration dimensions the map refers to, including
   those of its guard; #f for an opaque map."
  (and (not (index-map-proc m))
       (sort-integers
        (delete-duplicates
         (append (apply append (map index-expr-dims (vector->list (index-map-exprs m))))
                 (apply append (map (lambda (g) (index-expr-dims (car g)))
                                    (or (index-map-guard m) '()))))))))

(define (index-map->linear m strides offset space-rank)
  "The position in the operand's buffer of the element that index map m
   reads at each iteration point, as a linear function of the point.

   m:          the index map, from the iteration space to the operand's
               index
   strides:    the operand's strides, a vector with one entry per operand
               dimension
   offset:     the buffer position of the operand's element at index 0
   space-rank: the number of dimensions of the iteration space

   Returns a pair (base . coefs), coefs a vector of length space-rank, such
   that the element read at point i is at base + sum over d of
   coefs[d] * i[d].  Returns #f when m is opaque, has a guard, or has an
   index expression that is not affine (such as a floor division)."
  (and (not (index-map-proc m))
       (not (index-map-guard m))
       (let ((affs (map ie->affine (vector->list (index-map-exprs m)))))
         (and (every values affs)
              (let ((coefs (make-vector space-rank 0)))
                ;; Operand dimension k has the index c + sum of c_d * i[d]
                ;; (a is (c . ((d . c_d) ...))); it adds strides[k] * c to
                ;; base and strides[k] * c_d to coefs[d].
                (let loop ((affs affs) (k 0) (base offset))
                  (if (null? affs)
                      (cons base coefs)
                      (let ((s (vector-ref strides k)) (a (car affs)))
                        (for-each (lambda (t)
                                    (when (>= (car t) space-rank)
                                      (error 'index-map->linear "dimension out of range" (car t)))
                                    (vector-set! coefs (car t)
                                                 (+ (vector-ref coefs (car t)) (* s (cdr t)))))
                                  (cdr a))
                        (loop (cdr affs) (+ k 1) (+ base (* s (car a))))))))))))

(define (compose-index-maps f g)
  "The map reading through f at the point that g gives: (f o g)(i) =
   f(g(i)).  g maps the iteration space to the coordinates f is defined on.
   The guards of both maps are kept; when both have guards they must have
   the same fill value."
  (cond ((or (index-map-proc f) (index-map-proc g))
         (opaque-index-map
          (lambda (coords)
            (let ((mid (apply-index-map g coords)))
              (and mid (apply-index-map f mid))))))
        (else
         (let* ((subst (index-map-exprs g))
                (exprs (vector-map (lambda (ie) (simplify-index-expr (substitute-dims ie subst)))
                                   (index-map-exprs f)))
                (f-guard (map (lambda (gd)
                                (list (simplify-index-expr (substitute-dims (car gd) subst))
                                      (cadr gd) (caddr gd)))
                              (or (index-map-guard f) '())))
                (g-guard (or (index-map-guard g) '()))
                (fill (cond ((null? f-guard) (index-map-fill g))
                            ((null? g-guard) (index-map-fill f))
                            ((equal? (index-map-fill f) (index-map-fill g)) (index-map-fill f))
                            (else (error 'compose-index-maps "guards with different fill values"
                                         (index-map-fill f) (index-map-fill g)))))
                (guard (append g-guard f-guard)))
           (make-index-map exprs (and (pair? guard) guard) fill #f)))))

(define (linear-injective? coefs space)
  "Return True when the linear map with coefficient vector coefs is one-to-one on
   the iteration space of extents space.  The test is sufficient, not
   necessary: sorted by magnitude, each coefficient of a dimension of
   extent above 1 must exceed the largest offset the smaller ones reach."
  (or (any zero? (vector->list space))
      (let* ((dims (filter (lambda (d) (> (vector-ref space d) 1)) (iota (vector-length space))))
             (pairs (map (lambda (d) (cons (abs (vector-ref coefs d)) (vector-ref space d))) dims))
             (sorted (let sort ((ps pairs))
                       (if (null? ps)
                           '()
                           (let* ((m (fold (lambda (p acc) (if (< (car p) (car acc)) p acc))
                                           (car ps) ps))
                                  (rest (let rm ((ps ps))
                                          (cond ((eq? (car ps) m) (cdr ps))
                                                (else (cons (car ps) (rm (cdr ps))))))))
                             (cons m (sort rest)))))))
        (let loop ((ps sorted) (reach 0))
          (cond ((null? ps) #t)
                ((<= (caar ps) reach) #f)
                (else (loop (cdr ps) (+ reach (* (caar ps) (- (cdar ps) 1))))))))))


;;;; ============================================================
;;;; Operands
;;;; ============================================================

;; One array that a segop reads or writes.
;;   slot:    where the executor finds the array (for replay plans, a
;;            replay-ref or a pool index); not interpreted here.
;;   dtype:   element type.
;;   shape, strides, offset: the array's layout.
;;   map:     index map from iteration coordinates to the array's index.
;;   linear:  cache of index-map->linear, filled on first use.
(define-record-type segop-operand
  (%make-segop-operand slot dtype shape strides offset map linear-cache)
  segop-operand?
  (slot segop-operand-slot)
  (dtype segop-operand-dtype)
  (shape segop-operand-shape)
  (strides segop-operand-strides)
  (offset segop-operand-offset)
  (map segop-operand-map)
  (linear-cache segop-operand-linear-cache segop-operand-linear-cache-set!))

(define (make-segop-operand slot dtype shape strides offset map)
    "An operand array of the given dtype and layout, read or written through
     the index map map."
    (unless (index-map? map)
      (error 'make-segop-operand "not an index map" map))
    (unless (= (vector-length shape) (vector-length strides))
      (error 'make-segop-operand "shape and strides differ in rank" shape strides))
    (when (and (index-map-rank map) (not (= (index-map-rank map) (vector-length shape))))
      (error 'make-segop-operand "index map rank differs from operand rank"
             (index-map-rank map) shape))
    (%make-segop-operand slot dtype shape strides offset map 'unset))

(define (segop-operand-linear o space-rank)
  "The (base . coefs) form of the operand's map and layout over a space of
   rank space-rank (see index-map->linear), or #f."
  (when (eq? (segop-operand-linear-cache o) 'unset)
    (segop-operand-linear-cache-set!
     o (index-map->linear (segop-operand-map o) (segop-operand-strides o)
                          (segop-operand-offset o) space-rank)))
  (segop-operand-linear-cache o))


;;;; ============================================================
;;;; Scalar expressions
;;;; ============================================================

(define (scalar-expr-valid? e n-inputs space-rank #!key (reduced? #f))
  "True when e is a well-formed scalar expression whose (in k) refer to
   k < n-inputs and whose (idx d) refer to d < space-rank.  (reduced) is
   allowed only when reduced? is true."
  (let walk ((e e) (vars '()))
    (cond ((number? e) #t)
          ((symbol? e) (and (memq e vars) #t))
          ((not (and (pair? e) (list? e))) #f)
          (else
           (case (car e)
             ((in) (and (= (length e) 2) (exact-integer? (cadr e))
                        (<= 0 (cadr e)) (< (cadr e) n-inputs)))
             ((param) (and (= (length e) 2) (exact-integer? (cadr e)) (>= (cadr e) 0)))
             ((idx) (and (= (length e) 2) (exact-integer? (cadr e))
                         (<= 0 (cadr e)) (< (cadr e) space-rank)))
             ((reduced) (and reduced? (null? (cdr e))))
             ((if) (and (= (length e) 4) (every (lambda (x) (walk x vars)) (cdr e))))
             ((cast) (and (= (length e) 3) (memq (cadr e) cast-dtypes) (walk (caddr e) vars)))
             ((let*)
              (and (= (length e) 3) (list? (cadr e))
                   (let loop ((bs (cadr e)) (vars vars))
                     (if (null? bs)
                         (walk (caddr e) vars)
                         (and (pair? (car bs)) (symbol? (caar bs)) (= (length (car bs)) 2)
                              (walk (cadar bs) vars)
                              (loop (cdr bs) (cons (caar bs) vars)))))))
             ((opaque) (and (>= (length e) 2) (procedure? (cadr e))
                            (every (lambda (x) (walk x vars)) (cddr e))))
             (else
              (and (or (scalar-primitive? (car e))
                       (and (scalar-op? (car e))
                            (= (scalar-op-arity (car e)) (length (cdr e)))))
                   (every (lambda (x) (walk x vars)) (cdr e)))))))))

(define cast-dtypes '(f32 f64 s32 s64 u32 u64))

(define (fp-abs x) (abs (exact->inexact x)))

(define (primitive-procedure sym)
  (case sym
    ((+) scalar-add) ((-) -) ((*) *) ((/) /)
    ((exp) exp) ((log) log) ((sqrt) sqrt) ((sin) sin) ((cos) cos) ((tan) tan)
    ((floor) floor) ((ceiling) ceiling) ((expt) expt) ((fpabs) fp-abs)
    ((<) <) ((>) >) ((<=) <=) ((>=) >=) ((=) =)
    (else #f)))

(define (eval-scalar-expr e read-in #!key (params '#()) (coords '#()) (reduced #f))
  "Value of the scalar expression e.  read-in is a procedure giving the
   value of input k at the current point; params is the vector of runtime
   parameters, coords the iteration point and reduced the reduced value.  Ops
   of the scalar-op table are evaluated from their table expressions."
  (let ev ((e e) (env '()))
    (cond ((number? e) e)
          ((symbol? e)
           (let ((p (assq e env)))
             (if p (cdr p) (error 'eval-scalar-expr "unbound variable" e))))
          (else
           (case (car e)
             ((in) (read-in (cadr e)))
             ((param) (vector-ref params (cadr e)))
             ((idx) (vector-ref coords (cadr e)))
             ((reduced) (or reduced (error 'eval-scalar-expr "(reduced) outside a finalize body")))
             ((if) (if (ev (cadr e) env) (ev (caddr e) env) (ev (cadddr e) env)))
             ((cast) (round-to-dtype (ev (caddr e) env) (cadr e)))
             ((let*)
              (let loop ((bs (cadr e)) (env env))
                (if (null? bs)
                    (ev (caddr e) env)
                    (loop (cdr bs) (cons (cons (caar bs) (ev (cadar bs) env)) env)))))
             ((opaque) (apply (cadr e) (map (lambda (x) (ev x env)) (cddr e))))
             (else
              (let ((args (map (lambda (x) (ev x env)) (cdr e)))
                    (prim (primitive-procedure (car e))))
                (cond (prim (apply prim args))
                      ((scalar-op? (car e))
                       (ev (scalar-op-expr (car e))
                           (map cons (scalar-op-params (car e)) args)))
                      (else (error 'eval-scalar-expr "unknown operator" (car e)))))))))))

(define (scalar-expr-substitute-inputs e subst)
  "e with every (in k) replaced by (subst k), an expression."
  (cond ((not (pair? e)) e)
        ((eq? (car e) 'in) (subst (cadr e)))
        ((memq (car e) '(param idx reduced)) e)
        ((eq? (car e) 'cast) `(cast ,(cadr e) ,(scalar-expr-substitute-inputs (caddr e) subst)))
        ((eq? (car e) 'let*)
         `(let* ,(map (lambda (b) (list (car b) (scalar-expr-substitute-inputs (cadr b) subst)))
                      (cadr e))
            ,(scalar-expr-substitute-inputs (caddr e) subst)))
        ((eq? (car e) 'opaque)
         `(opaque ,(cadr e) ,@(map (lambda (x) (scalar-expr-substitute-inputs x subst)) (cddr e))))
        (else (cons (car e) (map (lambda (x) (scalar-expr-substitute-inputs x subst)) (cdr e))))))

(define (scalar-expr-substitute-reduced e r)
  "e with every (reduced) replaced by the expression r."
  (cond ((not (pair? e)) e)
        ((eq? (car e) 'reduced) r)
        ((memq (car e) '(in param idx)) e)
        ((eq? (car e) 'cast) `(cast ,(cadr e) ,(scalar-expr-substitute-reduced (caddr e) r)))
        ((eq? (car e) 'let*)
         `(let* ,(map (lambda (b) (list (car b) (scalar-expr-substitute-reduced (cadr b) r)))
                      (cadr e))
            ,(scalar-expr-substitute-reduced (caddr e) r)))
        ((eq? (car e) 'opaque)
         `(opaque ,(cadr e) ,@(map (lambda (x) (scalar-expr-substitute-reduced x r)) (cddr e))))
        (else (cons (car e) (map (lambda (x) (scalar-expr-substitute-reduced x r)) (cdr e))))))

(define (scalar-expr-inputs e)
  "Sorted list of the input indices k for which e contains (in k)."
  (define (walk e acc)
    (cond ((not (pair? e)) acc)
          ((eq? (car e) 'in) (if (memv (cadr e) acc) acc (cons (cadr e) acc)))
          ((memq (car e) '(param idx reduced)) acc)
          ((eq? (car e) 'cast) (walk (caddr e) acc))
          ((eq? (car e) 'let*)
           (walk (caddr e) (fold (lambda (b acc) (walk (cadr b) acc)) acc (cadr e))))
          ((eq? (car e) 'opaque) (fold walk acc (cddr e)))
          (else (fold walk acc (cdr e)))))
  (sort-integers (walk e '())))

(define (scalar-expr-opaque? e)
  "True when e contains an (opaque ...) part, so that it can only be
   evaluated on the host."
  (and (pair? e)
       (or (eq? (car e) 'opaque)
           (and (eq? (car e) 'let*)
                (or (any (lambda (b) (scalar-expr-opaque? (cadr b))) (cadr e))
                    (scalar-expr-opaque? (caddr e))))
           (and (eq? (car e) 'cast) (scalar-expr-opaque? (caddr e)))
           (and (not (memq (car e) '(in param idx reduced let* cast)))
                (any scalar-expr-opaque? (cdr e))))))


;;; ------------------------------------------------------------
;;; Body compiler
;;; ------------------------------------------------------------
;;;
;;; scalar-expr->procedure turns a body into a Scheme procedure when a plan
;;; is compiled, so that executors do not walk the expression for every
;;; element.  An op of the scalar-op table compiles to its combiner, not to
;;; its table formula: a fused body then applies exactly the procedures that
;;; the separate bindings applied, for integer as well as floating-point
;;; arguments.  A cast rounds as storing the value in an array of its dtype
;;; does (typed-vector-set!).
;;;
;;; Every compiled node is a procedure (node a b c v): a, b and c are the
;;; first three inputs when the body has at most three, and v is the vector
;;; of all inputs otherwise.  let* variables and casts use cells made when
;;; the node is compiled, so a compiled body must not be called again while
;;; one of its own calls is still running (an opaque procedure calling the
;;; same compiled body).

(define (scalar-expr->procedure body n-inputs #!key (reduced? #f))
  "A procedure of n-inputs arguments that computes body.  The body may use
   (in k), numbers, table ops, primitives, if, let*, cast and opaque; (param
   k) and (idx d) are not supported.  When reduced? is true, body is a
   finalize body that may also use (reduced), and the procedure takes the
   reduced value as one more argument, after the inputs."
  (unless (scalar-expr-valid? body n-inputs 0 reduced?: reduced?)
    (error 'scalar-expr->procedure "invalid body" body n-inputs))
  (if reduced?
      (scalar-expr->procedure (scalar-expr-substitute-reduced body `(in ,n-inputs))
                              (+ n-inputs 1))
      (compile-scalar-expr body n-inputs)))

(define (compile-scalar-expr body n-inputs)
  (arity-procedure (car (compile-scalar-nodes (list body) n-inputs #f)) n-inputs))

;; The procedure of n-inputs arguments that calls compiled node.
(define (arity-procedure node n-inputs)
  (case n-inputs
    ((0) (lambda () (node #f #f #f #f)))
    ((1) (lambda (x) (node x #f #f #f)))
    ((2) (lambda (x y) (node x y #f #f)))
    ((3) (lambda (x y z) (node x y z #f)))
    (else (lambda args
            (unless (= (length args) n-inputs)
              (error 'scalar-expr->procedure "wrong number of arguments" n-inputs args))
            (node #f #f #f (list->vector args))))))

(define (scalar-exprs->procedure bodies n-inputs)
  "A procedure of n-inputs arguments that computes every body of bodies
   and returns their values in a vector, one per body, which the next call
   reuses.  A subexpression that occurs more than once among the bodies,
   as the same object (eq?), is computed once per call."
  (for-each (lambda (body)
              (unless (scalar-expr-valid? body n-inputs 0)
                (error 'scalar-exprs->procedure "invalid body" body n-inputs)))
            bodies)
  (let* ((gen   (vector 0))
         (nodes (list->vector (compile-scalar-nodes bodies n-inputs gen)))
         (m     (vector-length nodes))
         (out   (make-vector m #f))
         (node  (lambda (a b c v)
                  (vector-set! gen 0 (+ 1 (vector-ref gen 0)))
                  (do ((j 0 (+ j 1))) ((= j m) out)
                    (vector-set! out j ((vector-ref nodes j) a b c v))))))
    (arity-procedure node n-inputs)))

(define (compile-scalar-nodes bodies n-inputs gen)
  "Compile each body of bodies into a node, a procedure (node a b c v) that
   computes the body's value (see the body compiler above).  With gen,
   a subexpression that occurs more than once among the bodies, as the
   same object, is computed once per evaluation and its value reused.

   bodies:   the scalar expressions to compile, all over the same inputs
   n-inputs: the number of inputs the bodies read
   gen:      #f to compile each body on its own; or a one-element vector
             holding a counter, which the caller increases before each
             evaluation, so that the nodes know when a value they share is
             out of date

   Returns the list of nodes, one per body.  Subexpressions that refer to
   let* variables are never shared, since the same object may then stand
   for different values."
  (let ((wide?  (> n-inputs 3))
        (counts (make-hash-table eq?))
        (shared (make-hash-table eq?)))
    ;; Counts the occurrences of each subexpression object, entering each
    ;; object's own subexpressions only the first time it is met.
    (define (count! e)
      (when (pair? e)
        (let ((k (hash-table-ref/default counts e 0)))
          (hash-table-set! counts e (+ k 1))
          (when (= k 0)
            (case (car e)
              ((in param idx reduced) #f)
              ((cast) (count! (caddr e)))
              ((opaque) (for-each count! (cddr e)))
              ((let*) #f)
              (else (for-each count! (cdr e))))))))
    ;; True when e refers to no let* variable.
    (define (closed? e)
      (cond ((symbol? e) #f)
            ((not (pair? e)) #t)
            ((eq? (car e) 'let*) #f)
            ((eq? (car e) 'cast) (closed? (caddr e)))
            ((eq? (car e) 'opaque) (every closed? (cddr e)))
            ((memq (car e) '(in param idx reduced)) #t)
            (else (every closed? (cdr e)))))
    ;; True when the value of e is computed once per evaluation.
    (define (share? e)
      (and gen (pair? e) (not (eq? (car e) 'in))
           (> (hash-table-ref/default counts e 0) 1)
           (closed? e)))
    (define (compile e env)
      (if (share? e)
          (or (hash-table-ref/default shared e #f)
              (let* ((node (compile-form e env))
                     (seen -1)
                     (val  #f)
                     ;; computes node once for each value of the counter
                     (once (lambda (a b c v)
                             (unless (eqv? seen (vector-ref gen 0))
                               (set! val (node a b c v))
                               (set! seen (vector-ref gen 0)))
                             val)))
                (hash-table-set! shared e once)
                once))
          (compile-form e env)))
    (define (compile-form e env)
      (cond
       ((number? e) (lambda (a b c v) e))
       ((symbol? e)
        (let ((cell (cdr (or (assq e env)
                             (error 'scalar-expr->procedure "unbound variable" e)))))
          (lambda (a b c v) (car cell))))
       (else
        (case (car e)
          ((in)
           (let ((k (cadr e)))
             (cond (wide? (lambda (a b c v) (vector-ref v k)))
                   ((= k 0) (lambda (a b c v) a))
                   ((= k 1) (lambda (a b c v) b))
                   (else    (lambda (a b c v) c)))))
          ((param idx reduced)
           (error 'scalar-expr->procedure "form not supported by the body compiler" e))
          ((if)
           (let ((t (compile (cadr e) env))
                 (y (compile (caddr e) env))
                 (n (compile (cadddr e) env)))
             (lambda (a b c v) (if (t a b c v) (y a b c v) (n a b c v)))))
          ((let*)
           (let loop ((bs (cadr e)) (env env) (inits '()))
             (if (null? bs)
                 (let ((body (compile (caddr e) env))
                       (inits (reverse inits)))
                   (lambda (a b c v)
                     (for-each (lambda (init) (set-car! (car init) ((cdr init) a b c v))) inits)
                     (body a b c v)))
                 (let ((cell (list #f))
                       (init (compile (cadar bs) env)))
                   (loop (cdr bs) (cons (cons (caar bs) cell) env) (cons (cons cell init) inits))))))
          ((cast) (compile-cast (cadr e) (compile (caddr e) env)))
          ((opaque) (compile-call (cadr e) (map (lambda (x) (compile x env)) (cddr e))))
          (else
           (compile-call (if (scalar-op? (car e))
                             (scalar-op-combiner (car e))
                             (primitive-procedure (car e)))
                         (map (lambda (x) (compile x env)) (cdr e))))))))
    (define (compile-call f args)
      (case (length args)
        ((0) (lambda (a b c v) (f)))
        ((1) (let ((x (car args)))
               (lambda (a b c v) (f (x a b c v)))))
        ((2) (let ((x (car args)) (y (cadr args)))
               (lambda (a b c v) (f (x a b c v) (y a b c v)))))
        (else (lambda (a b c v) (apply f (map (lambda (x) (x a b c v)) args))))))
    (define (compile-cast dtype x)
      (case dtype
        ((f64) (lambda (a b c v) (exact->inexact (x a b c v))))
        ((f32) (let ((cell (make-f32vector 1)))
                 (lambda (a b c v)
                   (f32vector-set! cell 0 (exact->inexact (x a b c v)))
                   (f32vector-ref cell 0))))
        (else (let ((cell (allocate-typed-vector dtype 1)))
                (lambda (a b c v)
                  (typed-vector-set! cell dtype 0 (x a b c v))
                  (typed-vector-ref cell dtype 0))))))
    (when gen (for-each count! bodies))
    (map (lambda (body) (compile body '())) bodies)))


;; The canonical form of a body, under which bodies that compute the same
;; function of their inputs, up to the order of the inputs, are equal.
;; Kernel catalogues are keyed by canonical bodies.

;; Table ops whose two operands may be exchanged without changing the
;; result: IEEE addition and multiplication are commutative.
(define commutative-ops '(add mul))

(define (input-ref? e) (and (pair? e) (eq? (car e) 'in)))

;; A string describing the structure of e with the input numbers left out,
;; used to order the operands of commutative ops.
(define (structure-key e)
  (cond ((input-ref? e) "i")
        ((number? e) (number->string e))
        ((symbol? e) (symbol->string e))
        ((procedure? e) "p")
        ((pair? e)
         (let loop ((xs e) (acc "("))
           (if (null? xs)
               (string-append acc ")")
               (loop (cdr xs) (string-append acc " " (structure-key (car xs)))))))
        (else "?")))

;; e with commutative operands in canonical order and directly nested casts
;; to the same dtype merged, from the leaves up.
(define (normalize-scalar-expr e)
  (cond ((not (pair? e)) e)
        ((memq (car e) '(in param idx reduced)) e)
        ((eq? (car e) 'cast)
         (let ((x (normalize-scalar-expr (caddr e))))
           (if (and (pair? x) (eq? (car x) 'cast) (eq? (cadr x) (cadr e)))
               x
               `(cast ,(cadr e) ,x))))
        ((eq? (car e) 'let*)
         `(let* ,(map (lambda (b) (list (car b) (normalize-scalar-expr (cadr b)))) (cadr e))
            ,(normalize-scalar-expr (caddr e))))
        ((eq? (car e) 'opaque)
         `(opaque ,(cadr e) ,@(map normalize-scalar-expr (cddr e))))
        (else
         (let ((args (map normalize-scalar-expr (cdr e))))
           (if (and (memq (car e) commutative-ops) (= (length args) 2))
               (let ((a (car args)) (b (cadr args)))
                 ;; an input comes first; otherwise the smaller structure,
                 ;; and equal structures keep their order
                 (if (and (not (input-ref? a))
                          (or (input-ref? b) (string<? (structure-key b) (structure-key a))))
                     (list (car e) b a)
                     (list (car e) a b)))
               (cons (car e) args))))))

;; The input numbers of e in the order of their first occurrence.
(define (input-order e)
  (define (walk e acc)
    (cond ((not (pair? e)) acc)
          ((eq? (car e) 'in) (if (memv (cadr e) acc) acc (cons (cadr e) acc)))
          ((memq (car e) '(param idx reduced)) acc)
          ((eq? (car e) 'cast) (walk (caddr e) acc))
          ((eq? (car e) 'let*)
           (walk (caddr e) (fold (lambda (b acc) (walk (cadr b) acc)) acc (cadr e))))
          ((eq? (car e) 'opaque) (fold walk acc (cddr e)))
          (else (fold walk acc (cdr e)))))
  (reverse (walk e '())))

(define (canonical-scalar-body body #!optional (n-inputs #f))
  "The canonical form of scalar expression body, and the permutation of
   its inputs, as two values.  In the canonical body the two operands of
   add and mul are ordered by their structure (an input first), directly
   nested casts to one dtype are merged, and inputs are numbered in the
   order in which they first occur.  The permutation is a list whose
   element i is the number in body of canonical input i; inputs that body
   does not read, when n-inputs is given, come last in increasing order."
  (let* ((norm  (normalize-scalar-expr body))
         (order (input-order norm))
         (order (if n-inputs
                    (append order (filter (lambda (k) (not (memv k order))) (iota n-inputs)))
                    order))
         (canon (scalar-expr-substitute-inputs
                 norm (lambda (k) `(in ,(list-index (lambda (j) (= j k)) order))))))
    (values canon order)))


;;;; ============================================================
;;;; Reductions
;;;; ============================================================

;; How the values of a reduction are combined.
;;   op:         add, mul, max, min, or (opaque proc) with proc a procedure
;;               of the accumulator and a value.
;;               max and min replace the accumulator by a value only when
;;               the value is greater (smaller); a NaN value never
;;               replaces it.  The Scheme procedures max and min, which
;;               treat NaN otherwise, are written (opaque max) and
;;               (opaque min).
;;   neutral:    the neutral element of op (0.0, 1.0, -inf.0, +inf.0, or the
;;               extreme values of an integer dtype).
;;   acc-dtype:  the dtype the accumulator is rounded to after every step:
;;               f64 accumulates in double precision, f32 rounds to single
;;               precision, and an integer dtype truncates.
;;   order:      sequential: increasing index;
;;               blocked: fixed contiguous chunks of block-size values, each
;;               folded in increasing order, the chunk results then folded
;;               in increasing order;
;;               unspecified: chosen by the backend.
;;   block-size: chunk length for the blocked order, otherwise #f.
(define-record-type combine-spec
  (%make-combine-spec op neutral acc-dtype order block-size)
  combine-spec?
  (op combine-spec-op)
  (neutral combine-spec-neutral)
  (acc-dtype combine-spec-acc-dtype)
  (order combine-spec-order)
  (block-size combine-spec-block-size))

(define (make-combine-spec op neutral acc-dtype #!optional (order 'sequential) (block-size #f))
    "A combine spec; see the record definition for the fields."
    (unless (or (memq op '(add mul max min))
                (and (pair? op) (eq? (car op) 'opaque) (pair? (cdr op)) (procedure? (cadr op))))
      (error 'make-combine-spec "unknown combine op" op))
    (unless (number? neutral)
      (error 'make-combine-spec "neutral element must be a number" neutral))
    (unless (memq acc-dtype '(f32 f64 s32 s64 u32 u64))
      (error 'make-combine-spec "unknown accumulator dtype" acc-dtype))
    (unless (memq order '(sequential blocked unspecified))
      (error 'make-combine-spec "unknown order" order))
    (when (and (eq? order 'blocked)
               (not (and (exact-integer? block-size) (> block-size 0))))
      (error 'make-combine-spec "the blocked order needs a positive block size" block-size))
    (%make-combine-spec op neutral acc-dtype order (and (eq? order 'blocked) block-size)))

;; One reduced value of a seg-red.
;;   out:      the output operand; its map uses only the parallel dimensions.
;;   body:     value at each point before combining.
;;   combine:  a combine-spec.
;;   finalize: expression over (reduced) and inputs read at the output point,
;;             or #f for (reduced) itself.
(define-record red-result out body combine finalize)


;;;; ============================================================
;;;; Segops
;;;; ============================================================

(define (procedure-or-false? x) (or (not x) (procedure? x)))
(define (space-vector? x)
  (and (vector? x) (every (lambda (n) (and (exact-integer? n) (>= n 0))) (vector->list x))))

(define-datatype segop segop?
  ;; out_j[omap_j(i)] = body_j(in_k[imap_k(i)] ...) for every i in space.
  (seg-map
    (space    space-vector?)
    (ins      list?)
    (outs     list?)
    (bodies   list?)
    (fallback procedure-or-false?))

  ;; For each result j: out_j[omap_j(p)] = finalize_j(combine_j over r of
  ;; body_j(in_k[imap_k(p, r)] ...)), where the last nred dimensions of space
  ;; are reduced (r) and the others are parallel (p).  map-outs is a list of
  ;; (operand . body) pairs written at every point of the space during the
  ;; same pass.
  (seg-red
    (space    space-vector?)
    (nred     exact-integer?)
    (ins      list?)
    (results  list?)
    (map-outs list?)
    (fallback procedure-or-false?))

  ;; A named operation with a fixed backend contract, such as an index
  ;; function run on the host; attrs is an alist of its parameters.
  (seg-lib
    (name  symbol?)
    (ins   list?)
    (outs  list?)
    (attrs list?)))

(define (check-operands who ops space-rank)
  (for-each
   (lambda (op)
     (unless (segop-operand? op)
       (error who "not a segop operand" op))
     (let ((dims (index-map-dims (segop-operand-map op))))
       (when (and dims (pair? dims) (>= (car (reverse dims)) space-rank))
         (error who "index map refers to a dimension outside the space"
                (segop-operand-map op) space-rank))))
   ops))

(define (check-output who o space)
  (let ((lin (segop-operand-linear o (vector-length space))))
    (unless lin
      (error who "an output map must be affine and unguarded" (segop-operand-map o)))
    (unless (linear-injective? (cdr lin) space)
      (error who "an output map must be one-to-one" (segop-operand-map o)))))

(define (check-body who e n-in space-rank reduced?)
  (unless (scalar-expr-valid? e n-in space-rank reduced?: reduced?)
    (error who "malformed scalar expression" e)))

(define (make-seg-map space ins outs bodies #!optional (fallback #f))
  "A seg-map over the iteration space space (a vector of extents) reading
   the operands ins and writing outs, one body per output.  Output maps must
   be affine and one-to-one.  fallback, when given, is the Scheme procedure
   computing the single output from the input values."
  (let ((rank (and (space-vector? space) (vector-length space))))
    (unless rank (error 'make-seg-map "space must be a vector of extents" space))
    (when (null? outs) (error 'make-seg-map "a seg-map needs at least one output"))
    (unless (= (length outs) (length bodies))
      (error 'make-seg-map "one body per output is needed" (length outs) (length bodies)))
    (when (and fallback (not (= (length outs) 1)))
      (error 'make-seg-map "a fallback procedure needs a single output"))
    (check-operands 'make-seg-map ins rank)
    (check-operands 'make-seg-map outs rank)
    (for-each (lambda (o) (check-output 'make-seg-map o space)) outs)
    (for-each (lambda (b) (check-body 'make-seg-map b (length ins) rank #f)) bodies)
    (seg-map space ins outs bodies fallback)))

(define (make-seg-red space nred ins results #!optional (map-outs '()) (fallback #f))
  "A seg-red over space whose last nred dimensions are reduced.  results is
   a non-empty list of red-result records.  An output map, and the map of an
   input read in a finalize body, may use only the parallel dimensions."
  (let* ((rank (and (space-vector? space) (vector-length space)))
         (npar (and rank (exact-integer? nred) (- rank nred))))
    (unless rank (error 'make-seg-red "space must be a vector of extents" space))
    (unless (and npar (<= 0 nred rank))
      (error 'make-seg-red "nred must be between 0 and the space rank" nred rank))
    (when (null? results) (error 'make-seg-red "a seg-red needs at least one result"))
    (check-operands 'make-seg-red ins rank)
    (let ((par-space (subvector space 0 npar)))
      (define (parallel-only? m)
        (let ((dims (index-map-dims m)))
          (and dims (every (lambda (d) (< d npar)) dims))))
      (for-each
       (lambda (r)
         (unless (red-result? r) (error 'make-seg-red "not a red-result" r))
         (unless (combine-spec? (red-result-combine r))
           (error 'make-seg-red "not a combine spec" (red-result-combine r)))
         (let ((o (red-result-out r)))
           (check-operands 'make-seg-red (list o) rank)
           (unless (parallel-only? (segop-operand-map o))
             (error 'make-seg-red "an output map may use only the parallel dimensions"
                    (segop-operand-map o)))
           (check-output 'make-seg-red o par-space))
         (check-body 'make-seg-red (red-result-body r) (length ins) rank #f)
         (let ((fin (red-result-finalize r)))
           (when fin
             (check-body 'make-seg-red fin (length ins) npar #t)
             (for-each (lambda (k)
                         (unless (parallel-only? (segop-operand-map (list-ref ins k)))
                           (error 'make-seg-red
                                  "a finalize input may use only the parallel dimensions" k)))
                       (scalar-expr-inputs fin)))))
       results)
      (for-each (lambda (mo)
                  (unless (pair? mo) (error 'make-seg-red "map-outs entries are (operand . body)" mo))
                  (check-operands 'make-seg-red (list (car mo)) rank)
                  (check-output 'make-seg-red (car mo) space)
                  (check-body 'make-seg-red (cdr mo) (length ins) rank #f))
                map-outs))
    (seg-red space nred ins results map-outs fallback)))

(define (make-seg-lib name ins outs attrs)
  "A named library operation (seg-lib) with operands ins and outs and the
   parameter alist attrs."
  (unless (symbol? name) (error 'make-seg-lib "name must be a symbol" name))
  (seg-lib name ins outs attrs))

(define (segop-kind s)
  "seg-map, seg-red or seg-lib."
  (cases segop s
    (seg-map (space ins outs bodies fallback) 'seg-map)
    (seg-red (space nred ins results map-outs fallback) 'seg-red)
    (seg-lib (name ins outs attrs) 'seg-lib)))

(define (segop-space s)
  "The iteration space of s, or #f for a seg-lib."
  (cases segop s
    (seg-map (space ins outs bodies fallback) space)
    (seg-red (space nred ins results map-outs fallback) space)
    (seg-lib (name ins outs attrs) #f)))

(define (segop-inputs s)
  "The input operands of s."
  (cases segop s
    (seg-map (space ins outs bodies fallback) ins)
    (seg-red (space nred ins results map-outs fallback) ins)
    (seg-lib (name ins outs attrs) ins)))

(define (segop-outputs s)
  "The output operands of s; for a seg-red, the result outputs followed by
   the map outputs."
  (cases segop s
    (seg-map (space ins outs bodies fallback) outs)
    (seg-red (space nred ins results map-outs fallback)
      (append (map red-result-out results) (map car map-outs)))
    (seg-lib (name ins outs attrs) outs)))

(define (segop-fallback s)
  "The Scheme procedure s was lowered from, or #f."
  (cases segop s
    (seg-map (space ins outs bodies fallback) fallback)
    (seg-red (space nred ins results map-outs fallback) fallback)
    (seg-lib (name ins outs attrs) #f)))


;;;; ============================================================
;;;; Reference interpreter
;;;; ============================================================

(define %acc-cells
  (map (lambda (dt) (cons dt (allocate-typed-vector dt 1))) '(f32 f64 s32 s64 u32 u64)))

;; x rounded to dtype as storing it in an array of that dtype would round
;; it (see typed-vector-set!).
(define (round-to-dtype x dtype)
  (let ((cell (cdr (assq dtype %acc-cells))))
    (typed-vector-set! cell dtype 0 x)
    (typed-vector-ref cell dtype 0)))

(define (float-dtype? dt) (memq dt '(f32 f64)))

;; Calls f on every point of space in row-major order, with a fresh
;; coordinate vector each time.
(define (for-each-point space f)
  (let ((rank (vector-length space)))
    (unless (any zero? (vector->list space))
      (let loop ((coords (make-vector rank 0)))
        (f (vector-copy coords))
        (let inc ((d (- rank 1)))
          (cond ((< d 0) #f)
                ((< (+ 1 (vector-ref coords d)) (vector-ref space d))
                 (vector-set! coords d (+ 1 (vector-ref coords d)))
                 (loop coords))
                (else (vector-set! coords d 0) (inc (- d 1)))))))))

(define (operand-flat o idx)
  (let ((strides (segop-operand-strides o)))
    (let loop ((k 0) (flat (segop-operand-offset o)))
      (if (= k (vector-length idx))
          flat
          (loop (+ k 1) (+ flat (* (vector-ref strides k) (vector-ref idx k))))))))

(define (read-operand o buf coords)
  (let ((idx (apply-index-map (segop-operand-map o) coords)))
    (if idx
        (typed-vector-ref buf (segop-operand-dtype o) (operand-flat o idx))
        (index-map-fill (segop-operand-map o)))))

(define (write-operand! o buf coords v)
  (let ((idx (apply-index-map (segop-operand-map o) coords)))
    (typed-vector-set! buf (segop-operand-dtype o) (operand-flat o idx) v)))

(define (combine-step spec acc v)
  (let ((r (case (combine-spec-op spec)
             ((add) (+ acc v))
             ((mul) (* acc v))
             ((max) (if (> v acc) v acc))
             ((min) (if (< v acc) v acc))
             (else ((cadr (combine-spec-op spec)) acc v)))))
    (round-to-dtype r (combine-spec-acc-dtype spec))))

(define (segop-interpret s in-bufs out-bufs #!key (params '#()) (use-fallback #f))
  "Execute s directly from its definition.  in-bufs and out-bufs are lists
   of SRFI-4 vectors, one per input and output operand (for a seg-red, the
   result outputs then the map outputs), laid out as the operands describe.
   params is the vector of runtime parameters.  A seg-map with a fallback
   uses it instead of its body when use-fallback is true or when an
   operand has an integer dtype.  Reductions follow their combine spec,
   with the unspecified order taken as sequential."
  (cases segop s
    (seg-map (space ins outs bodies fallback)
      (let* ((in-v (list->vector in-bufs))
             (ins-v (list->vector ins))
             (fb? (and fallback
                       (or use-fallback
                           (not (every (lambda (o) (float-dtype? (segop-operand-dtype o)))
                                       (append ins outs)))))))
        (for-each-point space
          (lambda (coords)
            (let* ((read (lambda (k) (read-operand (vector-ref ins-v k) (vector-ref in-v k) coords)))
                   (vals (if fb?
                             (list (apply fallback (map read (iota (length ins)))))
                             (map (lambda (b) (eval-scalar-expr b read params: params coords: coords))
                                  bodies))))
              (for-each (lambda (o buf v) (write-operand! o buf coords v)) outs out-bufs vals))))))

    (seg-red (space nred ins results map-outs fallback)
      (let* ((rank (vector-length space))
             (npar (- rank nred))
             (in-v (list->vector in-bufs))
             (ins-v (list->vector ins))
             (res-bufs (list-head out-bufs (length results)))
             (map-bufs (list-tail out-bufs (length results)))
             (red-space (subvector space npar rank))
             (red-points (let ((pts '()))
                           (for-each-point red-space (lambda (c) (set! pts (cons c pts))))
                           (reverse pts))))
        (for-each-point (subvector space 0 npar)
          (lambda (p)
            (let* ((point (lambda (r) (vector-append p r)))
                   (reader (lambda (coords)
                             (lambda (k) (read-operand (vector-ref ins-v k) (vector-ref in-v k) coords))))
                   (value-at (lambda (r body)
                               (let ((coords (point r)))
                                 (eval-scalar-expr body (reader coords)
                                                   params: params coords: coords)))))
              (for-each (lambda (mo buf)
                          (for-each (lambda (r)
                                      (let ((coords (point r)))
                                        (write-operand! (car mo) buf coords
                                                        (value-at r (cdr mo)))))
                                    red-points))
                        map-outs map-bufs)
              (for-each
               (lambda (res buf)
                 (let* ((spec (red-result-combine res))
                        (start (round-to-dtype (combine-spec-neutral spec)
                                               (combine-spec-acc-dtype spec)))
                        (fold-points (lambda (pts acc)
                                       (fold (lambda (r acc)
                                               (combine-step spec acc (value-at r (red-result-body res))))
                                             acc pts)))
                        (acc (if (eq? (combine-spec-order spec) 'blocked)
                                 (let loop ((pts red-points) (acc start))
                                   (if (null? pts)
                                       acc
                                       (let* ((n (min (combine-spec-block-size spec) (length pts)))
                                              (chunk (fold-points (list-head pts n) start)))
                                         (loop (list-tail pts n) (combine-step spec acc chunk)))))
                                 (fold-points red-points start)))
                        (fin (red-result-finalize res))
                        (coords (point (make-vector nred 0)))
                        (v (if fin
                               (eval-scalar-expr fin (reader coords)
                                                 params: params coords: p reduced: acc)
                               acc)))
                   (write-operand! (red-result-out res) buf coords v)))
               results res-bufs))))))

    (seg-lib (name ins outs attrs)
      (error 'segop-interpret "a seg-lib has no reference semantics" name))))

(define (list-head lst n)
  (if (= n 0) '() (cons (car lst) (list-head (cdr lst) (- n 1)))))


;;;; ============================================================
;;;; Segop templates
;;;; ============================================================

;; A layout-free description of a seg-map or a seg-red, for questions
;; asked before the arrays exist, such as whether a backend could run it
;; natively.  ins lists one template-input per input: its dtype, the index
;; map from the space to its index, and whether its layout is expected to
;; be flat (row-major, of the input's shape, at offset 0).  outs lists one
;; template-output per output: its dtype and body, and for a reduced
;; result its combine spec and finalize body.  map, when not #f, takes the
;; space to the output's index.  nred is the number of trailing dimensions
;; of the space that are reduced, 0 for a seg-map.
(define-record-type segop-template
  (%make-segop-template space ins outs nred)
  segop-template?
  (space segop-template-space)
  (ins   segop-template-ins)
  (outs  segop-template-outs)
  (nred  segop-template-nred))

(define (make-segop-template space ins outs #!optional (nred 0))
  (%make-segop-template space ins outs nred))

(define-record template-input dtype map flat?)

(define-record-type template-output
  (%make-template-output dtype body combine finalize map)
  template-output?
  (dtype    template-output-dtype)
  (body     template-output-body)
  (combine  template-output-combine)
  (finalize template-output-finalize)
  (map      template-output-map))

(define (make-template-output dtype body #!optional (combine #f) (finalize #f) (map #f))
  (%make-template-output dtype body combine finalize map))

;;;; ============================================================
;;;; Backends
;;;; ============================================================

;; A segop backend: select is a procedure (select segop) returning a pair
;; (tag . executor) when the backend can execute the segop, else #f.  The
;; executor's calling convention is set by the code that runs the segops
;; (for replay plans, see ssa.scm); tag names the pattern for profiles.
;;
;; kind says what the backend can run natively, that is, as compiled code
;; rather than as Scheme procedures on the host:
;;
;;   host       nothing; the backend runs segops with Scheme executors
;;   catalogue  a fixed set of precompiled kernels, which may include
;;              kernels for bodies that fuse several ops
;;   codegen    any body of its grammar, compiled when a plan is compiled
;;
;; native?, when given, is a procedure (native? template) that is true when
;; the backend would run the segop that template describes natively.  It
;; is asked before layouts are known (see segop templates above), so that
;; the fusion pass can fuse exactly the bindings some backend can run as
;; one native kernel.
;;
;; The backends in use are the value of the parameter segop-backends, a
;; list tried in order.  (segop-backends) reads it, (segop-backends list)
;; sets it, and (parameterize ((segop-backends list)) ...) replaces it for
;; a dynamic extent only.
(define-record-type segop-backend
  (%make-segop-backend name select kind native?)
  segop-backend?
  (name    segop-backend-name)
  (select  segop-backend-select)
  (kind    segop-backend-kind)
  (native? segop-backend-native?))

(define (make-segop-backend name select #!key (kind 'host) (native? #f))
  "A segop backend called name, selecting executors with select, of the
   given kind (host, catalogue or codegen), with the native-capability
   query native? or none."
  (unless (memq kind '(host catalogue codegen))
    (error 'make-segop-backend "kind must be host, catalogue or codegen" kind))
  (unless (or (not native?) (procedure? native?))
    (error 'make-segop-backend "native? must be a procedure or #f" native?))
  (%make-segop-backend name select kind native?))

(define (segop-native? tpl)
  "The first backend in (segop-backends) whose native-capability query
   accepts segop template tpl, or #f."
  (let loop ((bs (segop-backends)))
    (cond ((null? bs) #f)
          ((and (segop-backend-native? (car bs)) ((segop-backend-native? (car bs)) tpl))
           (car bs))
          (else (loop (cdr bs))))))

(define (check-backend-list who bs)
  (unless (and (list? bs) (every segop-backend? bs))
    (error who "not a list of segop backends" bs))
  bs)

(define segop-backends
  (make-parameter '() (lambda (bs) (check-backend-list 'segop-backends bs))))

(define (register-segop-backend! b #!optional (position 'first))
  "Add backend b to the current value of the segop-backends parameter.
   position is first (before the other backends), last (after them),
   (before name) or (after name), next to the backend called name, which
   must be present.  A backend of the same name as b is replaced.  Inside a
   parameterize of segop-backends the change lasts only until the
   parameterize exits."
  (unless (segop-backend? b)
    (error 'register-segop-backend! "not a segop backend" b))
  (let ((others (filter (lambda (x) (not (eq? (segop-backend-name x) (segop-backend-name b))))
                        (segop-backends))))
    (define (insert-next-to name after?)
      (let ((k (list-index (lambda (x) (eq? (segop-backend-name x) name)) others)))
        (unless k
          (error 'register-segop-backend! "no backend of that name" name))
        (let ((k (if after? (+ k 1) k)))
          (append (list-head others k) (list b) (list-tail others k)))))
    (segop-backends
     (cond ((eq? position 'first) (cons b others))
           ((eq? position 'last) (append others (list b)))
           ((and (pair? position) (memq (car position) '(before after))
                 (pair? (cdr position)) (symbol? (cadr position)))
            (insert-next-to (cadr position) (eq? (car position) 'after)))
           (else (error 'register-segop-backend!
                        "position must be first, last, (before name) or (after name)"
                        position))))))

(define (unregister-segop-backend! name)
  "Remove the backend called name from the current value of the
   segop-backends parameter."
  (segop-backends
   (filter (lambda (x) (not (eq? (segop-backend-name x) name))) (segop-backends))))

(define (select-segop-executor s #!optional (after #f))
  "The (tag . executor) pair of the first backend in (segop-backends) that
   accepts segop s, or #f when none does.  When after names a backend, only
   the backends that follow it are asked; a backend uses this to find the
   executor that would run s in its place."
  (let loop ((bs (if after
                     (let skip ((bs (segop-backends)))
                       (cond ((null? bs) '())
                             ((eq? (segop-backend-name (car bs)) after) (cdr bs))
                             (else (skip (cdr bs)))))
                     (segop-backends))))
    (cond ((null? bs) #f)
          (((segop-backend-select (car bs)) s) => (lambda (r) r))
          (else (loop (cdr bs))))))

;;;; ============================================================
;;;; Plan compilation
;;;; ============================================================
;;;
;;; While a replay plan is compiled, the parameter segop-plan-context holds
;;; a plan context: a place where backends keep what the plan's segops
;;; share, and a list of finalizers.  A finalizer is a procedure of no
;;; arguments that the plan compiler calls once, after the last
;;; instruction is compiled and before the plan first runs.  It lets a
;;; backend do work for all of a plan's segops at once, such as allocating
;;; a buffer they share or compiling their kernels together.  A finalizer
;;; returns #f, or an alist of entries that the plan's statistics report.
;;;
;;; An executor cell holds an executor that may be replaced after it is
;;; selected.  A backend whose kernels are made by a finalizer returns the
;;; cell's executor, which calls whatever the cell holds when it runs: an
;;; executor of another backend until the finalizer installs the kernel.

(define-record-type segop-plan-context
  (%make-segop-plan-context finalizers properties)
  segop-plan-context?
  (finalizers segop-plan-context-finalizers segop-plan-context-finalizers-set!)
  (properties segop-plan-context-properties segop-plan-context-properties-set!))

(define (make-segop-plan-context)
  "A new plan context with no finalizers and no properties."
  (%make-segop-plan-context '() '()))

;; The context of the plan being compiled, or #f outside plan compilation.
(define segop-plan-context
  (make-parameter #f (lambda (x)
                       (unless (or (not x) (segop-plan-context? x))
                         (error 'segop-plan-context "not a plan context" x))
                       x)))

(define (segop-plan-add-finalizer! thunk)
  "Add finalizer thunk to the plan being compiled, after those added
   before it.  Returns #t, or #f when no plan is being compiled (thunk is
   then not called)."
  (let ((ctx (segop-plan-context)))
    (and ctx
         (begin
           (segop-plan-context-finalizers-set!
            ctx (cons thunk (segop-plan-context-finalizers ctx)))
           #t))))

(define (segop-plan-property key #!optional (default #f))
  "The value stored under key in the plan being compiled, or default when
   there is none or no plan is being compiled."
  (let* ((ctx (segop-plan-context))
         (p   (and ctx (assq key (segop-plan-context-properties ctx)))))
    (if p (cdr p) default)))

(define (segop-plan-property-set! key value)
  "Store value under key in the plan being compiled; an error outside plan
   compilation."
  (let ((ctx (segop-plan-context)))
    (unless ctx
      (error 'segop-plan-property-set! "no plan is being compiled" key))
    (let ((p (assq key (segop-plan-context-properties ctx))))
      (if p
          (set-cdr! p value)
          (segop-plan-context-properties-set!
           ctx (cons (cons key value) (segop-plan-context-properties ctx)))))))

(define (segop-plan-finalize! ctx)
  "Call the finalizers of plan context ctx in the order they were added,
   and return the concatenation of the alists they return."
  (let loop ((fs (reverse (segop-plan-context-finalizers ctx))) (notes '()))
    (if (null? fs)
        notes
        (let ((r ((car fs))))
          (loop (cdr fs) (if (pair? r) (append notes r) notes))))))

(define-record-type segop-executor-cell
  (%make-segop-executor-cell executor)
  segop-executor-cell?
  (executor segop-executor-cell-ref segop-executor-cell-set!))

(define (make-segop-executor-cell executor)
  "A cell holding executor."
  (unless (procedure? executor)
    (error 'make-segop-executor-cell "not a procedure" executor))
  (%make-segop-executor-cell executor))

(define (segop-executor-cell-executor cell)
  "An executor that calls the executor cell holds at the time it runs,
   with the same arguments."
  (lambda (deref buf #!optional (extra-bufs '()))
    (let ((e (segop-executor-cell-ref cell)))
      (if (null? extra-bufs)
          (e deref buf)
          (e deref buf extra-bufs)))))

) ;; end module array-morphisms-segop
