;;; test-segop.scm
;;;
;;; Tests of the segop IR (segop.scm): index expressions and maps, scalar
;;; expressions, construction checks, the reference interpreter and the
;;; backend registry.
;;;
;;; Organisation:
;;;   Group 1 - Index expressions
;;;   Group 2 - Index maps: constructors, linear form, composition
;;;   Group 3 - Scalar expressions
;;;   Group 4 - seg-map interpretation
;;;   Group 5 - seg-red interpretation
;;;   Group 6 - Construction checks
;;;   Group 7 - Backend registry
;;;   Plan contexts and executor cells

(import scheme (chicken base) (chicken format) test
        (only srfi-1 every iota fold make-list filter)
        srfi-4
        datatype
        array-morphisms-core
        array-morphisms-index-fn
        array-morphisms-scalar-ops
        array-morphisms-segop)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (sg-same? a b)
  (if (and (flonum? a) (flonum? b))
      (or (and (nan? a) (nan? b)) (eqv? a b))
      (eqv? a b)))

;; A row-major operand of the given shape read or written through map.
(define (sg-op dtype shape map #!optional (slot #f))
  (make-segop-operand slot dtype shape (compute-strides shape) 0 map))

;; All points of a space, row-major.
(define (sg-points space)
  (map (lambda (i) (linear-to-multi-index i space))
       (iota (shape-size space))))

(define (sg-f64 lst) (list->f64vector (map exact->inexact lst)))
(define (sg-f32 lst) (list->f32vector (map exact->inexact lst)))

(define (sg-vec->list v)
  (cond ((f64vector? v) (f64vector->list v))
        ((f32vector? v) (f32vector->list v))
        ((s32vector? v) (s32vector->list v))
        (else (s64vector->list v))))

(define (any-of pred lst) (let loop ((l lst)) (and (pair? l) (or (pred (car l)) (loop (cdr l))))))

(define (filter-pos lst)
  (let loop ((l lst))
    (cond ((null? l) '())
          ((or (nan? (car l)) (and (> (car l) 0.0))) (cons (car l) (loop (cdr l))))
          (else (loop (cdr l))))))

(define %sg-f32 (make-f32vector 1))
(define (round-to-f32 x) (f32vector-set! %sg-f32 0 x) (f32vector-ref %sg-f32 0))

;; Inputs for agreement tests, including float specials.
(define sg-values '(0.0 -0.0 1.5 -2.25 3.0 -0.1 20.0 -20.0 1e30 -1e30 +inf.0 -inf.0 +nan.0 7.5))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1 - Index expressions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "index expressions"
  (test-assert "well-formed expressions"
    (every index-expr? '(3 (dim 0) (+ (dim 0) 2) (* 4 (dim 1)) (floordiv (dim 0) 3) (mod (dim 2) 5))))
  (test-assert "malformed expressions"
    (not (any-of index-expr? '((dim -1) (* (dim 0) (dim 1)) (mod (dim 0) 0) (foo 1) 1.5))))
  (test "evaluation" 17 (eval-index-expr '(+ (* 3 (dim 0)) (+ (dim 1) 2)) #(4 3)))
  (test "floordiv rounds towards minus infinity" -4 (eval-index-expr '(floordiv (dim 0) 2) #(-7)))
  (test "mod is non-negative" 1 (eval-index-expr '(mod (dim 0) 2) #(-7)))
  (test "affine expressions are put in canonical form"
    '(+ (+ (* 3 (dim 0)) (dim 1)) 2)
    (simplify-index-expr '(+ (dim 1) (+ 2 (* 3 (dim 0))))))
  (test "cancelling terms disappear" 5
    (simplify-index-expr '(+ (+ (dim 0) 5) (* -1 (dim 0)))))
  (test "floordiv and mod of constants are folded" 2
    (simplify-index-expr '(+ (floordiv 7 3) (mod 6 3))))
  (test "floordiv by 1 is removed" '(dim 2) (simplify-index-expr '(floordiv (dim 2) 1)))
  (test "dimensions referred to" '(0 2)
    (index-expr-dims '(+ (mod (dim 2) 3) (* 4 (dim 0))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2 - Index maps
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "index maps"
  (test "identity map" #(1 2 0) (apply-index-map (identity-index-map 3) #(1 2 0)))
  (test "broadcast of a lower-rank operand" #(2)
    (apply-index-map (broadcast-index-map #(4) 3) #(1 0 2)))
  (test "broadcast of an extent-1 dimension reads index 0" #(1 0)
    (apply-index-map (broadcast-index-map #(3 1) 2) #(1 5)))
  (test-error "an operand of higher rank cannot be broadcast"
    (broadcast-index-map #(2 2 2) 2))
  (test "extra leading dimensions of extent 1 are read at index 0" #(0 0 2)
    (apply-index-map (broadcast-index-map #(1 1 3) 1) #(2)))
  (test "a [1] operand broadcasts to a rank-0 space" #(0)
    (apply-index-map (broadcast-index-map #(1) 0) #()))
  (test-assert "identity-fn between different shapes is the reshape map"
    (let ((m (affine-index-fn->index-map (identity-fn) #(3 4) #(2 6))))
      (every (lambda (i)
               (equal? (apply-index-map m (linear-to-multi-index i #(3 4)))
                       (linear-to-multi-index i #(2 6))))
             (iota 12))))
  (test "permutation map" #(5 7 6) (apply-index-map (permutation-index-map '(1 2 0)) #(6 5 7)))
  (test "offset map" #(3 5) (apply-index-map (offset-index-map 2 '(2 0)) #(1 5)))
  (test-assert "affine index functions translate to maps"
    (every
     (lambda (fn)
       (let ((m (affine-index-fn->index-map fn #(3 4) #(3 4))))
         (every (lambda (p)
                  (equal? (vector->list (apply-index-map m p))
                          (apply-affine-index-fn fn (vector->list p))))
                (sg-points #(3 4)))))
     (list (identity-fn) (permutation-fn '(1 0)) (diagonal-fn '(2 1) '(1 3))
           (general-fn '((1 2) (0 1)) '(1 0)) (general-fn '((1 0) (1 1)) '(0 0)))))
  (test-assert "reshape map reads the element at the same row-major position"
    (let ((m (reshape-index-map #(3 4) #(2 6))))
      (every (lambda (i)
               (equal? (apply-index-map m (linear-to-multi-index i #(3 4)))
                       (linear-to-multi-index i #(2 6))))
             (iota 12))))
  (test-error "reshape between different sizes is an error" (reshape-index-map #(3 4) #(5 2)))
  (test-assert "a guarded map gives #f outside the guard"
    (let ((m (make-index-map #((+ (dim 0) -1)) '(((+ (dim 0) -1) 0 3)) 0.0 #f)))
      (and (not (apply-index-map m #(0))) (equal? #(1) (apply-index-map m #(2)))
           (not (apply-index-map m #(4))))))
  (test "dims of a map" '(0 2) (index-map-dims (permutation-index-map '(2 0))))

  (test "linear form of the identity on a row-major layout" '(0 . #(12 4 1))
    (index-map->linear (identity-index-map 3) #(12 4 1) 0 3))
  (test "linear form of a transposed view" '(0 . #(1 3))
    (index-map->linear (identity-index-map 2) #(1 3) 0 2))
  (test "linear form with an offset and a broadcast" '(5 . #(0 1))
    (index-map->linear (broadcast-index-map #(4) 2) #(1) 5 2))
  (test "linear form of an extent-1 dimension" '(0 . #(1 0))
    (index-map->linear (broadcast-index-map #(3 1) 2) #(1 1) 0 2))
  (test "linear form of a rank-0 operand" '(0 . #(0 0))
    (index-map->linear (make-index-map #() #f 0.0 #f) #() 0 2))
  (test-assert "a guarded map has no linear form"
    (not (index-map->linear (make-index-map #((dim 0)) '(((dim 0) 0 2)) 0.0 #f) #(1) 0 1)))
  (test-assert "a reshape map has no linear form"
    (not (index-map->linear (reshape-index-map #(3 4) #(2 6)) #(6 1) 0 2)))
  (test-assert "an opaque map has no linear form"
    (not (index-map->linear (opaque-index-map (lambda (c) c)) #(1) 0 1)))

  (test-assert "a permutation composed with its inverse is the identity"
    (let ((m (compose-index-maps (permutation-index-map '(2 0 1)) (permutation-index-map '(1 2 0)))))
      (equal? (index-map-exprs m) (index-map-exprs (identity-index-map 3)))))
  (test-assert "composition agrees with applying the maps in turn"
    (let* ((g (affine-index-fn->index-map (diagonal-fn '(2 1) '(1 0)) #(3 3) #(7 3)))
           (f (permutation-index-map '(1 0)))
           (fg (compose-index-maps f g)))
      (every (lambda (p) (equal? (apply-index-map fg p) (apply-index-map f (apply-index-map g p))))
             (sg-points #(3 3)))))
  (test-assert "composition keeps both guards"
    (let* ((g (make-index-map #((+ (dim 0) -1)) '(((dim 0) 1 5)) 0.0 #f))
           (f (make-index-map #((* 2 (dim 0))) '(((dim 0) 0 3)) 0.0 #f))
           (fg (compose-index-maps f g)))
      (and (= 2 (length (index-map-guard fg)))
           (not (apply-index-map fg #(0)))      ; g's guard fails
           (equal? #(2) (apply-index-map fg #(2)))
           (not (apply-index-map fg #(4))))))   ; f's guard fails at 3
  (test-error "guards with different fill values cannot be composed"
    (compose-index-maps (make-index-map #((dim 0)) '(((dim 0) 0 3)) 1.0 #f)
                        (make-index-map #((dim 0)) '(((dim 0) 0 3)) 0.0 #f)))
  (test-assert "composition with an opaque map is opaque and still correct"
    (let ((m (compose-index-maps (permutation-index-map '(1 0))
                                 (opaque-index-map (lambda (c) (vector (vector-ref c 0) 7))))))
      (and (index-map-opaque? m) (equal? #(7 3) (apply-index-map m #(3 9))))))

  (test-assert "row-major coefficients are one-to-one" (linear-injective? #(12 4 1) #(2 3 4)))
  (test-assert "permuted coefficients are one-to-one" (linear-injective? #(1 2) #(2 3)))
  (test-assert "a broadcast (stride 0) is not one-to-one" (not (linear-injective? #(0 1) #(2 3))))
  (test-assert "overlapping strides are not one-to-one" (not (linear-injective? #(2 1) #(2 3))))
  (test-assert "a stride-0 dimension of extent 1 is allowed" (linear-injective? #(0 1) #(1 3)))
  (test-assert "an empty space is trivially one-to-one" (linear-injective? #(0 0) #(0 3))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3 - Scalar expressions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (sg-reader . vals) (lambda (k) (list-ref vals k)))

(test-group "scalar expressions"
  (test-assert "valid bodies"
    (every (lambda (e) (scalar-expr-valid? e 2 3))
           '((in 0) (relu (in 1)) (add (in 0) (mul (in 1) 2.0)) (+ (idx 2) (param 0))
             (let* ((a (in 0)) (b (* a a))) (if (> b 1.0) b a)))))
  (test-assert "invalid bodies"
    (not (any-of (lambda (e) (scalar-expr-valid? e 2 3))
                 '((in 2) (idx 3) (relu (in 0) (in 1)) (foo (in 0)) (reduced) unbound-var
                   (let* ((a)) a)))))
  (test-assert "(reduced) is valid in a finalize body" (scalar-expr-valid? '(/ (reduced) 4.0) 0 0 reduced?: #t))
  (test-assert "opaque bodies are valid" (scalar-expr-valid? (list 'opaque sqrt '(in 0)) 1 1))
  (test "table ops are evaluated from their expressions" 0.0
    (eval-scalar-expr '(relu (in 0)) (sg-reader -3.0)))
  (test "nested table ops" 13.0
    (eval-scalar-expr '(add (in 0) (mul (in 1) 4.0)) (sg-reader 1.0 3.0)))
  (test "params, idx and let*" 11.0
    (eval-scalar-expr '(let* ((a (param 1)) (b (+ a (idx 1)))) (* b 1.0))
                      (sg-reader) params: #(0.0 4.0) coords: #(9 7)))
  (test "opaque parts call their procedure" 5.0
    (eval-scalar-expr (list 'opaque (lambda (x y) (+ x y)) '(in 0) 2.0) (sg-reader 3.0)))
  (test "reduced in a finalize body" 2.5 (eval-scalar-expr '(/ (reduced) 4.0) (sg-reader) reduced: 10.0))
  (test-assert "every table op evaluates as its combiner on flonums"
    (every (lambda (op)
             (let ((f (scalar-op-combiner op)) (n (scalar-op-arity op)))
               (every (lambda (x)
                        (let* ((args (if (= n 1) (list x) (list x 1.75)))
                               (body (cons op (map (lambda (k) `(in ,k)) (iota n)))))
                          (sg-same? (apply f args)
                                    (eval-scalar-expr body (lambda (k) (list-ref args k))))))
                      (if (memq op '(sqrt log pow))
                          (filter-pos sg-values)
                          sg-values))))
           (scalar-op-names)))
  (test "substitution of inputs"
    '(relu (add (in 2) (in 0)))
    (scalar-expr-substitute-inputs '(relu (in 0))
                                   (lambda (k) (if (= k 0) '(add (in 2) (in 0)) `(in ,k)))))
  (test "substitution does not touch let* names or params"
    '(let* ((a (exp (in 1)))) (* a (param 0)))
    (scalar-expr-substitute-inputs '(let* ((a (in 0))) (* a (param 0)))
                                   (lambda (k) '(exp (in 1)))))
  (test "inputs used" '(0 2) (scalar-expr-inputs '(add (in 2) (let* ((a (in 0))) (mul a (in 2))))))
  (test-assert "opaque detection"
    (and (scalar-expr-opaque? (list 'add (list 'opaque abs '(in 0)) '(in 1)))
         (not (scalar-expr-opaque? '(relu (in 0))))))
  ;; (cast dtype e) rounds as a store to dtype would
  (test-assert "casts are valid with a known dtype"
    (and (scalar-expr-valid? '(relu (cast f32 (reduced))) 0 0 reduced?: #t)
         (scalar-expr-valid? '(cast s64 (in 1)) 2 0)))
  (test-assert "invalid casts"
    (not (any-of (lambda (e) (scalar-expr-valid? e 2 0))
                 '((cast f16 (in 0)) (cast f32) (cast f32 (in 0) (in 1)) (cast (in 0) f32)
                   (cast f32 (in 2))))))
  (test "a cast to f32 rounds" (round-to-f32 0.1) (eval-scalar-expr '(cast f32 (in 0)) (sg-reader 0.1)))
  (test-assert "a cast to f32 keeps -0.0, infinities and NaN"
    (every (lambda (x) (sg-same? (round-to-f32 x) (eval-scalar-expr '(cast f32 (in 0)) (sg-reader x))))
           sg-values))
  (test-assert "a cast to f64 is the identity"
    (every (lambda (x) (sg-same? x (eval-scalar-expr '(cast f64 (in 0)) (sg-reader x)))) sg-values))
  (test "a cast to s32 truncates" -2 (eval-scalar-expr '(cast s32 (in 0)) (sg-reader -2.7)))
  (test "a cast inside a finalize body" (round-to-f32 (/ 1.0 3.0))
    (eval-scalar-expr '(cast f32 (/ (reduced) 3.0)) (sg-reader) reduced: 1.0))
  (test "substitution rewrites under a cast and keeps its dtype"
    '(cast f32 (relu (in 3)))
    (scalar-expr-substitute-inputs '(cast f32 (relu (in 0))) (lambda (k) '(in 3))))
  (test "inputs used under a cast" '(1) (scalar-expr-inputs '(cast f32 (exp (in 1)))))
  (test-assert "opaque detection under a cast"
    (and (scalar-expr-opaque? (list 'cast 'f32 (list 'opaque abs '(in 0))))
         (not (scalar-expr-opaque? '(cast f32 (in 0)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4 - seg-map interpretation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "seg-map interpretation"
  (let* ((space #(2 7))
         (x (sg-f32 (append sg-values)))
         (out (make-f32vector 14 9.0))
         (sm (make-seg-map space
                           (list (sg-op 'f32 space (identity-index-map 2)))
                           (list (sg-op 'f32 space (identity-index-map 2)))
                           '((relu (in 0)))
                           (scalar-op-combiner 'relu))))
    (segop-interpret sm (list x) (list out))
    (test-assert "relu over f32 equals the combiner loop, bit for bit"
      (every sg-same? (f32vector->list out)
             (map (lambda (v) (exact->inexact (round-to-f32 ((scalar-op-combiner 'relu) v))))
                  (f32vector->list x))))
    (let ((out2 (make-f32vector 14 9.0)))
      (segop-interpret sm (list x) (list out2) use-fallback: #t)
      (test-assert "the body and the fallback give the same f32 results"
        (every sg-same? (f32vector->list out) (f32vector->list out2)))))

  (let* ((space #(3 4))
         (a (sg-f64 (iota 12)))
         (row (sg-f64 '(10 20 30)))        ; shape [3 1]
         (col (sg-f64 '(1 2 3 4)))         ; shape [4]
         (sc  (sg-f64 '(0.5)))             ; shape [1]
         (out (make-f64vector 12 0.0))
         (sm (make-seg-map space
                           (list (sg-op 'f64 space (identity-index-map 2))
                                 (sg-op 'f64 #(3 1) (broadcast-index-map #(3 1) 2))
                                 (sg-op 'f64 #(4) (broadcast-index-map #(4) 2))
                                 (sg-op 'f64 #(1) (broadcast-index-map #(1) 2)))
                           (list (sg-op 'f64 space (identity-index-map 2)))
                           '((add (mul (in 0) (in 3)) (sub (in 1) (in 2)))))))
    (segop-interpret sm (list a row col sc) (list out))
    (test "row, column and scalar broadcasts"
      (map (lambda (i) (+ (* 0.5 i) (- (* 10.0 (+ 1 (quotient i 4))) (+ 1.0 (remainder i 4)))))
           (iota 12))
      (f64vector->list out)))

  (let* ((data (sg-f64 '(0 1 2 3 4 5)))      ; a [2 3] array, read transposed as [3 2]
         (out (make-f64vector 6 0.0))
         (in-op (make-segop-operand #f 'f64 #(3 2) #(1 3) 0 (identity-index-map 2)))
         (sm (make-seg-map #(3 2) (list in-op)
                           (list (sg-op 'f64 #(3 2) (identity-index-map 2)))
                           '((in 0)))))
    (segop-interpret sm (list data) (list out))
    (test "a transposed view is read in place" '(0.0 3.0 1.0 4.0 2.0 5.0) (f64vector->list out)))

  (let* ((data (sg-f64 '(9 9 1 2 3 4 5 6)))  ; [2 3] view at offset 2
         (out (make-f64vector 6 0.0))
         (in-op (make-segop-operand #f 'f64 #(2 3) #(3 1) 2 (identity-index-map 2)))
         (sm (make-seg-map #(2 3) (list in-op)
                           (list (sg-op 'f64 #(2 3) (identity-index-map 2)))
                           '((negate (in 0))))))
    (segop-interpret sm (list data) (list out))
    (test "an offset view" '(-1.0 -2.0 -3.0 -4.0 -5.0 -6.0) (f64vector->list out)))

  (let* ((src (sg-f64 '(1 2 3)))
         (out (make-f64vector 5 0.0))
         (pad (make-index-map #((+ (dim 0) -1)) '(((+ (dim 0) -1) 0 3)) -7.0 #f))
         (sm (make-seg-map #(5) (list (sg-op 'f64 #(3) pad))
                           (list (sg-op 'f64 #(5) (identity-index-map 1)))
                           '((in 0)))))
    (segop-interpret sm (list src) (list out))
    (test "a guarded read gives the fill value outside the guard"
      '(-7.0 1.0 2.0 3.0 -7.0) (f64vector->list out)))

  (let* ((src (sg-f64 '(1 2 3 4 5 6)))
         (out (make-f64vector 6 0.0))
         (sm (make-seg-map #(3 2) (list (sg-op 'f64 #(2 3) (reshape-index-map #(3 2) #(2 3))))
                           (list (sg-op 'f64 #(3 2) (identity-index-map 2)))
                           '((in 0)))))
    (segop-interpret sm (list src) (list out))
    (test "a reshape gather" '(1.0 2.0 3.0 4.0 5.0 6.0) (f64vector->list out)))

  ;; An Adam-style update: inputs (p g m v), outputs (p' m' v'), with the
  ;; hyperparameters as runtime parameters #(lr beta1 beta2 eps).
  (let* ((n 4)
         (p (sg-f64 '(1.0 -0.5 0.25 2.0))) (g (sg-f64 '(0.1 -0.2 0.3 0.0)))
         (m (sg-f64 '(0.0 0.01 -0.02 0.5))) (v (sg-f64 '(0.0 0.04 0.01 0.2)))
         (params #(0.01 0.9 0.999 1e-8))
         (op (lambda () (sg-op 'f64 #(4) (identity-index-map 1))))
         (m-new '(+ (* (param 1) (in 2)) (* (- 1.0 (param 1)) (in 1))))
         (v-new '(+ (* (param 2) (in 3)) (* (- 1.0 (param 2)) (* (in 1) (in 1)))))
         (sm (make-seg-map #(4) (list (op) (op) (op) (op)) (list (op) (op) (op))
                           (list `(- (in 0) (/ (* (param 0) ,m-new) (+ (sqrt ,v-new) (param 3))))
                                 m-new v-new)))
         (p2 (make-f64vector n)) (m2 (make-f64vector n)) (v2 (make-f64vector n)))
    (segop-interpret sm (list p g m v) (list p2 m2 v2) params: params)
    (test-assert "a multi-output seg-map with parameters (Adam step)"
      (every (lambda (i)
               (let* ((gi (f64vector-ref g i))
                      (mi (+ (* 0.9 (f64vector-ref m i)) (* (- 1.0 0.9) gi)))
                      (vi (+ (* 0.999 (f64vector-ref v i)) (* (- 1.0 0.999) (* gi gi))))
                      (pi (- (f64vector-ref p i) (/ (* 0.01 mi) (+ (sqrt vi) 1e-8)))))
                 (and (eqv? mi (f64vector-ref m2 i)) (eqv? vi (f64vector-ref v2 i))
                      (eqv? pi (f64vector-ref p2 i)))))
             (iota n))))

  (let* ((a (list->s32vector '(1 -2 3 -4)))
         (b (list->s32vector '(5 6 7 8)))
         (out (make-s32vector 4 0))
         (op (lambda () (sg-op 's32 #(4) (identity-index-map 1))))
         (sm (make-seg-map #(4) (list (op) (op)) (list (op)) '((mul (in 0) (in 1))) *)))
    (segop-interpret sm (list a b) (list out))
    (test "integer dtypes use the exact fallback" '(5 -12 21 -32) (s32vector->list out)))

  (let* ((out (make-f64vector 0))
         (sm (make-seg-map #(0 3) (list (sg-op 'f64 #(0 3) (identity-index-map 2)))
                           (list (sg-op 'f64 #(0 3) (identity-index-map 2))) '((exp (in 0))))))
    (segop-interpret sm (list (make-f64vector 0)) (list out))
    (test-assert "an empty space writes nothing" #t))

  (let* ((out (make-f64vector 1 0.0))
         (sm (make-seg-map #() (list (sg-op 'f64 #() (identity-index-map 0)))
                           (list (sg-op 'f64 #() (identity-index-map 0))) '((sqrt (in 0))))))
    (segop-interpret sm (list (sg-f64 '(16))) (list out))
    (test "a rank-0 space has one point" 4.0 (f64vector-ref out 0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5 - seg-red interpretation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; A seg-red summing a [rows cols] array over axis 0 (space (cols rows),
;; the input read transposed) with one result.
(define (sg-col-sum dtype rows cols spec #!optional (finalize #f))
  (make-seg-red (vector cols rows) 1
                (list (sg-op dtype (vector rows cols) (permutation-index-map '(1 0))))
                (list (make-red-result (sg-op dtype (vector cols) (identity-index-map 1))
                                       '(in 0) spec finalize))))

(test-group "seg-red interpretation"
  ;; 1e8 followed by ones: in single precision each 1 is lost (1e8 + 1
  ;; rounds back to 1e8), in double precision they are kept until the end.
  (let* ((rows 9) (data (sg-f32 (cons 1e8 (make-list (- rows 1) 1.0))))
         (o32 (make-f32vector 1)) (o64 (make-f32vector 1)))
    (segop-interpret (sg-col-sum 'f32 rows 1 (make-combine-spec 'add 0.0 'f32)) (list data) (list o32))
    (segop-interpret (sg-col-sum 'f32 rows 1 (make-combine-spec 'add 0.0 'f64)) (list data) (list o64))
    (test "f32 accumulation rounds at every step" 1e8 (f32vector-ref o32 0))
    (test "f64 accumulation rounds once" (round-to-f32 (+ 1e8 8.0)) (f32vector-ref o64 0)))

  (let* ((data (sg-f64 '(1 2 3 4 5 6)))  ; [2 3]
         (out (make-f64vector 3)))
    (segop-interpret (sg-col-sum 'f64 2 3 (make-combine-spec 'add 0.0 'f64) '(/ (reduced) 2.0))
                     (list data) (list out))
    (test "mean over axis 0 as a finalize step" '(2.5 3.5 4.5) (f64vector->list out)))

  (let* ((data (sg-f64 '(-5 2 -1 -3 -inf.0 -2)))
         (out (make-f64vector 3)))
    (segop-interpret (sg-col-sum 'f64 2 3 (make-combine-spec 'max -inf.0 'f64)) (list data) (list out))
    (test "max with neutral -inf.0" '(-3.0 2.0 -1.0) (f64vector->list out)))

  (let* ((data (sg-f64 (map (lambda (i) (- (* 3 i) 20)) (iota 40))))
         (seq (make-f64vector 1)) (blk1 (make-f64vector 1)) (blk2 (make-f64vector 1)))
    (segop-interpret (sg-col-sum 'f64 40 1 (make-combine-spec 'add 0.0 'f64)) (list data) (list seq))
    (segop-interpret (sg-col-sum 'f64 40 1 (make-combine-spec 'add 0.0 'f64 'blocked 7)) (list data) (list blk1))
    (segop-interpret (sg-col-sum 'f64 40 1 (make-combine-spec 'add 0.0 'f64 'blocked 7)) (list data) (list blk2))
    (test-assert "the blocked order equals the sequential one on integer-valued data"
      (eqv? (f64vector-ref seq 0) (f64vector-ref blk1 0)))
    (test-assert "the blocked order is the same on every run"
      (eqv? (f64vector-ref blk1 0) (f64vector-ref blk2 0))))

  (let* ((data (sg-f64 '(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1)))
         (blk (make-f64vector 1)))
    (segop-interpret (sg-col-sum 'f64 11 1 (make-combine-spec 'add 0.0 'f64 'blocked 4))
                     (list data) (list blk))
    (test "the blocked order folds chunks, then chunk results"
      (+ (+ (+ 0.0 (+ (+ (+ (+ 0.0 0.1) 0.2) 0.3) 0.4))
            (+ (+ (+ (+ 0.0 0.5) 0.6) 0.7) 0.8))
         (+ (+ (+ 0.0 0.9) 1.0) 1.1))
      (f64vector-ref blk 0)))

  ;; Row sums and sums of squares in one pass, plus the squares stored as
  ;; a map output: [2 3] input, space (2 3) with the last dimension reduced.
  (let* ((data (sg-f64 '(1 2 3 4 5 6)))
         (sum (make-f64vector 2)) (sq (make-f64vector 2)) (squares (make-f64vector 6))
         (in (sg-op 'f64 #(2 3) (identity-index-map 2)))
         (sr (make-seg-red #(2 3) 1 (list in)
                           (list (make-red-result (sg-op 'f64 #(2) (identity-index-map 1))
                                                  '(in 0) (make-combine-spec 'add 0.0 'f64) #f)
                                 (make-red-result (sg-op 'f64 #(2) (identity-index-map 1))
                                                  '(mul (in 0) (in 0)) (make-combine-spec 'add 0.0 'f64) #f))
                           (list (cons (sg-op 'f64 #(2 3) (identity-index-map 2)) '(mul (in 0) (in 0)))))))
    (segop-interpret sr (list data) (list sum sq squares))
    (test "first result: row sums" '(6.0 15.0) (f64vector->list sum))
    (test "second result: row sums of squares" '(14.0 77.0) (f64vector->list sq))
    (test "map output written in the same pass" '(1.0 4.0 9.0 16.0 25.0 36.0) (f64vector->list squares)))

  ;; A contraction with an epilogue: out = relu(A @ B + bias).
  (let* ((A (sg-f64 '(1 -2 3 4 0 -1)))     ; [2 3]
         (B (sg-f64 '(1 0 2 1 -1 1)))      ; [3 2]
         (bias (sg-f64 '(0.5 -10)))         ; [2]
         (out (make-f64vector 4))
         (sr (make-seg-red #(2 2 3) 1
                           (list (sg-op 'f64 #(2 3) (permutation-index-map '(0 2)))
                                 (sg-op 'f64 #(3 2) (permutation-index-map '(2 1)))
                                 (sg-op 'f64 #(2) (make-index-map #((dim 1)) #f 0.0 #f)))
                           (list (make-red-result (sg-op 'f64 #(2 2) (identity-index-map 2))
                                                  '(mul (in 0) (in 1))
                                                  (make-combine-spec 'add 0.0 'f64 'unspecified)
                                                  '(relu (add (reduced) (in 2))))))))
    (segop-interpret sr (list A B bias) (list out))
    (test "a GEMM with a bias and relu epilogue" '(0.0 0.0 5.5 0.0)
      (f64vector->list out)))

  (let* ((out (make-f64vector 2 9.0))
         (sr (make-seg-red #(2 0) 1 (list (sg-op 'f64 #(2 0) (identity-index-map 2)))
                           (list (make-red-result (sg-op 'f64 #(2) (identity-index-map 1))
                                                  '(in 0) (make-combine-spec 'add 0.0 'f64) #f)))))
    (segop-interpret sr (list (make-f64vector 0)) (list out))
    (test "an empty reduction gives the neutral element" '(0.0 0.0) (f64vector->list out)))

  (let* ((data (sg-f64 '(2 3 4)))
         (out (make-f64vector 1))
         (sr (make-seg-red #(3) 1 (list (sg-op 'f64 #(3) (identity-index-map 1)))
                           (list (make-red-result (sg-op 'f64 #() (identity-index-map 0))
                                                  '(in 0)
                                                  (make-combine-spec (list 'opaque (lambda (a b) (+ (* 10.0 a) b)))
                                                                     0.0 'f64)
                                                  #f)))))
    (segop-interpret sr (list data) (list out))
    (test "an opaque combine op, full reduction to rank 0" 234.0 (f64vector-ref out 0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6 - Construction checks
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "combine semantics"
  (let* ((data (sg-f64 '(1.0 +nan.0 3.0 -0.0 0.0 -1.0)))   ; [2 3], reduce axis 0
         (sel (make-f64vector 3)) (opq (make-f64vector 3)))
    (segop-interpret (sg-col-sum 'f64 2 3 (make-combine-spec 'max -inf.0 'f64)) (list data) (list sel))
    (segop-interpret (sg-col-sum 'f64 2 3 (make-combine-spec (list 'opaque max) -inf.0 'f64))
                     (list data) (list opq))
    (test-assert "max replaces only by greater values: NaN is skipped, -0.0 kept first"
      (and (eqv? 1.0 (f64vector-ref sel 0)) (eqv? 3.0 (f64vector-ref sel 2))
           (eqv? 0.0 (f64vector-ref sel 1))))
    (test-assert "(opaque max) applies the Scheme procedure"
      (every sg-same? (f64vector->list opq)
             (list (max (max -inf.0 1.0) -0.0) (max (max -inf.0 +nan.0) 0.0)
                   (max (max -inf.0 3.0) -1.0)))))
  (let* ((data (list->s32vector '(3 -1 4 1 -5 9)))
         (out (make-s32vector 3 0))
         (sr (make-seg-red #(3 2) 1
                           (list (sg-op 's32 #(2 3) (permutation-index-map '(1 0))))
                           (list (make-red-result (sg-op 's32 #(3) (identity-index-map 1)) '(in 0)
                                                  (make-combine-spec 'max (- (expt 2 31)) 's32) #f)))))
    (segop-interpret sr (list data) (list out))
    (test "an integer accumulator starting at the dtype minimum" '(3 -1 9) (s32vector->list out)))
  (let* ((data (sg-f64 '(0.75 0.75 0.75)))
         (out (make-f64vector 1))
         (sr (make-seg-red #(3) 1 (list (sg-op 'f64 #(3) (identity-index-map 1)))
                           (list (make-red-result (sg-op 'f64 #() (identity-index-map 0)) '(in 0)
                                                  (make-combine-spec 'add 0.0 's32) #f)))))
    (segop-interpret sr (list data) (list out))
    (test "an integer accumulator truncates after every step" 0.0 (f64vector-ref out 0)))
  (test-error "an unknown accumulator dtype is an error" (make-combine-spec 'add 0.0 'c64)))

(test-group "construction checks"
  (let ((id2 (lambda () (sg-op 'f64 #(2 3) (identity-index-map 2))))
        (res (lambda (out) (make-red-result out '(in 0) (make-combine-spec 'add 0.0 'f64) #f))))
    (test-error "a broadcast output map is rejected"
      (make-seg-map #(2 3) (list (id2)) (list (sg-op 'f64 #(3) (broadcast-index-map #(3) 2))) '((in 0))))
    (test-error "a guarded output map is rejected"
      (make-seg-map #(3) (list (sg-op 'f64 #(3) (identity-index-map 1)))
                    (list (sg-op 'f64 #(3) (make-index-map #((dim 0)) '(((dim 0) 0 2)) 0.0 #f)))
                    '((in 0))))
    (test-assert "a permuted output map is accepted"
      (make-seg-map #(2 3) (list (id2)) (list (sg-op 'f64 #(3 2) (permutation-index-map '(1 0))))
                    '((in 0))))
    (test-assert "an offset output map is accepted"
      (make-seg-map #(2 3) (list (id2)) (list (sg-op 'f64 #(4 3) (offset-index-map 2 '(2 0))))
                    '((in 0))))
    (test-error "one body per output" (make-seg-map #(2 3) (list (id2)) (list (id2)) '()))
    (test-error "a body may not use an input that does not exist"
      (make-seg-map #(2 3) (list (id2)) (list (id2)) '((in 1))))
    (test-error "a map may not refer to a dimension outside the space"
      (make-seg-map #(2 3) (list (sg-op 'f64 #(3) (make-index-map #((dim 2)) #f 0.0 #f)))
                    (list (id2)) '((in 0))))
    (test-error "an operand map must match the operand rank"
      (sg-op 'f64 #(2 3) (identity-index-map 3)))
    (test-error "a fallback needs a single output"
      (make-seg-map #(2 3) (list (id2)) (list (id2) (id2)) '((in 0) (in 0)) abs))
    (test-error "nred may not exceed the space rank"
      (make-seg-red #(2 3) 3 (list (id2)) (list (res (sg-op 'f64 #(2) (identity-index-map 1))))))
    (test-error "a seg-red needs a result" (make-seg-red #(2 3) 1 (list (id2)) '()))
    (test-error "an output map may not use a reduced dimension"
      (make-seg-red #(2 3) 1 (list (id2))
                    (list (res (sg-op 'f64 #(3) (make-index-map #((dim 1)) #f 0.0 #f))))))
    (test-error "a finalize input may not use a reduced dimension"
      (make-seg-red #(2 3) 1 (list (id2))
                    (list (make-red-result (sg-op 'f64 #(2) (identity-index-map 1)) '(in 0)
                                           (make-combine-spec 'add 0.0 'f64) '(add (reduced) (in 0))))))
    (test-error "an unknown combine op" (make-combine-spec 'avg 0.0 'f64))
    (test-error "the blocked order needs a block size" (make-combine-spec 'add 0.0 'f64 'blocked))
    (test-error "an unknown accumulator dtype" (make-combine-spec 'add 0.0 'f16))
    (test-assert "a seg-lib is built"
      (eq? 'seg-lib (segop-kind (make-seg-lib 'conv2d-fwd (list (id2)) (list (id2)) '((KH . 3))))))
    (test-error "a seg-lib has no reference semantics"
      (segop-interpret (make-seg-lib 'conv2d-fwd (list (id2)) (list (id2)) '()) '() '()))
    (let ((sm (make-seg-map #(2 3) (list (id2)) (list (id2)) '((exp (in 0))) exp)))
      (test "accessors" (list 'seg-map #(2 3) 1 1 exp)
        (list (segop-kind sm) (segop-space sm) (length (segop-inputs sm))
              (length (segop-outputs sm)) (segop-fallback sm))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7 - Backend registry
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "backend registry"
  (let* ((saved (segop-backends))
         (sm (make-seg-map #(2) (list (sg-op 'f64 #(2) (identity-index-map 1)))
                           (list (sg-op 'f64 #(2) (identity-index-map 1))) '((exp (in 0)))))
         (b1 (make-segop-backend 'test-a (lambda (s) (cons 'tag-a 'exec-a))))
         (b2 (make-segop-backend 'test-b (lambda (s) (cons 'tag-b 'exec-b))))
         (none (make-segop-backend 'test-none (lambda (s) #f))))
    (register-segop-backend! b1 'last)
    (register-segop-backend! none)
    (test "a backend that declines is skipped" 'tag-a
      (and (select-segop-executor sm) (car (select-segop-executor sm))))
    (register-segop-backend! b2)
    (test "a backend registered first is tried first" 'tag-b (car (select-segop-executor sm)))
    (register-segop-backend! (make-segop-backend 'test-b (lambda (s) (cons 'tag-b2 'x))))
    (test "a backend of the same name is replaced" 'tag-b2 (car (select-segop-executor sm)))
    (unregister-segop-backend! 'test-b)
    (unregister-segop-backend! 'test-none)
    (unregister-segop-backend! 'test-a)
    (test "unregistering restores the registry" (length saved) (length (segop-backends)))
    (test-error "registering a non-backend is an error" (register-segop-backend! 'x))
    (test-error "the parameter accepts only a list of backends" (segop-backends '(x)))
    (test "a parameterize replaces the backends for its extent only" '(tag-a #t)
      (list (parameterize ((segop-backends (list b1)))
              (car (select-segop-executor sm)))
            (equal? saved (segop-backends))))
    (test-assert "registering inside a parameterize does not outlive it"
      (begin (parameterize ((segop-backends (segop-backends)))
               (register-segop-backend! b2))
             (not (memq b2 (segop-backends)))))
    (test "with no backends nothing is selected" #f
      (parameterize ((segop-backends '())) (select-segop-executor sm)))
    (parameterize ((segop-backends (list b1 none)))
      (register-segop-backend! b2 '(before test-none))
      (test "a backend can be placed before a named one" '(test-a test-b test-none)
        (map segop-backend-name (segop-backends)))
      (register-segop-backend! b2 '(after test-none))
      (test "a backend can be placed after a named one" '(test-a test-none test-b)
        (map segop-backend-name (segop-backends)))
      (test-error "placing next to a missing backend is an error"
        (register-segop-backend! b2 '(before test-missing)))
      (test-error "an unknown position is an error" (register-segop-backend! b2 'middle))
      (test "only the backends after a named one are asked" 'tag-b
        (car (select-segop-executor sm 'test-a)))
      (test "nothing follows the last backend" #f (select-segop-executor sm 'test-b)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Plan contexts and executor cells
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "plan contexts"
  (test "outside a plan there is no context" #f (segop-plan-context))
  (test "outside a plan a finalizer is not added" #f (segop-plan-add-finalizer! (lambda () #f)))
  (test "outside a plan a property has its default" 'd (segop-plan-property 'k 'd))
  (test-error "outside a plan a property cannot be set" (segop-plan-property-set! 'k 1))
  (let* ((ctx (make-segop-plan-context))
         (trail (list)))
    (parameterize ((segop-plan-context ctx))
      (test-assert "inside a plan a finalizer is added"
        (segop-plan-add-finalizer! (lambda () (set! trail (cons 'first trail)) '((a . 1)))))
      (segop-plan-add-finalizer! (lambda () (set! trail (cons 'second trail)) #f))
      (segop-plan-add-finalizer! (lambda () (set! trail (cons 'third trail)) '((b . 2) (c . 3))))
      (segop-plan-property-set! 'k 1)
      (segop-plan-property-set! 'k 2)
      (test "a property keeps the last value set" 2 (segop-plan-property 'k)))
    (test "finalizers have not run before the plan is finalized" 0 (length trail))
    (test "finalizers run in order and their notes are joined" '((a . 1) (b . 2) (c . 3))
      (segop-plan-finalize! ctx))
    (test "every finalizer ran once" '(first second third) (reverse trail)))
  (test-error "the context parameter accepts only plan contexts"
    (parameterize ((segop-plan-context 'x)) #f)))

(test-group "executor cells"
  (let* ((cell (make-segop-executor-cell (lambda (deref buf) (list 'host buf))))
         (exec (segop-executor-cell-executor cell)))
    (test "the executor calls what the cell holds" '(host 1) (exec #f 1))
    (segop-executor-cell-set! cell (lambda (deref buf #!optional (extra '()))
                                     (list 'kernel buf extra)))
    (test "a replaced executor is called by the same executor" '(kernel 2 ()) (exec #f 2))
    (test "extra output buffers are passed on" '(kernel 3 (4 5)) (exec #f 3 '(4 5)))
    (test-error "a cell holds a procedure" (make-segop-executor-cell 'x))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Body compiler
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; The compiled body of e applied to args, and e evaluated directly.
(define (bc-compiled e args) (apply (scalar-expr->procedure e (length args)) args))
(define (bc-evaluated e args) (eval-scalar-expr e (lambda (k) (list-ref args k))))

(test-group "body compiler"
  (test-assert "every table op compiles to its combiner"
    (every (lambda (op)
             (let ((f (scalar-op-combiner op)) (n (scalar-op-arity op)))
               (every (lambda (x)
                        (let* ((args (if (= n 1) (list x) (list x 1.75)))
                               (body (cons op (map (lambda (k) `(in ,k)) (iota n)))))
                          (and (sg-same? (apply f args) (bc-compiled body args))
                               (sg-same? (bc-evaluated body args) (bc-compiled body args)))))
                      (if (memq op '(sqrt log pow)) (filter-pos sg-values) sg-values))))
           (scalar-op-names)))
  (test-assert "generic table ops on integer arguments give the combiner's result"
    (every (lambda (op)
             (let ((args (if (= (scalar-op-arity op) 1) '(7) '(7 -3))))
               (equal? (apply (scalar-op-combiner op) args)
                       (bc-compiled (cons op (map (lambda (k) `(in ,k)) (iota (length args)))) args))))
           (filter scalar-op-generic? (filter (lambda (op) (not (memq op '(div pow log sqrt))))
                                              (scalar-op-names)))))
  (test-assert "binary table ops on signed zeros agree with the combiner and the interpreter"
    (every (lambda (op)
             (every (lambda (args)
                      (let ((body (list op '(in 0) '(in 1))))
                        (and (sg-same? (apply (scalar-op-combiner op) args) (bc-compiled body args))
                             (sg-same? (bc-evaluated body args) (bc-compiled body args)))))
                    '((0.0 0.0) (0.0 -0.0) (-0.0 0.0) (-0.0 -0.0))))
           (filter (lambda (op) (= (scalar-op-arity op) 2)) (scalar-op-names))))
  (test-assert "add of -0.0 and -0.0 is -0.0, compiled and interpreted, as op and as +"
    (every (lambda (body)
             (and (eqv? -0.0 (bc-compiled body '(-0.0 -0.0)))
                  (eqv? -0.0 (bc-evaluated body '(-0.0 -0.0)))))
           '((add (in 0) (in 1)) (+ (in 0) (in 1)))))
  (test "the add combiner gives -0.0 for -0.0 and -0.0" -0.0 ((scalar-op-combiner 'add) -0.0 -0.0))
  (test "the add combiner adds integers exactly" 4 ((scalar-op-combiner 'add) 7 -3))
  (test-assert "nested bodies agree with eval-scalar-expr"
    (every (lambda (case)
             (sg-same? (bc-evaluated (car case) (cdr case)) (bc-compiled (car case) (cdr case))))
           (list (cons '(add (mul (in 0) (in 1)) (relu (in 2))) '(1.5 -2.0 0.25))
                 (cons '(if (> (in 0) 0.0) (sigmoid (in 0)) (tanh (in 1))) '(-0.3 0.8))
                 (cons '(let* ((a (exp (in 0))) (b (* a a))) (/ b (+ a 1.0))) '(0.7))
                 (cons '(let* ((a (in 0))) (if (< a 0.0) (let* ((a (- a))) (sqrt a)) a)) '(-4.0))
                 (cons '(fpabs (- (in 0) (in 1))) '(1.0 3.5))
                 (cons (list 'opaque (lambda (x y) (- (* x y) 1.0)) '(in 0) '(cast f32 (in 1))) '(2.0 0.1))
                 (cons '(sub (cast f32 (mul (in 0) (in 0))) (cast f32 (in 1))) '(0.1 0.2))
                 (cons 3.25 '()))))
  (test-assert "casts round as a store would, including NaN and infinities"
    (every (lambda (dt)
             (every (lambda (x)
                      (let ((cell (allocate-typed-vector dt 1)))
                        (typed-vector-set! cell dt 0 x)
                        (sg-same? (typed-vector-ref cell dt 0) (bc-compiled `(cast ,dt (in 0)) (list x)))))
                    (if (memq dt '(f32 f64))
                        sg-values
                        '(0.0 -0.0 1.5 -2.75 3.0 -7.9 1e6 -1e6))))
           '(f32 f64 s32 s64 u32 u64)))
  (test "a cast to f32 of 0.1 is the f32 value" (round-to-f32 0.1) (bc-compiled '(cast f32 (in 0)) '(0.1)))
  (test "a cast to s32 truncates towards zero" -2 (bc-compiled '(cast s32 (in 0)) '(-2.7)))
  (test "a cast to u32 keeps the magnitude" 2 (bc-compiled '(cast u32 (in 0)) '(-2.7)))
  (test-assert "every arity from 0 to 6 computes the sum"
    (every (lambda (n)
             (let ((body (let loop ((k 0) (e 0.0)) (if (= k n) e (loop (+ k 1) `(+ ,e (in ,k))))))
                   (args (map (lambda (k) (+ 1.0 k)) (iota n))))
               (= (bc-compiled body args) (fold + 0.0 args))))
           '(0 1 2 3 4 6)))
  (test-assert "the arguments keep their positions beyond three inputs"
    (equal? '(4.0 6.0) (list (bc-compiled '(in 3) '(1.0 2.0 3.0 4.0))
                             (bc-compiled '(in 5) '(1.0 2.0 3.0 4.0 5.0 6.0)))))
  (test-error "a wrong number of arguments beyond three inputs is an error"
    ((scalar-expr->procedure '(in 0) 4) 1.0 2.0))
  (test-error "(param k) is not supported" (scalar-expr->procedure '(param 0) 0))
  (test-error "(idx d) is not supported" (scalar-expr->procedure '(idx 0) 0))
  (test-error "(reduced) needs reduced?" (scalar-expr->procedure '(reduced) 0))
  (test "(reduced) alone is the reduced value" 2.5
        ((scalar-expr->procedure '(reduced) 0 reduced?: #t) 2.5))
  (test-assert "a finalize body agrees with eval-scalar-expr for 0, 2 and 4 inputs"
    (every (lambda (case)
             (let* ((body (car case)) (args (cdr case)) (r -1.25)
                    (f (scalar-expr->procedure body (length args) reduced?: #t)))
               (sg-same? (eval-scalar-expr body (lambda (k) (list-ref args k)) reduced: r)
                         (apply f (append args (list r))))))
           (list (cons '(sigmoid (cast f32 (reduced))) '())
                 (cons '(relu (add (reduced) (in 1))) '(0.5 3.0))
                 (cons '(add (mul (in 3) (reduced)) (in 0)) '(1.0 2.0 3.0 4.0)))))
  (test-error "the old form (acc) is not a valid body"
    (scalar-expr->procedure '(acc) 0 reduced?: #t))
  (test "scalar-expr-substitute-reduced replaces every (reduced)"
        '(add (in 0) (mul (cast f32 (in 1)) (cast f32 (in 1))))
        (scalar-expr-substitute-reduced '(add (in 0) (mul (reduced) (reduced))) '(cast f32 (in 1))))
  (test-error "an invalid body is rejected" (scalar-expr->procedure '(relu (in 0) (in 1)) 2))
  (let* ((calls 0)
         (f (lambda (x) (set! calls (+ calls 1)) (* x 3.0)))
         (shared (list 'opaque f '(in 0)))
         (bodies (list `(add ,shared (in 1)) `(mul ,shared (cast f32 (in 1))) shared))
         (proc (scalar-exprs->procedure bodies 2)))
    (test "several bodies give one value each" (list (+ 6.0 0.5) (* 6.0 0.5) 6.0)
      (vector->list (proc 2.0 0.5)))
    (test "a subexpression shared by the bodies is computed once per call" 1
      (begin (set! calls 0) (proc 2.0 0.5) calls))
    (test-assert "the result vector is reused by the next call" (eq? (proc 1.0 1.0) (proc 2.0 2.0))))
  (let* ((calls 0)
         (f (lambda (x) (set! calls (+ calls 1)) x))
         (proc (scalar-exprs->procedure (list (list 'opaque f '(in 0)) (list 'opaque f '(in 0))) 1)))
    (test "equal bodies that are not the same object are computed separately" 2
      (begin (proc 1.0) calls)))
  (let* ((calls 0)
         (f (lambda (x) (set! calls (+ calls 1)) x))
         (inner (list 'opaque f 't))
         (proc (scalar-exprs->procedure (list `(let* ((t (in 0))) ,inner) `(let* ((t (in 1))) ,inner)) 2)))
    (test "a subexpression referring to let* variables is not shared" '(1.0 2.0)
      (vector->list (proc 1.0 2.0)))
    (test "and is computed for each body" 4 (begin (proc 1.0 2.0) calls)))
  (test-assert "several bodies agree with eval-scalar-expr, casts and 5 inputs included"
    (let* ((p0 '(cast f32 (mul (in 0) (in 1))))
           (bodies (list `(add ,p0 (in 2)) p0 `(sub (in 4) (cast f32 (exp ,p0)))))
           (args '(0.1 0.7 -2.5 3.0 1.25))
           (vals (vector->list (apply (scalar-exprs->procedure bodies 5) args))))
      (every (lambda (b v) (sg-same? v (eval-scalar-expr b (lambda (k) (list-ref args k)))))
             bodies vals)))
  (test-error "an invalid body among several is rejected"
    (scalar-exprs->procedure (list '(in 0) '(relu (in 0) (in 1))) 2))
  (test-error "an input beyond the arity is rejected" (scalar-expr->procedure '(in 2) 2)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Canonical bodies, templates and backend kinds
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (canon body #!optional n)
  (call-with-values (lambda () (canonical-scalar-body body n)) list))

;; True when body and its canonical form agree on the inputs in-vals,
;; given to the canonical body in the order of the permutation.
(define (canon-agrees? body in-vals)
  (let* ((r (canon body (length in-vals)))
         (c (car r)) (perm (cadr r))
         (read-orig (lambda (k) (list-ref in-vals k)))
         (read-canon (lambda (i) (list-ref in-vals (list-ref perm i)))))
    (sg-same? (eval-scalar-expr body read-orig) (eval-scalar-expr c read-canon))))

(test-group "canonical bodies"
  (test "an input comes first in a product"
    '((mul (in 0) (cast f32 (relu-deriv (in 1)))) (1 0))
    (canon '(mul (cast f32 (relu-deriv (in 0))) (in 1))))
  (test "a canonical body is unchanged"
    '((mul (in 0) (cast f32 (relu-deriv (in 1)))) (0 1))
    (canon '(mul (in 0) (cast f32 (relu-deriv (in 1))))))
  (test "inputs are renumbered by first occurrence"
    '((add (in 0) (cast f32 (mul (in 1) (in 2)))) (2 0 1))
    (canon '(add (cast f32 (mul (in 0) (in 1))) (in 2))))
  (test "two inputs of an add keep their order, then are renumbered"
    '((add (in 0) (in 1)) (1 0))
    (canon '(add (in 1) (in 0))))
  (test "operands are ordered by structure"
    (car (canon '(mul (exp (in 0)) (abs (in 1)))))
    (car (canon '(mul (abs (in 1)) (exp (in 0))))))
  (test "a non-commutative op keeps its operands" '((sub (exp (in 0)) (in 1)) (0 1))
    (canon '(sub (exp (in 0)) (in 1))))
  (test "nested casts to one dtype are merged" '((cast f32 (exp (in 0))) (0))
    (canon '(cast f32 (cast f32 (exp (in 0))))))
  (test "casts to different dtypes are kept" '((cast f64 (cast f32 (in 0))) (0))
    (canon '(cast f64 (cast f32 (in 0)))))
  (test "unread inputs come last" '((exp (in 0)) (2 0 1)) (canon '(exp (in 2)) 3))
  (test "(reduced) is a leaf of a canonical body"
    '((mul (in 0) (reduced)) (0))
    (canon '(mul (reduced) (in 0))))
  (for-each
   (lambda (body)
     (test-assert (sprintf "canonical form is idempotent: ~S" body)
       (let ((c (car (canon body)))) (equal? c (car (canon c)))))
     (test-assert (sprintf "canonical form computes the same value: ~S" body)
       (canon-agrees? body '(0.75 -1.5 2.25))))
   '((mul (cast f32 (sigmoid-deriv (in 0))) (in 1))
     (add (cast f32 (mul (in 1) (in 0))) (in 2))
     (mul (in 2) (add (in 1) (exp (in 0))))
     (if (gt (in 0) (in 1)) (mul (in 1) (in 2)) (sub (in 0) (in 2)))
     (let* ((t (mul (in 0) (in 1)))) (add t (in 2))))))

(test-group "templates and backend kinds"
  (let* ((tin (make-template-input 'f32 (identity-index-map 2) #t))
         (tpl (make-segop-template #(2 3) (list tin)
                                   (list (make-template-output 'f32 '(exp (in 0)))))))
    (test "a template keeps its space" #(2 3) (segop-template-space tpl))
    (test "and its inputs" '((f32 #t)) (map (lambda (i) (list (template-input-dtype i) (template-input-flat? i)))
                                         (segop-template-ins tpl)))
    (test "and its outputs" '((f32 (exp (in 0))))
      (map (lambda (o) (list (template-output-dtype o) (template-output-body o)))
           (segop-template-outs tpl)))
    (let ((plain (make-segop-backend 'plain (lambda (sg) #f)))
          (cat   (make-segop-backend 'cat (lambda (sg) #f) kind: 'catalogue
                                     native?: (lambda (t) (equal? (segop-template-space t) #(2 3))))))
      (test "a backend is a host backend by default" 'host (segop-backend-kind plain))
      (test "and has no native query" #f (segop-backend-native? plain))
      (test "a catalogue backend has its kind" 'catalogue (segop-backend-kind cat))
      (test "no backend: no native backend" #f
        (parameterize ((segop-backends (list plain))) (segop-native? tpl)))
      (test "the backend whose query accepts the template is returned" 'cat
        (parameterize ((segop-backends (list plain cat)))
          (segop-backend-name (segop-native? tpl))))
      (test "a query that refuses gives #f" #f
        (parameterize ((segop-backends (list cat)))
          (segop-native? (make-segop-template #(4) (list tin) '())))))
    (test-error "an unknown kind is an error"
      (make-segop-backend 'x (lambda (sg) #f) kind: 'gpu)))
  (test "a template without nred describes a seg-map" 0
    (segop-template-nred (make-segop-template #(2) '() '())))
  (test "an output without a combine spec is a map output" '(#f #f #f)
    (let ((o (make-template-output 'f32 '(in 0))))
      (list (template-output-combine o) (template-output-finalize o) (template-output-map o))))
  (let* ((spec (make-combine-spec 'add 0.0 'f32))
         (m    (permutation-index-map '(1 0)))
         (o    (make-template-output 'f32 '(in 0) spec '(sqrt (reduced)) m))
         (tpl  (make-segop-template #(3 4) '() (list o) 1)))
    (test "a reduction template keeps nred" 1 (segop-template-nred tpl))
    (test-assert "a reduced output keeps its combine spec, finalize and map"
      (and (eq? spec (template-output-combine o))
           (equal? '(sqrt (reduced)) (template-output-finalize o))
           (eq? m (template-output-map o))))))


(test-exit)
