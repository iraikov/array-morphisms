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

(import scheme (chicken base) test
        (only srfi-1 every iota fold make-list)
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
                 '((in 2) (idx 3) (relu (in 0) (in 1)) (foo (in 0)) (acc) unbound-var
                   (let* ((a)) a)))))
  (test-assert "(acc) is valid in a finalize body" (scalar-expr-valid? '(/ (acc) 4.0) 0 0 acc?: #t))
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
  (test "acc in a finalize body" 2.5 (eval-scalar-expr '(/ (acc) 4.0) (sg-reader) acc: 10.0))
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
         (not (scalar-expr-opaque? '(relu (in 0)))))))

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
    (segop-interpret (sg-col-sum 'f64 2 3 (make-combine-spec 'add 0.0 'f64) '(/ (acc) 2.0))
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
                                                  '(relu (add (acc) (in 2))))))))
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
                                           (make-combine-spec 'add 0.0 'f64) '(add (acc) (in 0))))))
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
      (parameterize ((segop-backends '())) (select-segop-executor sm)))))

(test-exit)
