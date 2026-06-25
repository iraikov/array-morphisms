;;; test-array-morphisms-phase3.scm
;;; Test Suite for Phase 3: Basic Morphisms
;;;
;;; Tests binary arithmetic, unary operations, comparisons,
;;; and higher-order morphisms with broadcasting and type promotion

(import scheme (chicken base) test
        array-morphisms-core
        array-morphisms-basic-ops
        datatype matchable srfi-1 srfi-4)

;;; ============================================================================
;;; Test Utilities
;;; ============================================================================

(define (vectors-approx-equal? v1 v2 #!optional (tol 1e-6))
  "Test if two typed vectors are approximately equal"
  (and (= (vector-length v1) (vector-length v2))
       (every (lambda (i)
                (< (abs (- (vector-ref v1 i)
                          (vector-ref v2 i)))
                   tol))
              (iota (vector-length v1)))))

(define (morphisms-equal? m1 m2 #!optional (tol 1e-6))
  "Test if two concrete morphisms have equal data
   
   Note: Only works for concrete-array morphisms
         Abstract morphisms need to be realized first"
  (cases array-morphism m1
    (concrete-array (data1 shape1 strides1 offset1 dtype1 alloc1 batch1)
      (cases array-morphism m2
        (concrete-array (data2 shape2 strides2 offset2 dtype2 alloc2 batch2)
          (and (equal? shape1 shape2)
               (equal? dtype1 dtype2)
               (vectors-approx-equal? data1 data2 tol)))
        (else #f)))
    (else #f)))

;;; ============================================================================
;;; Test Group 1: Binary Arithmetic - Structure
;;; ============================================================================

(test-group "Binary Arithmetic - Structure"
  
  ;; Test 1.1: morph+ creates morphism-expr
  (let ((m1 (morph-from-list '(1 2 3) #(3) 'f64))
        (m2 (morph-from-list '(4 5 6) #(3) 'f64)))
    (let ((result (morph+ m1 m2)))
      (test-assert "morph+ creates morphism-expr with correct operation"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (and (eq? op 'add)
                 (= (length operands) 2)
                 (equal? shape #(3))
                 (eq? dtype 'f64)))
          (else #f)))))
  
  ;; Test 1.2: morph- creates correct structure
  (let ((m1 (morph-from-list '(10 20) #(2) 'f32))
        (m2 (morph-from-list '(1 2) #(2) 'f32)))
    (let ((result (morph- m1 m2)))
      (test-assert "morph- creates morphism-expr"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (and (eq? op 'sub)
                 (= (length operands) 2)))
          (else #f)))))
  
  ;; Test 1.3: morph* structure
  (let ((m1 (morph-from-list '(2 3) #(2) 's32))
        (m2 (morph-from-list '(4 5) #(2) 's32)))
    (let ((result (morph* m1 m2)))
      (test-assert "morph* creates morphism-expr"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (eq? op 'mul))
          (else #f)))))
  
  ;; Test 1.4: morph/ structure
  (let ((m1 (morph-from-list '(10.0 20.0) #(2) 'f64))
        (m2 (morph-from-list '(2.0 4.0) #(2) 'f64)))
    (let ((result (morph/ m1 m2)))
      (test-assert "morph/ creates morphism-expr"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (eq? op 'div))
          (else #f)))))
  
  ;; Test 1.5: morph-pow structure
  (let ((m1 (morph-from-list '(2.0 3.0) #(2) 'f64))
        (m2 (morph-from-list '(2.0 3.0) #(2) 'f64)))
    (let ((result (morph-pow m1 m2)))
      (test-assert "morph-pow creates morphism-expr"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (eq? op 'pow))
          (else #f)))))
)

;;; ============================================================================
;;; Test Group 2: Broadcasting
;;; ============================================================================

(test-group "Broadcasting"
  
  ;; Test 2.1: Same shape - no broadcasting
  (let ((m1 (morph-from-list '(1 2 3) #(3) 'f64))
        (m2 (morph-from-list '(4 5 6) #(3) 'f64)))
    (test "Same shape - result shape correct"
          #(3)
          (get-morphism-shape (morph+ m1 m2))))
  
  ;; Test 2.2: Scalar broadcasting
  (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
        (m2 (morph-from-list '(10) #(1) 'f64)))
    (test "Scalar broadcast - result shape"
          #(2 2)
          (get-morphism-shape (morph+ m1 m2))))
  
  ;; Test 2.3: Row broadcasting
  (let ((m1 (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64))
        (m2 (morph-from-list '(10 20 30) #(3) 'f64)))
    (test "Row broadcast - result shape"
          #(2 3)
          (get-morphism-shape (morph+ m1 m2))))
  
  ;; Test 2.4: Column broadcasting
  (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
        (m2 (morph-from-list '((10) (20)) #(2 1) 'f64)))
    (test "Column broadcast - result shape"
          #(2 2)
          (get-morphism-shape (morph+ m1 m2))))
  
  ;; Test 2.5: Different ranks
  (let ((m1 (morph-from-list '(((1 2))) #(1 1 2) 'f32))
        (m2 (morph-from-list '(10 20) #(2) 'f32)))
    (test "Different rank broadcast - result shape"
          #(1 1 2)
          (get-morphism-shape (morph* m1 m2))))
  
  ;; Test 2.6: Complex broadcasting
  (let ((m1 (morph-from-list '(((1))) #(1 1 1) 'f64))
        (m2 (morph-from-list '((2 3 4)) #(1 3) 'f64)))
    (test "Complex broadcast - result shape"
          #(1 1 3)
          (get-morphism-shape (morph+ m1 m2))))
)

;;; ============================================================================
;;; Test Group 3: Type Promotion
;;; ============================================================================

(test-group "Type Promotion"
  
  ;; Test 3.1: f64 + f32 -> f64
  (let ((m1 (morph-from-list '(1.0) #(1) 'f64))
        (m2 (morph-from-list '(2.0) #(1) 'f32)))
    (test "f64 + f32 -> f64"
          'f64
          (get-morphism-dtype (morph+ m1 m2))))
  
  ;; Test 3.2: f32 + s32 -> f32
  (let ((m1 (morph-from-list '(1.0) #(1) 'f32))
        (m2 (morph-from-list '(2) #(1) 's32)))
    (test "f32 + s32 -> f32"
          'f32
          (get-morphism-dtype (morph+ m1 m2))))
  
  ;; Test 3.3: s64 + s32 -> s64
  (let ((m1 (morph-from-list '(100) #(1) 's64))
        (m2 (morph-from-list '(200) #(1) 's32)))
    (test "s64 + s32 -> s64"
          's64
          (get-morphism-dtype (morph+ m1 m2))))
  
  ;; Test 3.4: Symmetric promotion
  (let ((m1 (morph-from-list '(1) #(1) 's32))
        (m2 (morph-from-list '(2.0) #(1) 'f64)))
    (test "s32 + f64 -> f64 (symmetric)"
          'f64
          (get-morphism-dtype (morph+ m1 m2))))
  
  ;; Test 3.5: Same type preservation
  (let ((m1 (morph-from-list '(1 2) #(2) 's32))
        (m2 (morph-from-list '(3 4) #(2) 's32)))
    (test "s32 + s32 -> s32"
          's32
          (get-morphism-dtype (morph+ m1 m2))))
)

;;; ============================================================================
;;; Test Group 4: Unary Operations - Structure
;;; ============================================================================

(test-group "Unary Operations - Structure"
  
  ;; Test 4.1: morph-sqrt structure
  (let ((m (morph-from-list '(4.0 9.0 16.0) #(3) 'f64)))
    (let ((result (morph-sqrt m)))
      (test-assert "morph-sqrt creates morphism-expr"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (and (eq? op 'sqrt)
                 (= (length operands) 1)
                 (equal? shape #(3))))
          (else #f)))))
  
  ;; Test 4.2: morph-exp structure
  (let ((m (morph-from-list '(0.0 1.0) #(2) 'f32)))
    (test-assert "morph-exp creates morphism-expr"
      (cases array-morphism (morph-exp m)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'exp))
        (else #f))))
  
  ;; Test 4.3: morph-log structure
  (let ((m (morph-from-list '(1.0 2.718281828) #(2) 'f64)))
    (test-assert "morph-log creates morphism-expr"
      (cases array-morphism (morph-log m)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'log))
        (else #f))))
  
  ;; Test 4.4: morph-sin structure
  (let ((m (morph-from-list '(0.0 1.5708) #(2) 'f64)))
    (test-assert "morph-sin creates morphism-expr"
      (cases array-morphism (morph-sin m)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'sin))
        (else #f))))
  
  ;; Test 4.5: morph-negate structure
  (let ((m (morph-from-list '(1 -2 3) #(3) 's32)))
    (test-assert "morph-negate creates morphism-expr"
      (cases array-morphism (morph-negate m)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'negate))
        (else #f))))
  
  ;; Test 4.6: morph-abs structure
  (let ((m (morph-from-list '(-5 3 -7) #(3) 's32)))
    (test-assert "morph-abs creates morphism-expr"
      (cases array-morphism (morph-abs m)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'abs))
        (else #f))))
)

;;; ============================================================================
;;; Test Group 5: Unary Type Inference
;;; ============================================================================

(test-group "Unary Type Inference"
  
  ;; Test 5.1: sqrt promotes s32 -> f64
  (let ((m (morph-from-list '(4 9) #(2) 's32)))
    (test "sqrt: s32 -> f64"
          'f64
          (get-morphism-dtype (morph-sqrt m))))
  
  ;; Test 5.2: sqrt preserves f32
  (let ((m (morph-from-list '(4.0 9.0) #(2) 'f32)))
    (test "sqrt: f32 -> f32"
          'f32
          (get-morphism-dtype (morph-sqrt m))))
  
  ;; Test 5.3: exp promotes s64 -> f64
  (let ((m (morph-from-list '(0 1) #(2) 's64)))
    (test "exp: s64 -> f64"
          'f64
          (get-morphism-dtype (morph-exp m))))
  
  ;; Test 5.4: log preserves f64
  (let ((m (morph-from-list '(1.0 2.0) #(2) 'f64)))
    (test "log: f64 -> f64"
          'f64
          (get-morphism-dtype (morph-log m))))
  
  ;; Test 5.5: negate preserves s32
  (let ((m (morph-from-list '(1 2) #(2) 's32)))
    (test "negate: s32 -> s32"
          's32
          (get-morphism-dtype (morph-negate m))))
  
  ;; Test 5.6: abs preserves f64
  (let ((m (morph-from-list '(-1.0 2.0) #(2) 'f64)))
    (test "abs: f64 -> f64"
          'f64
          (get-morphism-dtype (morph-abs m))))
  
  ;; Test 5.7: floor preserves s64
  (let ((m (morph-from-list '(1 2) #(2) 's64)))
    (test "floor: s64 -> s64"
          's64
          (get-morphism-dtype (morph-floor m))))
)

;;; ============================================================================
;;; Test Group 6: Comparison Operations
;;; ============================================================================

(test-group "Comparison Operations"
  
  ;; Test 6.1: morph> structure
  (let ((m1 (morph-from-list '(3 1 4) #(3) 'f64))
        (m2 (morph-from-list '(2 2 2) #(3) 'f64)))
    (let ((result (morph> m1 m2)))
      (test-assert "morph> creates morphism-expr"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (and (eq? op 'gt)
                 (= (length operands) 2)
                 (equal? shape #(3))
                 (eq? dtype 'f64)))
          (else #f)))))
  
  ;; Test 6.2: morph< structure
  (let ((m1 (morph-from-list '(1 5) #(2) 's32))
        (m2 (morph-from-list '(2 4) #(2) 's32)))
    (let ((result (morph< m1 m2)))
      (test-assert "morph< creates morphism-expr with f64 result"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (and (eq? op 'lt)
                 (eq? dtype 'f64)))
          (else #f)))))
  
  ;; Test 6.3: morph= structure
  (let ((m1 (morph-from-list '(1.0 2.0) #(2) 'f32))
        (m2 (morph-from-list '(1.0 3.0) #(2) 'f32)))
    (test-assert "morph= creates morphism-expr"
      (cases array-morphism (morph= m1 m2)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'eq))
        (else #f))))
  
  ;; Test 6.4: morph>= structure
  (let ((m1 (morph-from-list '(3 2 1) #(3) 'f64))
        (m2 (morph-from-list '(1 2 3) #(3) 'f64)))
    (test-assert "morph>= creates morphism-expr"
      (cases array-morphism (morph>= m1 m2)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'ge))
        (else #f))))
  
  ;; Test 6.5: morph<= structure
  (let ((m1 (morph-from-list '(1 2) #(2) 's64))
        (m2 (morph-from-list '(2 1) #(2) 's64)))
    (test-assert "morph<= creates morphism-expr"
      (cases array-morphism (morph<= m1 m2)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch)
          (eq? op 'le))
        (else #f))))
  
  ;; Test 6.6: Comparison broadcasting
  (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64))
        (m2 (morph-from-list '(2) #(1) 'f64)))
    (test "Comparison with broadcast"
          #(2 2)
          (get-morphism-shape (morph> m1 m2))))
  
  ;; Test 6.7: Comparison always returns f64
  (let ((m1 (morph-from-list '(1 2) #(2) 's32))
        (m2 (morph-from-list '(2 1) #(2) 's32)))
    (test "Comparison: s32 inputs -> f64 result"
          'f64
          (get-morphism-dtype (morph> m1 m2))))
)

;;; ============================================================================
;;; Test Group 7: morph-map
;;; ============================================================================

(test-group "morph-map"
  
  ;; Test 7.1: morph-map structure
  (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
    (let ((result (morph-map (lambda (x) (* x x)) m)))
      (test-assert "morph-map creates morphism-expr"
        (cases array-morphism result
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (and (eq? op 'map)
                 (= (length operands) 1)
                 (equal? shape #(3))))
          (else #f)))))
  
  ;; Test 7.2: morph-map preserves shape
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f32)))
    (test "morph-map preserves shape"
          #(2 2)
          (get-morphism-shape (morph-map (lambda (x) (+ x 10)) m))))
  
  ;; Test 7.3: morph-map preserves dtype
  (let ((m (morph-from-list '(1 2 3) #(3) 's32)))
    (test "morph-map preserves dtype"
          's32
          (get-morphism-dtype (morph-map abs m))))
  
  ;; Test 7.4: morph-map preserves batch axis
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0)))
    (test "morph-map preserves batch axis"
          0
          (get-morphism-batch-axis (morph-map (lambda (x) x) m))))
  
  ;; Test 7.5: morph-map with complex function
  (let ((m (morph-from-list '(0.0 1.0) #(2) 'f64)))
    (test-assert "morph-map with complex function"
      (cases array-morphism (morph-map (lambda (x) (sin (* x 3.14159))) m)
        (morphism-expr (_ op operands idx-fn shape dtype meta batch) #t)
        (else #f))))
)

;;; ============================================================================
;;; Test Group 8: morph-reduce
;;; ============================================================================

(test-group "morph-reduce"
  
  ;; Test 8.1: morph-reduce structure
  (let ((m (morph-from-list '(1 2 3 4) #(4) 'f64)))
    (let ((result (morph-reduce 'sum m)))
      (test-assert "morph-reduce creates reduction-morphism"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (and (eq? op 'sum)
                 (equal? shape #())
                 (eq? dtype 'f64)))
          (else #f)))))
  
  ;; Test 8.2: Reduce along axis 0
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))
    (let ((result (morph-reduce 'sum m '(0))))
      (test-assert "Reduce axis 0 - shape"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #(2)))
          (else #f)))))
  
  ;; Test 8.3: Reduce along axis 1
  (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f32)))
    (let ((result (morph-reduce 'mean m '(1))))
      (test-assert "Reduce axis 1 - shape"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #(2)))
          (else #f)))))
  
  ;; Test 8.4: Reduce with keepdims
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64)))
    (let ((result (morph-reduce 'sum m '(0) #t)))
      (test-assert "Reduce with keepdims - shape"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #(1 2)))
          (else #f)))))
  
  ;; Test 8.5: Reduce multiple axes
  (let ((m (morph-from-list '(((1 2) (3 4)) ((5 6) (7 8))) #(2 2 2) 'f64)))
    (let ((result (morph-reduce 'max m '(0 2))))
      (test-assert "Reduce multiple axes - shape"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #(2)))
          (else #f)))))
  
  ;; Test 8.6: Reduce all axes (default)
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f32)))
    (let ((result (morph-reduce 'prod m)))
      (test-assert "Reduce all axes - scalar result"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #()))
          (else #f)))))
  
  ;; Test 8.7: Reduce with negative axis
  (let ((m (morph-from-list '((1 2 3) (4 5 6)) #(2 3) 'f64)))
    (let ((result (morph-reduce 'sum m '(-1))))
      (test-assert "Reduce with negative axis"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #(2)))
          (else #f)))))
  
  ;; Test 8.8: Mean promotes integer to f64
  (let ((m (morph-from-list '(1 2 3 4) #(4) 's32)))
    (let ((result (morph-reduce 'mean m)))
      (test-assert "Mean: s32 -> f64"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (eq? dtype 'f64))
          (else #f)))))
  
  ;; Test 8.9: Sum preserves integer type
  (let ((m (morph-from-list '(1 2 3) #(3) 's64)))
    (let ((result (morph-reduce 'sum m)))
      (test-assert "Sum preserves s64"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (eq? dtype 's64))
          (else #f)))))
)

;;; ============================================================================
;;; Test Group 9: Batch Axis Tracking
;;; ============================================================================

(test-group "Batch Axis Tracking"
  
  ;; Test 9.1: Binary op - both not batched
  (let ((m1 (morph-from-list '(1 2) #(2) 'f64))
        (m2 (morph-from-list '(3 4) #(2) 'f64)))
    (test "Binary op: both not batched -> not batched"
          -1
          (get-morphism-batch-axis (morph+ m1 m2))))
  
  ;; Test 9.2: Binary op - one batched, one not
  (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0))
        (m2 (morph-from-list '(10 20) #(2) 'f64)))
    (test "Binary op: one batched -> batched"
          0
          (get-morphism-batch-axis (morph+ m1 m2))))
  
  ;; Test 9.3: Binary op - both batched, same axis
  (let ((m1 (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0))
        (m2 (morph-from-list '((5 6) (7 8)) #(2 2) 'f64 batch-axis: 0)))
    (test "Binary op: both batched (same axis) -> batched"
          0
          (get-morphism-batch-axis (morph+ m1 m2))))
  
  ;; Test 9.4: Unary op preserves batch axis
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0)))
    (test "Unary op preserves batch axis"
          0
          (get-morphism-batch-axis (morph-sqrt m))))
  
  ;; Test 9.5: Unary op preserves non-batched
  (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
    (test "Unary op preserves non-batched"
          -1
          (get-morphism-batch-axis (morph-exp m))))
  
  ;; Test 9.6: Reduce removes batch axis if reduced
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0)))
    (let ((result (morph-reduce 'sum m '(0))))
      (test-assert "Reduce along batch axis -> not batched"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (= batch -1))
          (else #f)))))
  
  ;; Test 9.7: Reduce keeps batch axis if not reduced
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0)))
    (let ((result (morph-reduce 'sum m '(1))))
      (test-assert "Reduce non-batch axis -> still batched"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (= batch 0))
          (else #f)))))
  
  ;; Test 9.8: Reduce with keepdims preserves batch axis
  (let ((m (morph-from-list '((1 2) (3 4)) #(2 2) 'f64 batch-axis: 0)))
    (let ((result (morph-reduce 'sum m '(0) #t)))
      (test-assert "Reduce with keepdims preserves batch axis"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (= batch 0))
          (else #f)))))
)

;;; ============================================================================
;;; Test Group 10: Edge Cases and Error Handling
;;; ============================================================================

(test-group "Edge Cases"
  
  ;; Test 10.1: Empty axes in reduce = reduce all
  (let ((m (morph-from-list '(1 2 3 4) #(4) 'f64)))
    (let ((result (morph-reduce 'sum m '())))
      (test-assert "Empty axes reduces all"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #()))
          (else #f)))))
  
  ;; Test 10.2: Single element reduction
  (let ((m (morph-from-list '(42) #(1) 'f64)))
    (let ((result (morph-reduce 'max m)))
      (test-assert "Single element reduction"
        (cases array-morphism result
          (reduction-morphism (_ op operand axes idx-fn shape dtype batch)
            (equal? shape #()))
          (else #f)))))
  
  ;; Test 10.3: Identity map function
  (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
    (test "Identity map function"
          (get-morphism-shape m)
          (get-morphism-shape (morph-map (lambda (x) x) m))))
  
  ;; Test 10.4: Chained operations
  (let ((m (morph-from-list '(1 2 3) #(3) 'f64)))
    (let* ((squared (morph-map (lambda (x) (* x x)) m))
           (plus-one (morph+ squared (morph-from-list '(1) #(1) 'f64)))
           (final (morph-sqrt plus-one)))
      (test-assert "Chained operations create valid structure"
        (cases array-morphism final
          (morphism-expr (_ op operands idx-fn shape dtype meta batch)
            (eq? op 'sqrt))
          (else #f)))))
  
  ;; Test 10.5: Broadcasting with leading singleton dimensions
  (let ((m1 (morph-from-list '(((1 2))) #(1 1 2) 'f64))
        (m2 (morph-from-list '(3 4) #(2) 'f64)))
    (test "Leading singleton broadcasting"
          #(1 1 2)
          (get-morphism-shape (morph* m1 m2))))
)

;;; ============================================================================
;;; Run Tests
;;; ============================================================================

(test-exit)
