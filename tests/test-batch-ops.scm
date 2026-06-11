;;; test-batch-ops.scm
;;; Batch Operations Test Suite
;;;
;;; Organisation:
;;;   Group 1  - Batch dimension management
;;;   Group 2  - Core batch combinators
;;;   Group 3  - Batch broadcasting
;;;   Group 4  - Batch-aware arithmetic
;;;   Group 5  - Batched matrix multiply (morph-batch-matmul)
;;;   Group 6  - BLAS in context (realize-morphism-expr/ctx with BLAS)
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s tests/test-batch-ops.scm

(import scheme (chicken base))
(import test)
(import (only srfi-1 every iota make-list fold))
(import datatype)
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-blas-exec)
(import array-morphisms-context)
(import array-morphisms-batch-ops)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Test Utilities
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (approx= a b #!optional (tol 1e-9))
  (< (abs (- a b)) tol))

(define (flatten-nested-list lst)
  (cond ((null? lst) '())
        ((pair? lst) (append (flatten-nested-list (car lst))
                             (flatten-nested-list (cdr lst))))
        (else (list lst))))

(define (morph->flat m)
  (let ((lst (morph->list m)))
    (if (pair? lst) (flatten-nested-list lst) (list lst))))

(define (flat-approx? m expected-list #!optional (tol 1e-9))
  (let ((actual (morph->flat (realize m))))
    (and (= (length actual) (length expected-list))
         (every (lambda (a e) (approx= a e tol)) actual expected-list))))

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) #f)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1: Batch Dimension Management
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 7 - Batch Dimension Management"

  (test-assert "add-batch-dimension: shape (3,) -> (1,3)"
    (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b (add-batch-dimension m)))
      (equal? (get-morphism-shape b) #(1 3))))

  (test-assert "add-batch-dimension: result is batched"
    (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b (add-batch-dimension m)))
      (batched? b)))

  (test-assert "add-batch-dimension: batch-axis is 0"
    (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b (add-batch-dimension m)))
      (= (get-morphism-batch-axis b) 0)))

  (test-assert "add-batch-dimension: batch-size is 1"
    (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b (add-batch-dimension m)))
      (= (batch-size b) 1)))

  (test-assert "add-batch-dimension: values preserved after realize"
    (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b (add-batch-dimension m)))
      (flat-approx? b '(1.0 2.0 3.0))))

  (test-assert "add-batch-dimension: non-zero axis"
    (let* ((m (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (b (add-batch-dimension m 1)))
      (and (equal? (get-morphism-shape b) #(2 1 2))
           (= (get-morphism-batch-axis b) 1))))

  (test-assert "remove-batch-dimension: inverts add-batch-dimension"
    (let* ((m  (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b  (add-batch-dimension m))
           (m2 (remove-batch-dimension b)))
      (and (equal? (get-morphism-shape m2) #(3))
           (not (batched? m2)))))

  (test-assert "remove-batch-dimension: values preserved"
    (let* ((m  (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b  (add-batch-dimension m))
           (m2 (remove-batch-dimension b)))
      (flat-approx? m2 '(1.0 2.0 3.0))))

  (test-assert "extract-batch-element: extracts correct slice"
    ;; Build a (3,2) batched array: rows are [1,2], [3,4], [5,6]
    (let* ((data (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64))
           (e1   (extract-batch-element data 1)))
      (and (equal? (get-morphism-shape e1) #(2))
           (not (batched? e1))
           (flat-approx? e1 '(3.0 4.0)))))

  (test-assert "extract-batch-element: all three elements correct"
    (let* ((data (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64)))
      (and (flat-approx? (extract-batch-element data 0) '(1.0 2.0))
           (flat-approx? (extract-batch-element data 1) '(3.0 4.0))
           (flat-approx? (extract-batch-element data 2) '(5.0 6.0)))))

  (test-assert "stack-into-batch: produces correct shape"
    (let* ((a (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b (morph-from-list '(3.0 4.0) #(2) 'f64))
           (c (morph-from-list '(5.0 6.0) #(2) 'f64))
           (s (stack-into-batch (list a b c))))
      (and (equal? (get-morphism-shape s) #(3 2))
           (batched? s)
           (= (get-morphism-batch-axis s) 0))))

  (test-assert "stack-into-batch: values preserved"
    (let* ((a (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b (morph-from-list '(3.0 4.0) #(2) 'f64))
           (s (stack-into-batch (list a b))))
      (flat-approx? s '(1.0 2.0 3.0 4.0))))

  (test-assert "concat-batch: shape is concatenated"
    (let* ((a (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (b (morph-from-list '((5.0 6.0) (7.0 8.0) (9.0 10.0)) #(3 2) 'f64))
           ;; mark as batched by realizing via stack
           (sa (stack-into-batch
                (list (morph-from-list '(1.0 2.0) #(2) 'f64)
                      (morph-from-list '(3.0 4.0) #(2) 'f64))))
           (sb (stack-into-batch
                (list (morph-from-list '(5.0 6.0) #(2) 'f64)
                      (morph-from-list '(7.0 8.0) #(2) 'f64)
                      (morph-from-list '(9.0 10.0) #(2) 'f64))))
           (c  (concat-batch (list sa sb))))
      (equal? (get-morphism-shape c) #(5 2)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2: Core Batch Combinators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 7 - Core Batch Combinators"

  (test-assert "batch-map identity: shape unchanged"
    (let* ((m (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64))
           (r (batch-map (lambda (x) x) m)))
      (equal? (get-morphism-shape r) #(3 2))))

  (test-assert "batch-map identity: values unchanged"
    (let* ((m (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64))
           (r (batch-map (lambda (x) x) m)))
      (flat-approx? r '(1.0 2.0 3.0 4.0 5.0 6.0))))

  (test-assert "batch-map morph-negate: values negated"
    (let* ((m (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (r (batch-map morph-negate m)))
      (flat-approx? r '(-1.0 -2.0 -3.0 -4.0))))

  (test-assert "batch-map with shape-changing fn: sum along elem axis"
    ;; m shape (3, 4): each batch element (4,) -> sum = scalar
    ;; fn: morph-reduce 'sum over axis 0 of a 1-D array
    (let* ((m (stack-into-batch
               (list (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64)
                     (morph-from-list '(5.0 6.0 7.0 8.0) #(4) 'f64)
                     (morph-from-list '(9.0 10.0 11.0 12.0) #(4) 'f64))))
           (r (batch-map (lambda (x) (morph-reduce 'sum x)) m)))
      ;; result shape (3,) with sums 10, 26, 42
      (and (equal? (get-morphism-shape r) #(3))
           (flat-approx? r '(10.0 26.0 42.0)))))

  (test-assert "batch-reduce sum: shape"
    (let* ((m (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64))
           (r (batch-reduce 'sum m)))
      (equal? (get-morphism-shape r) #(2))))

  (test-assert "batch-reduce sum: values"
    (let* ((m (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64))
           (r (batch-reduce 'sum m)))
      ;; sum of columns: (1+3+5)=9, (2+4+6)=12
      (flat-approx? r '(9.0 12.0))))

  (test-assert "batch-reduce with keepdims"
    (let* ((m (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (r (batch-reduce 'sum m #t)))
      (equal? (get-morphism-shape r) #(1 2))))

  (test-assert "batch-zip morph+: element-wise sum"
    (let* ((a (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (b (morph-from-list '(5.0 6.0 7.0 8.0) #(2 2) 'f64))
           (r (batch-zip morph+ a b)))
      (flat-approx? r '(6.0 8.0 10.0 12.0))))

  (test-assert "batch-fold morph+: result equals batch-reduce sum"
    (let* ((data (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64))
           (zero (morph-from-list '(0.0 0.0) #(2) 'f64))
           (folded (batch-fold morph+ zero data))
           (reduced (batch-reduce 'sum data)))
      (flat-approx? folded (morph->flat (realize reduced)))))

  (test-assert "batch-scan morph+: running cumulative sums"
    ;; A (3,2): [1,2],[3,4],[5,6]
    ;; scan: [1,2], [1+3,2+4]=[4,6], [4+5,6+6]=[9,12]
    (let* ((data (stack-into-batch
                  (list (morph-from-list '(1.0 2.0) #(2) 'f64)
                        (morph-from-list '(3.0 4.0) #(2) 'f64)
                        (morph-from-list '(5.0 6.0) #(2) 'f64))))
           (zero (morph-from-list '(0.0 0.0) #(2) 'f64))
           (scan (batch-scan morph+ zero data)))
      (and (equal? (get-morphism-shape scan) #(3 2))
           (flat-approx? scan '(1.0 2.0 4.0 6.0 9.0 12.0))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3: Batch Broadcasting
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 7 - Batch Broadcasting"

  (test-assert "broadcast-to-batch: non-batched gets size-1 batch dim"
    (let* ((m (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b (broadcast-to-batch m 4)))
      (and (batched? b)
           (= (batch-size b) 1)
           (equal? (get-morphism-shape b) #(1 3)))))

  (test-assert "broadcast-to-batch: batched size-1 returned as-is"
    (let* ((m (add-batch-dimension (morph-from-list '(1.0 2.0) #(2) 'f64)))
           (b (broadcast-to-batch m 8)))
      (eq? b m)))

  (test-assert "broadcast-to-batch: correct size returned as-is"
    (let* ((m (stack-into-batch
                (list (morph-from-list '(1.0 2.0) #(2) 'f64)
                      (morph-from-list '(3.0 4.0) #(2) 'f64))))
           (b (broadcast-to-batch m 2)))
      (eq? b m)))

  (test-assert "broadcast-to-batch: mismatch signals error"
    (let* ((m (stack-into-batch
                (list (morph-from-list '(1.0 2.0) #(2) 'f64)
                      (morph-from-list '(3.0 4.0) #(2) 'f64)))))
      (condition-case
          (begin (broadcast-to-batch m 5) #f)
        (e (exn) #t)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4: Batch-Aware Arithmetic
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 7 - Batch-Aware Arithmetic"

  (test-assert "morph+/batch: both batched same size"
    (let* ((a (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (b (morph-from-list '(10.0 20.0 30.0 40.0) #(2 2) 'f64))
           (r (morph+/batch a b)))
      (flat-approx? r '(11.0 22.0 33.0 44.0))))

  (test-assert "morph+/batch: left batched right not (broadcast)"
    ;; a (2,3) batched, b (3,) not batched
    ;; result: row 0 = [1+10,2+20,3+30]=[11,22,33]
    ;;         row 1 = [4+10,5+20,6+30]=[14,25,36]
    (let* ((a (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(2 3) 'f64))
           (b (morph-from-list '(10.0 20.0 30.0) #(3) 'f64))
           (r (morph+/batch a b)))
      (flat-approx? r '(11.0 22.0 33.0 14.0 25.0 36.0))))

  (test-assert "morph+/batch: right batched left not (broadcast)"
    (let* ((a (morph-from-list '(10.0 20.0 30.0) #(3) 'f64))
           (b (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(2 3) 'f64))
           (r (morph+/batch a b)))
      (flat-approx? r '(11.0 22.0 33.0 14.0 25.0 36.0))))

  (test-assert "morph+/batch: neither batched falls through to morph+"
    (let* ((a (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b (morph-from-list '(3.0 4.0) #(2) 'f64))
           (r (morph+/batch a b)))
      (flat-approx? r '(4.0 6.0))))

  (test-assert "morph*/batch: batched x non-batched"
    (let* ((a (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (b (morph-from-list '(2.0 3.0) #(2) 'f64))
           (r (morph*/batch a b)))
      ;; row 0: [1*2, 2*3]=[2,6]; row 1: [3*2, 4*3]=[6,12]
      (flat-approx? r '(2.0 6.0 6.0 12.0))))

  (test-assert "morph-/batch: basic subtraction"
    (let* ((a (morph-from-list '(5.0 6.0 7.0 8.0) #(2 2) 'f64))
           (b (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
           (r (morph-/batch a b)))
      (flat-approx? r '(4.0 4.0 4.0 4.0))))

  (test-assert "morph-div/batch: basic division"
    (let* ((a (morph-from-list '(6.0 8.0 12.0 16.0) #(2 2) 'f64))
           (b (morph-from-list '(2.0 4.0 3.0 8.0) #(2 2) 'f64))
           (r (morph-div/batch a b)))
      (flat-approx? r '(3.0 2.0 4.0 2.0))))

  (test-assert "morph+/batch: batch mismatch signals error"
    (let* ((a (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b (morph-from-list '(1.0 2.0) #(2) 'f64)))
      ;; Both non-batched, so morph+/batch falls through to morph+.
      ;; We need batched inputs with different sizes for the error.
      (let ((ba (morph-from-list '(1.0 2.0 3.0 4.0) #(2 2) 'f64))
            (bb (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(3 2) 'f64)))
        (condition-case
            (begin (morph+/batch ba bb) #f)
          (e (exn) #t))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5: Batched Matrix Multiply
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 7 - Batched Matrix Multiply"

  (test-assert "morph-batch-matmul: non-batched delegates to morph-matmul"
    (let* ((A (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (B (morph-from-list '((1.0 0.0) (0.0 1.0)) #(2 2) 'f64))
           (r (morph-batch-matmul A B)))
      ;; A * I = A
      (flat-approx? r '(1.0 2.0 3.0 4.0))))

  (test-assert "morph-batch-matmul: (N,M,K) x (K,P) shape"
    ;; A (2,2,3), B (3,4) -> result (2,2,4)
    (let* ((A (morph-from-list (make-list 12 1.0) #(2 2 3) 'f64))
           (B (morph-from-list (make-list 12 1.0) #(3 4) 'f64))
           (r (morph-batch-matmul A B)))
      (equal? (get-morphism-shape r) #(2 2 4))))

  (test-assert "morph-batch-matmul: (N,M,K) x (K,P) values correct"
    ;; A (2,2,2): [[1,2],[3,4]], [[5,6],[7,8]]
    ;; B (2,2): identity [[1,0],[0,1]]
    ;; result: same as A
    (let* ((A (stack-into-batch
               (list (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64)
                     (morph-from-list '((5.0 6.0) (7.0 8.0)) #(2 2) 'f64))))
           (I (morph-from-list '((1.0 0.0) (0.0 1.0)) #(2 2) 'f64))
           (r (morph-batch-matmul A I)))
      (flat-approx? r '(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0))))

  (test-assert "morph-batch-matmul: (N,M,K) x (N,K,P) shape"
    ;; A (3,2,4), B (3,4,5) -> result (3,2,5)
    (let* ((A (morph-from-list (make-list 24 1.0) #(3 2 4) 'f64))
           (B (morph-from-list (make-list 60 1.0) #(3 4 5) 'f64))
           (r (morph-batch-matmul A B)))
      (equal? (get-morphism-shape r) #(3 2 5))))

  (test-assert "morph-batch-matmul: (N,M,K) x (N,K,P) values correct"
    ;; A (2,1,2): [[1,2]], [[3,4]]
    ;; B (2,2,1): [[5],[6]], [[7],[8]]
    ;; result (2,1,1): [[1*5+2*6]]=[17], [[3*7+4*8]]=[53]
    (let* ((A (stack-into-batch
               (list (morph-from-list '(1.0 2.0) #(1 2) 'f64)
                     (morph-from-list '(3.0 4.0) #(1 2) 'f64))))
           (B (stack-into-batch
               (list (morph-from-list '(5.0 6.0) #(2 1) 'f64)
                     (morph-from-list '(7.0 8.0) #(2 1) 'f64))))
           (r (morph-batch-matmul A B)))
      (flat-approx? r '(17.0 53.0))))

  (test-assert "morph-batch-matmul: BLAS vs fallback equal"
    (let* ((A (stack-into-batch
               (list (morph-from-list '((1.0 2.0 3.0) (4.0 5.0 6.0)) #(2 3) 'f64)
                     (morph-from-list '((7.0 8.0 9.0) (10.0 11.0 12.0)) #(2 3) 'f64))))
           (B (morph-from-list '((1.0 0.0) (0.0 1.0) (1.0 1.0)) #(3 2) 'f64))
           (expr (morph-batch-matmul A B)))
      (enable-blas!)
      (let ((blas-result (morph->flat (realize expr))))
        (disable-blas!)
        (let ((scheme-result (morph->flat (realize expr))))
          (enable-blas!)
          (and (= (length blas-result) (length scheme-result))
               (every (lambda (a b) (approx= a b 1e-6))
                      blas-result scheme-result)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6: BLAS in Context (realize-morphism-expr/ctx Phase 7 fix)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "Phase 7 - BLAS in Context"

  (test-assert "matmul in context: trace values match non-context values"
    (let* ((A    (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (B    (morph-from-list '((5.0 6.0) (7.0 8.0)) #(2 2) 'f64))
           (expr (morph-matmul A B))
           (ref  (morph->flat (realize expr)))
           (ctx  (make-morphism-context))
           (ctx-result (morph->flat (realize/ctx ctx expr))))
      (and (= (length ref) (length ctx-result))
           (every (lambda (a b) (approx= a b 1e-9)) ref ctx-result))))

  (test-assert "matmul in context: replay values match trace values"
    (let* ((A    (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (B    (morph-from-list '((5.0 6.0) (7.0 8.0)) #(2 2) 'f64))
           (expr (morph-matmul A B))
           (ctx  (make-morphism-context))
           (trace-result (morph->flat (realize/ctx ctx expr))))
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((replay-result (morph->flat (realize/ctx ctx expr))))
        (and (= (length trace-result) (length replay-result))
             (every (lambda (a b) (approx= a b 1e-9))
                    trace-result replay-result)))))

  (test-assert "batch-map matmul in context: trace then replay correct"
    ;; A (2,2,2), B (2,2) shared weight
    (let* ((A    (stack-into-batch
                  (list (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64)
                        (morph-from-list '((5.0 6.0) (7.0 8.0)) #(2 2) 'f64))))
           (B    (morph-from-list '((1.0 0.0) (0.0 1.0)) #(2 2) 'f64))
           (expr (morph-batch-matmul A B))
           (ref  (morph->flat (realize expr)))
           (ctx  (make-morphism-context))
           (trace-result (morph->flat (realize/ctx ctx expr))))
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((replay-result (morph->flat (realize/ctx ctx expr))))
        ;; trace matches reference, replay matches trace
        (and (every (lambda (a b) (approx= a b 1e-9)) ref trace-result)
             (every (lambda (a b) (approx= a b 1e-9))
                    trace-result replay-result)))))

  (test-assert "matvec in context: trace then replay correct"
    (let* ((A    (morph-from-list '((1.0 2.0 3.0) (4.0 5.0 6.0)) #(2 3) 'f64))
           (v    (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (expr (morph-matvec A v))
           (ref  (morph->flat (realize expr)))
           (ctx  (make-morphism-context))
           (trace-result (morph->flat (realize/ctx ctx expr))))
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((replay-result (morph->flat (realize/ctx ctx expr))))
        (and (every (lambda (a b) (approx= a b 1e-9)) ref trace-result)
             (every (lambda (a b) (approx= a b 1e-9))
                    trace-result replay-result))))))

(test-exit)
