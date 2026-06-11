;;; example-09-linear-regression.scm
;;; Integration Example: Autograd + Batch Matmul + Buffer Reuse
;;; Domain: Fit y = x * W^T + b by gradient descent
;;;
;;; This example combines several advantages of array morphisms:
;;;
;;;   Cross-boundary fusion: the forward pass expression
;;;     tree X@W^T + b -> residual -> MSE loss is one lazy morphism
;;;     tree; no intermediate materialises until backward! is called.
;;;     The gradient morphisms produced by backward! are also lazy and
;;;     compose into the same unified algebra.
;;;
;;;   Unified abstraction: the arithmetic of the forward
;;;     pass (matmul, add, subtract, square, mean) and the structural
;;;     reshape inside var-transpose are all the same kind of object
;;;     and thread through backward! without type-boundary copies.
;;;
;;;   Batch awareness: X has shape [N, 2] and var-matmul
;;;     handles the [N, 2] x [2, 1] product across the batch dimension
;;;     without a user-written loop over N.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-09-linear-regression.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota map))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-batch-ops)
(import array-morphisms-blas-exec)
(import array-morphisms-grad)
(import array-morphisms-context)

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) 0)))

(format #t "=== Example 9: Linear Regression via Gradient Descent ===~%~%")

;;; -----------------------------------------------------------------------
;;; Dataset
;;; True parameters: W_true = [[2.0, -1.0]], b_true = 0.5
;;; x0[i] = i/10, x1[i] = (9-i)/10  for i = 0..9
;;; y[i]  = 2*x0[i] - x1[i] + 0.5
;;; -----------------------------------------------------------------------
(define N 10)

(define x-data
  (apply append
         (map (lambda (i) (list (/ i 10.0) (/ (- 9 i) 10.0)))
              (iota N))))

(define y-data
  (map (lambda (i)
         (let ((x0 (/ i 10.0))
               (x1 (/ (- 9 i) 10.0)))
           (+ (* 2.0 x0) (* -1.0 x1) 0.5)))
       (iota N)))

;;; Morphisms for the fixed dataset (no gradient needed)
(define X-morph (morph-from-list x-data #(10 2) 'f64))   ; shape [10, 2]
(define y-morph (morph-from-list y-data #(10 1) 'f64))   ; shape [10, 1]

(format #t "Dataset: ~a samples, 2 features~%" N)
(format #t "True parameters: W = [[2.0, -1.0]], b = 0.5~%~%")

;;; -----------------------------------------------------------------------
;;; One training step: forward pass, backward pass, SGD update.
;;; Returns (list W-new b-new loss-val).
;;; -----------------------------------------------------------------------
(define (train-step W-val b-val lr)
  (let* ((W-var (make-var W-val #t))
         (b-var (make-var b-val #t))
         (X-var (make-var X-morph #f))
         (y-var (make-var y-morph #f))

         ;; Forward: pred = X @ W^T + b
         ;; var-transpose W-var: [1, 2] -> [2, 1]
         ;; var-matmul [10, 2] x [2, 1] -> [10, 1]
         ;; var+ [10, 1] + [1, 1] -> [10, 1]  (broadcast)
         (W-T   (var-transpose W-var '(1 0)))
         (XW    (var-matmul X-var W-T))
         (pred  (var+ XW b-var))

         ;; MSE loss: mean((pred - y)^2), scalar shape #()
         (res   (var- pred y-var))
         (loss  (var-mean (var* res res)))

         ;; Backward pass
         (_ (backward! loss))

         ;; Materialise gradients; dW shape [1,2], db shape [1,1]
         (dW (realize (var-grad W-var)))
         (db (realize (var-grad b-var)))

         ;; SGD update: param <- param - lr * grad
         ;; lr-morph shape [1] broadcasts to [1,2] and [1,1]
         (lr-m  (morph-from-list (list lr) #(1) 'f64))
         (W-new (realize (morph- W-val (morph* lr-m dW))))
         (b-new (realize (morph- b-val (morph* lr-m db))))

         ;; Scalar loss value for printing
         (loss-val (morph->list (realize (var-value loss)))))
    (list W-new b-new loss-val)))

;;; -----------------------------------------------------------------------
;;; Training loop
;;; -----------------------------------------------------------------------
(define W0 (morph-from-list '(0.0 0.0) #(1 2) 'f64))
(define b0 (morph-from-list '(0.0) #(1 1) 'f64))
(define lr 0.5)
(define steps 30)

(format #t "Training for ~a steps, lr = ~a~%~%" steps lr)
(let iter ((W W0) (b b0) (step 1))
  (when (<= step steps)
    (let* ((result    (train-step W b lr))
           (W-new     (car result))
           (b-new     (cadr result))
           (loss-val  (caddr result)))
      (when (member step '(1 5 10 20 30))
        (format #t "  step ~a: loss = ~a~%" step loss-val))
      (iter W-new b-new (+ step 1)))))

;;; -----------------------------------------------------------------------
;;; Final parameter check after training
;;; -----------------------------------------------------------------------
(define final-result
  (let iter ((W W0) (b b0) (step 1))
    (if (> step steps)
        (list W b)
        (let* ((result (train-step W b lr)))
          (iter (car result) (cadr result) (+ step 1))))))

(define W-final (car final-result))
(define b-final (cadr final-result))

(format #t "~%Recovered parameters after ~a steps:~%" steps)
(format #t "  W = ~a  (true: ((2.0 -1.0)))~%"
        (morph->list (realize W-final)))
(format #t "  b = ~a  (true: ((0.5)))~%"
        (morph->list (realize b-final)))
