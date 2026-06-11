;;; example-07-batch-awareness.scm
;;; Demonstrates Advantage 7: Batch Awareness as a First-Class Attribute
;;; Domain: Batched linear layer forward pass
;;;
;;; morph-batch-matmul accepts a rank-3 activation tensor X of shape
;;; [N, M, K] and a shared weight matrix W of shape [K, P] and produces
;;; [N, M, P] without any explicit loop over the batch dimension N.  The
;;; batch axis is a first-class attribute of the morphism, not a
;;; control-flow loop the user must write.
;;;
;;; For comparison, the same result is verified element-by-element using
;;; individual morph-matmul calls via morph-slice + morph-reshape.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-07-batch-awareness.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-batch-ops)
(import array-morphisms-blas-exec)

(format #t "=== Example 7: Batch Awareness (Batched Linear Layer) ===~%~%")

;;; Weight matrix W: shape [3, 2]  (maps 3-D input to 2-D output)
;;;   W = [[1  0]
;;;        [0  1]
;;;        [1  1]]
(define W
  (morph-from-list '(1.0 0.0
                     0.0 1.0
                     1.0 1.0)
                   #(3 2) 'f64))

;;; Batch of N=2 input row-vectors, each treated as a [1, 3] matrix.
;;; Shape [N=2, M=1, K=3]
;;;   X[0] = [[1  2  3]]
;;;   X[1] = [[4  5  6]]
(define X
  (morph-from-list '(1.0 2.0 3.0
                     4.0 5.0 6.0)
                   #(2 1 3) 'f64))

;;; Batched matmul: X[n] @ W for each n, no user-written loop needed.
;;; Expected result shapes: [2, 1, 2]
;;;   Y[0] = X[0] @ W = [[1*1+2*0+3*1, 1*0+2*1+3*1]] = [[4, 5]]
;;;   Y[1] = X[1] @ W = [[4*1+5*0+6*1, 4*0+5*1+6*1]] = [[10, 11]]
(define Y (morph-batch-matmul X W))

(format #t "X shape: ~a  (N=2 samples, each a [1x3] row vector)~%"
        (get-morphism-shape X))
(format #t "W shape: ~a  (3-D input -> 2-D output)~%"
        (get-morphism-shape W))
(format #t "Y shape: ~a  (N=2 output row-vectors, each [1x2])~%~%"
        (get-morphism-shape Y))

(define Y-vals (morph->list (realize Y)))
(format #t "Y[0] = ~a  (expected ((4.0 5.0)))~%"  (car Y-vals))
(format #t "Y[1] = ~a  (expected ((10.0 11.0)))~%" (cadr Y-vals))

;;; Verification: compute Y[0] manually via morph-slice + morph-reshape + morph-matmul.
;;; Extract X[0] as a concrete [1, 1, 3] view, then reshape to [1, 3].
(define X0-2d
  (morph-reshape (morph-slice X '(0 0 0) '(1 1 3)) #(1 3)))

;;; morph-matmul [1,3] x [3,2] -> [1,2]; morph->list gives ((4.0 5.0))
(define Y0-manual (morph->list (realize (morph-matmul X0-2d W))))
(format #t "~%Manual morph-matmul for sample 0: ~a~%" Y0-manual)
(format #t "Matches batch result Y[0]:         ~a~%"
        (equal? (car Y-vals) Y0-manual))
