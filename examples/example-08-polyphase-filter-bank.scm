;;; example-08-polyphase-filter-bank.scm
;;; Demonstrates Advantage 8: Extensible Affine Operations
;;; Domain: M=4 polyphase decomposition of an FIR filter-bank input
;;;
;;; A polyphase decomposition of a length-N signal into M branches
;;; reads every M-th sample from offset m:
;;;   branch_m[k] = x[M*k + m]
;;;
;;; This is an affine map (stride M, offset m) into the flat buffer.
;;; In SRFI-231 each branch requires a hand-coded specialized-array-share
;;; call with manual stride arithmetic; there is no compositional
;;; mechanism for adding further decimation.
;;;
;;; Here morph-slice with the step parameter expresses each branch
;;; directly.  Composing two such slices (further stride-2 decimation)
;;; produces another affine morphism via the index algebra -- no manual
;;; stride arithmetic, no allocation.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-08-polyphase-filter-bank.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota map))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-context)

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) 0)))

(format #t "=== Example 8: Polyphase Filter Bank (M=4) ===~%~%")

;;; Signal x[n] = n,  n = 0..15
(define x (morph-from-list (map exact->inexact (iota 16)) #(16) 'f64))

;;; Four polyphase branches: branch m selects x[m::4].
;;; Each uses morph-slice(x, start=m, end=16, step=4).
;;;   branch-0: x[0], x[4], x[8],  x[12]
;;;   branch-1: x[1], x[5], x[9],  x[13]
;;;   branch-2: x[2], x[6], x[10], x[14]
;;;   branch-3: x[3], x[7], x[11], x[15]
(define branch-0 (morph-slice x '(0) '(16) 4))
(define branch-1 (morph-slice x '(1) '(16) 4))
(define branch-2 (morph-slice x '(2) '(16) 4))
(define branch-3 (morph-slice x '(3) '(16) 4))

;;; Further stride-2 decimation of branch-0:
;;; selects positions 0 and 2 within branch-0, i.e. x[0] and x[8].
;;; This is a second affine composition: no new allocation.
(define decimated-0 (morph-slice branch-0 '(0) '(4) 2))

;;; All polyphase branches + nested decimation: 0 allocations
(define ctx (make-morphism-context))
(for-each (lambda (b) (realize/ctx ctx b))
          (list branch-0 branch-1 branch-2 branch-3 decimated-0))

(format #t "Signal shape: ~a~%~%" (get-morphism-shape x))
(format #t "Polyphase branches (M=4, all shape [4]):~%")
(format #t "  branch-0 (x[0::4]):  ~a~%" (morph->list (realize branch-0)))
(format #t "  branch-1 (x[1::4]):  ~a~%" (morph->list (realize branch-1)))
(format #t "  branch-2 (x[2::4]):  ~a~%" (morph->list (realize branch-2)))
(format #t "  branch-3 (x[3::4]):  ~a~%" (morph->list (realize branch-3)))
(format #t "~%Stride-2 decimation of branch-0 (x[0::8]): ~a~%"
        (morph->list (realize decimated-0)))
(format #t "~%Allocations for all branches + nested decimation: ~a  (expected 0)~%"
        (stats-ref (context-stats ctx) 'allocations))

;;; Apply a 4-tap averaging FIR to branch-0:
;;;   output = sum(branch-0[k] * h[k]) = inner product
;;; morph* is the computational step; morph-reduce folds to a scalar.
(define h (morph-from-list '(0.25 0.25 0.25 0.25) #(4) 'f64))
(define filtered (morph-reduce 'sum (morph* branch-0 h) '()))

(define ctx2 (make-morphism-context))
(realize/ctx ctx2 filtered)

(format #t "~%4-tap average of branch-0:~%")
(format #t "  Allocations (morph* + morph-reduce): ~a  (expected 2)~%"
        (stats-ref (context-stats ctx2) 'allocations))
(define filt-val (morph->list (realize filtered)))
(format #t "  Result: ~a  (expected ~a)~%"
        filt-val
        (* 0.25 (apply + (morph->list (realize branch-0)))))
