;;; example-01-affine-composition.scm
;;; Demonstrates Index Functions as First-Class Objects
;;; Domain: Polyphase downsampling of a 1-D signal
;;;
;;; When morph-slice is applied twice (stride 2, then stride 2 again),
;;; the system builds two affine index maps.  Both are evaluated on
;;; demand through a single index-function composition; no intermediate
;;; buffer is allocated.  The context allocation counter makes this
;;; observable: structural (affine-only) chains record 0 allocations
;;; while arithmetic operations record 1 per computation.
;;;
;;; In SRFI-231, the indexer of a specialized array is an opaque
;;; procedure.  One can observe its effects but cannot compose two
;;; indexers algebraically.  Here the affine maps are inspectable
;;; algebraic data; the composition law is programmable.
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s examples/example-01-affine-composition.scm

(import scheme (chicken base) (chicken format))
(import (only srfi-1 iota))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-context)

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) 0)))

(format #t "=== Example 1: Affine Index Composition ===~%~%")

;;; Signal x[n] = n,  n = 0 .. 15
(define x (morph-from-list (map exact->inexact (iota 16)) #(16) 'f64))

;;; Stride-2 downsample: select every other sample
;;; Affine map i -> 2*i,  output shape [8]
(define x-even (morph-slice x '(0) '(16) 2))

;;; Stride-2 again on the even-sample view
;;; Composed affine map i -> 4*i,  output shape [4]
(define x-quarter (morph-slice x-even '(0) '(8) 2))

;;; Both slices are structural (affine) -- 0 buffer allocations
(define ctx-struct (make-morphism-context))
(realize/ctx ctx-struct x-quarter)

(format #t "Input shape:               ~a~%" (get-morphism-shape x))
(format #t "After stride-2 slice:      ~a~%" (get-morphism-shape x-even))
(format #t "After composed stride-4:   ~a~%" (get-morphism-shape x-quarter))
(format #t "~%Allocations for structural chain: ~a  (expected 0)~%"
        (stats-ref (context-stats ctx-struct) 'allocations))
(format #t "Values x[0::4]:  ~a~%"
        (morph->list (realize x-quarter)))

;;; Now add arithmetic: scale each of the 4 downsampled samples by 0.5.
;;; The multiply is a computational morphism layered on top of the affine chain.
;;; The affine index function feeds the arithmetic without materialising
;;; an intermediate buffer for the strided view.
(define scale (morph-from-list '(0.5 0.5 0.5 0.5) #(4) 'f64))
(define scaled (morph* x-quarter scale))

(define ctx-arith (make-morphism-context))
(realize/ctx ctx-arith scaled)

(format #t "~%After multiply by 0.5:~%")
(format #t "Allocations (structural chain + arithmetic): ~a  (expected 1)~%"
        (stats-ref (context-stats ctx-arith) 'allocations))
(format #t "Scaled values:   ~a~%"
        (morph->list (realize scaled)))
