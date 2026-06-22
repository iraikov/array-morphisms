;;; test-context.scm
;;;
;;; Phase 6: Memory Reuse Integration Tests
;;;
;;; Run with:
;;;   /home/igr/bin/chicken/bin/csi -s tests/test-context.scm

(import scheme chicken.base)
(import test)
(import (only srfi-1 every iota))
(import array-morphisms-core)
(import array-morphisms-index-fn)
(import array-morphisms-basic-ops)
(import array-morphisms-structural-ops)
(import array-morphisms-realization)
(import array-morphisms-context)

;;; Helpers

(define (approx= a b #!optional (tol 1e-10))
  (< (abs (- a b)) tol))

(define (lists-approx= xs ys #!optional (tol 1e-10))
  (and (= (length xs) (length ys))
       (every (lambda (a b) (approx= a b tol)) xs ys)))

(define (morph->flat m)
  (let ((lst (morph->list m)))
    (if (list? lst) (flatten-nested-list lst) (list lst))))

(define (flatten-nested-list lst)
  (cond ((null? lst) '())
        ((pair? lst) (append (flatten-nested-list (car lst))
                             (flatten-nested-list (cdr lst))))
        (else (list lst))))

(define (stats-ref stats key)
  (let ((p (assq key stats)))
    (if p (cdr p) #f)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 1: Trace recording
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "trace recording"

  (test-assert "fresh context is in trace mode"
    (let ((ctx (make-morphism-context)))
      (eq? (context-mode ctx) 'trace)))

  (test-assert "single binary op records one allocation"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b   (morph-from-list '(4.0 5.0 6.0) #(3) 'f64))
           (c   (morph+ a b)))
      (realize/ctx ctx c)
      (= (stats-ref (context-stats ctx) 'allocations) 1)))

  (test-assert "chain of two ops records two allocations"
    ;; Use a concrete second operand to avoid double-realizing the abstract b.
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (two (morph-from-list '(2.0 2.0) #(2) 'f64))
           (b   (morph+ a a))      ; alloc 0
           (c   (morph* b two)))   ; alloc 1
      (realize/ctx ctx c)
      (= (stats-ref (context-stats ctx) 'allocations) 2)))

  (test-assert "concrete input arrays do not generate allocations"
    (let* ((ctx (make-morphism-context))
           ;; 'a' is already concrete; realize/ctx on it should record nothing
           (a   (morph-from-list '(1.0 2.0 3.0) #(3) 'f64)))
      (realize/ctx ctx a)
      (= (stats-ref (context-stats ctx) 'allocations) 0))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 2: Zero-copy transparency
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "zero-copy transparency"

  (test-assert "reshape of concrete array records no allocation"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0 4.0 5.0 6.0) #(6) 'f64))
           (b   (morph-reshape a #(2 3))))   ; zero-copy
      (realize/ctx ctx b)
      (= (stats-ref (context-stats ctx) 'allocations) 0)))

  (test-assert "transpose of concrete array records no allocation"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (b   (morph-transpose a)))
      (realize/ctx ctx b)
      (= (stats-ref (context-stats ctx) 'allocations) 0)))

  (test-assert "reshape then arithmetic: only arithmetic records allocation"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (b   (morph-reshape a #(2 2)))    ; zero-copy
           (c   (morph+ b b)))              ; alloc 0
      (realize/ctx ctx c)
      (= (stats-ref (context-stats ctx) 'allocations) 1))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 3: Finalization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "finalization"

  (test-assert "finalize switches mode to replay"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (eq? (context-mode ctx) 'replay)))

  (test-assert "finalize creates buffer pool (buffers count >= 1)"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (>= (stats-ref (context-stats ctx) 'buffers) 1)))

  (test-assert "finalize on already-finalized context signals error"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0) #(1) 'f64))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (condition-case
          (begin (finalize-context! ctx) #f)
        (e (exn) #t)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 4: Replay correctness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "replay correctness"

  (test-assert "replay of simple addition gives same values"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b   (morph-from-list '(10.0 20.0 30.0) #(3) 'f64))
           (c   (morph+ a b))
           (ref (morph->flat (realize c))))
      (realize/ctx ctx c)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx c))))
        (lists-approx= result ref))))

  (test-assert "replay of chained ops gives same values"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (b   (morph+ a a))
           (c   (morph* b (morph-from-list '(0.5 0.5 0.5 0.5) #(4) 'f64)))
           (ref (morph->flat (realize c))))
      (realize/ctx ctx c)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx c))))
        (lists-approx= result ref))))

  (test-assert "replay with zero-copy reshape gives same values"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (b   (morph-reshape a #(2 2)))
           (c   (morph+ b b))
           (ref (morph->flat (realize c))))
      (realize/ctx ctx c)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx c))))
        (lists-approx= result ref))))

  (test-assert "replay of unary op gives same values"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 4.0 9.0) #(3) 'f64))
           (b   (morph-sqrt a))
           (ref (morph->flat (realize b))))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx b))))
        (lists-approx= result ref 1e-9)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 5: Memory reduction
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "memory reduction"

  (test-assert "buffer count <= allocation count after finalize"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (b   (morph+ a a))           ; alloc 0 -- dies at step 1
           (c   (morph* b b))           ; alloc 1 -- dies at step 2
           (d   (morph+ c a)))          ; alloc 2 -- result
      (realize/ctx ctx d)
      (finalize-context! ctx)
      (let ((stats (context-stats ctx)))
        (<= (stats-ref stats 'buffers)
            (stats-ref stats 'allocations)))))

  (test-assert "linear chain of 4 ops uses at most 2 buffers"
    ;; b -> c -> d -> e, each step uses concrete 'two' as second operand
    ;; so no shared abstract subexpressions.  Each intermediate is used
    ;; exactly once, allowing alternating buffer reuse: buf0, buf1, buf0, buf1.
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (two (morph-from-list '(2.0 2.0) #(2) 'f64))
           (b   (morph+ a   two))   ; alloc 0
           (c   (morph+ b   two))   ; alloc 1
           (d   (morph+ c   two))   ; alloc 2  (can reuse alloc-0 buf)
           (e   (morph+ d   two)))  ; alloc 3  (can reuse alloc-1 buf)
      (realize/ctx ctx e)
      (finalize-context! ctx)
      (<= (stats-ref (context-stats ctx) 'buffers) 2))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 6: Multi-inference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "multi-inference"

  (test-assert "three consecutive replay runs give identical results"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0) #(3) 'f64))
           (b   (morph+ a a))
           (ref (morph->flat (realize b))))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (let loop ((i 0))
        (if (= i 3)
            #t
            (begin
              (reset-context! ctx)
              (let ((r (morph->flat (realize/ctx ctx b))))
                (and (lists-approx= r ref)
                     (loop (+ i 1)))))))))

  (test-assert "reset-context! before finalize signals error"
    (let* ((ctx (make-morphism-context)))
      (condition-case
          (begin (reset-context! ctx) #f)
        (e (exn) #t))))

  (test-assert "multi-op graph replays stably across 5 runs"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0 4.0) #(4) 'f64))
           (b   (morph* a a))
           (c   (morph+ b (morph-from-list '(1.0 1.0 1.0 1.0) #(4) 'f64)))
           (ref (morph->flat (realize c))))
      (realize/ctx ctx c)
      (finalize-context! ctx)
      (let loop ((i 0))
        (if (= i 5)
            #t
            (begin
              (reset-context! ctx)
              (let ((r (morph->flat (realize/ctx ctx c))))
                (and (lists-approx= r ref)
                     (loop (+ i 1))))))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 7: Reduction morphisms
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "reduction morphisms"

  (test-assert "reduction records one allocation in trace"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (r   (morph-reduce 'sum a '(1))))
      (realize/ctx ctx r)
      (= (stats-ref (context-stats ctx) 'allocations) 1)))

  (test-assert "reduction replay gives correct result"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '((1.0 2.0) (3.0 4.0)) #(2 2) 'f64))
           (r   (morph-reduce 'sum a '(1)))
           (ref (morph->flat (realize r))))
      (realize/ctx ctx r)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx r))))
        (lists-approx= result ref))))

  (test-assert "mean reduction replay gives correct result"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(2.0 4.0 6.0 8.0) #(4) 'f64))
           (r   (morph-reduce 'mean a '(0)))
           (ref (morph->flat (realize r))))
      (realize/ctx ctx r)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx r))))
        (lists-approx= result ref))))

  (test-assert "reduction followed by arithmetic replays correctly"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '((1.0 3.0) (5.0 7.0)) #(2 2) 'f64))
           (s   (morph-reduce 'sum a '(0)))   ; alloc 0
           (two (morph-from-list '(2.0 2.0) #(2) 'f64))
           (b   (morph* s two))               ; alloc 1 (avoid sharing s)
           (ref (morph->flat (realize b))))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx b))))
        (lists-approx= result ref)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 8: Mixed dtypes
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "mixed dtypes"

  (test-assert "f32 op traces and replays correctly"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0) #(3) 'f32))
           (b   (morph+ a a))
           (ref (morph->flat (realize b))))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (reset-context! ctx)
      (let ((result (morph->flat (realize/ctx ctx b))))
        (lists-approx= result ref 1e-5)))
    )

  (test-assert "separate f32 and f64 contexts do not interfere"
    (let* ((ctx32 (make-morphism-context))
           (ctx64 (make-morphism-context))
           (a32   (morph-from-list '(1.0 2.0) #(2) 'f32))
           (a64   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b32   (morph+ a32 a32))
           (b64   (morph+ a64 a64)))
      (realize/ctx ctx32 b32)
      (realize/ctx ctx64 b64)
      (finalize-context! ctx32)
      (finalize-context! ctx64)
      (reset-context! ctx32)
      (reset-context! ctx64)
      (let ((r32 (morph->flat (realize/ctx ctx32 b32)))
            (r64 (morph->flat (realize/ctx ctx64 b64))))
        (and (lists-approx= r32 '(2.0 4.0) 1e-5)
             (lists-approx= r64 '(2.0 4.0) 1e-10)))))

  (test-assert "f32 buffer dtype preserved after finalize"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0 3.0) #(3) 'f32))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      ;; After replay, result should have f32 dtype
      (reset-context! ctx)
      (let ((result (realize/ctx ctx b)))
        (eq? (get-morphism-dtype result) 'f32)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Group 9: Context pinning API
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(test-group "context pinning API"

  (test-assert "context-counter starts at 0"
    (let ((ctx (make-morphism-context)))
      (= (context-counter ctx) 0)))

  (test-assert "context-counter increments after non-zero-copy realize/ctx"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (= (context-counter ctx) 1)))

  (test-assert "context-counter increments once per op in a chain"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (two (morph-from-list '(2.0 2.0) #(2) 'f64))
           (b   (morph+ a a))
           (c   (morph* b two)))
      (realize/ctx ctx c)
      (= (context-counter ctx) 2)))

  (test-assert "context-counter unchanged for concrete input (zero-copy)"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64)))
      (realize/ctx ctx a)
      (= (context-counter ctx) 0)))

  (test-assert "context-pin-output! succeeds in trace mode"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (context-pin-output! ctx 0)
      #t))

  (test-assert "context-pin-output! signals error for unknown alloc-id"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (condition-case
          (begin (context-pin-output! ctx 99) #f)
        (e (exn) #t))))

  (test-assert "context-pin-output! signals error in replay mode"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (b   (morph+ a a)))
      (realize/ctx ctx b)
      (finalize-context! ctx)
      (condition-case
          (begin (context-pin-output! ctx 0) #f)
        (e (exn) #t))))

  (test-assert "pinned alloc gets its own buffer slot (not reused by independent later alloc)"
    ;; Two independent allocs of same dtype/size:
    ;;   alloc 0 (b = a+a): inputs = (external, external) -> last-use = 0 without pinning
    ;;   alloc 1 (c = x+x): inputs = (external, external) -> last-use = 1 without pinning
    ;; Without pinning: alloc 0 dies at step 0, alloc 1 born at step 1 -> slot reused -> 1 buffer.
    ;; With pinning: alloc 0's last-use extended to n-1=1 -> overlaps alloc 1 -> 2 buffers.
    ;; Use concrete arrays as inputs so realize/ctx produces exactly one alloc per call.
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64))
           (x   (morph-from-list '(3.0 4.0) #(2) 'f64)))
      (realize/ctx ctx (morph+ a a))    ; alloc 0, independent (external inputs only)
      (realize/ctx ctx (morph+ x x))    ; alloc 1, independent (external inputs only)
      (context-pin-output! ctx 0)
      (finalize-context! ctx)
      (= (stats-ref (context-stats ctx) 'buffers) 2)))

  (test-assert "print-context-plan runs without error after pinning"
    (let* ((ctx (make-morphism-context))
           (a   (morph-from-list '(1.0 2.0) #(2) 'f64)))
      (realize/ctx ctx (morph+ a a))
      (context-pin-output! ctx 0)
      (finalize-context! ctx)
      (print-context-plan ctx)
      #t)))

(test-exit)
