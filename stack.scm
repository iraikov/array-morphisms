
;; Stack routines.  Based on dfa2.sc in the benchmarks code supplied
;; with Stalin 0.11

(define (make-stack)
  (box '()))

(define (stack-empty? s)
  (null? (unbox s)))

(define (stack-push! s obj)
  (set-box! s (cons obj (unbox s)))
  s)

(define (stack-pop! s)
  (let ((l (unbox s)))
    (set-box! s (cdr l))
    (car l)))

(define (stack-depth s)
  (let ((l (unbox s)))
    (- (length l) 1)))

(define (stack-peek s)
  (let ((l (unbox s)))
    (car l)))

(define (stack-ppeek s) 
  (let ((l (unbox s)))
    (values (car l)  (car (cdr l)))))

(define (stack-rest s)
  (let ((l (unbox s)))
    (cdr l)))

(define (list->stack lst)
  "Create a stack seeded with the elements of lst.
   The head of lst becomes the top of the stack."
  (box lst))

(define (stack->list s)
  "Return the stack contents as a list, top first. Non-destructive."
  (unbox s))
