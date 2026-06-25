;;; tests/test-morph-env.scm
;;; Unit tests for the array-morphisms-morph-env module.

(import scheme (chicken base) (chicken format))
(import test)
(import array-morphisms-morph-env)


;;; ==========================================================
;;; Helper
;;; ==========================================================

(define (run-env-run m)
  (call-with-values (lambda () (env-run m)) list))


;;; ==========================================================
;;; Layer 1: morph-env
;;; ==========================================================

(test-group "morph-env: empty env"
  (test "empty-morph-env is a morph-env?" #t (morph-env? empty-morph-env))
  (test "lookup in empty returns #f" #f (morph-env-lookup empty-morph-env 'x))
  (test "size of empty is 0" 0 (morph-env-size empty-morph-env))
  (test "alist of empty is '()" '() (morph-env->alist empty-morph-env)))

(test-group "morph-env: extend and lookup"
  (let ((e (morph-env-extend empty-morph-env 'x 42)))
    (test "lookup present key" 42 (morph-env-lookup e 'x))
    (test "lookup absent key"  #f (morph-env-lookup e 'y))
    (test "size after one extend" 1 (morph-env-size e)))
  (let* ((e1 (morph-env-extend empty-morph-env 'a 1))
         (e2 (morph-env-extend e1 'b 2))
         (e3 (morph-env-extend e2 'a 99)))
    (test "shadowing: newest wins" 99 (morph-env-lookup e3 'a))
    (test "other key unaffected"   2  (morph-env-lookup e3 'b))
    (test "size includes shadows"  3  (morph-env-size e3))))

(test-group "morph-env: remove"
  (let* ((e (morph-env-extend
              (morph-env-extend empty-morph-env 'x 1)
              'y 2))
         (e2 (morph-env-remove e 'x)))
    (test "removed key is absent" #f (morph-env-lookup e2 'x))
    (test "other key still present" 2 (morph-env-lookup e2 'y)))
  (let ((e (morph-env-remove empty-morph-env 'x)))
    (test "remove from empty is empty" 0 (morph-env-size e))))

(test-group "morph-env: fold and alist"
  (let* ((e  (morph-env-extend
               (morph-env-extend empty-morph-env 'a 10)
               'b 20))
         (al (morph-env->alist e)))
    (test "alist has both keys"
          #t
          (equal? al '((b . 20) (a . 10)))))
  (let* ((alist '((p . 1) (q . 2)))
         (e     (alist->morph-env alist)))
    (test "round-trip: p key present" 1 (morph-env-lookup e 'p))
    (test "round-trip: q key present" 2 (morph-env-lookup e 'q))))

(test-group "morph-env: merge"
  (let* ((base     (morph-env-extend empty-morph-env 'x 1))
         (override (morph-env-extend
                     (morph-env-extend empty-morph-env 'x 99)
                     'y 2))
         (merged   (morph-env-merge base override)))
    (test "override shadows base for common key" 99 (morph-env-lookup merged 'x))
    (test "override-only key present"             2  (morph-env-lookup merged 'y))
    (test "base-only key still present"          #f  (morph-env-lookup merged 'z))))


;;; ==========================================================
;;; morph-env-stack
;;; ==========================================================

(test-group "morph-env-stack"
  (let* ((e1 (morph-env-extend empty-morph-env 'a 1))
         (e2 (morph-env-extend empty-morph-env 'b 2))
         (s  (morph-env-stack-push e2
               (morph-env-stack-push e1 empty-morph-env-stack))))
    (test "top frame lookup" 2 (morph-env-stack-lookup 'b s))
    (test "bottom frame lookup" 1 (morph-env-stack-lookup 'a s))
    (test "absent key" #f (morph-env-stack-lookup 'z s))
    (let ((s2 (morph-env-stack-pop s)))
      (test "after pop, 'b absent" #f (morph-env-stack-lookup 'b s2))
      (test "after pop, 'a present" 1 (morph-env-stack-lookup 'a s2)))))


;;; ==========================================================
;;; Layer 2: env-builder
;;; ==========================================================

(test-group "env-builder: basic extend and lookup"
  (let ((b (make-env-builder)))
    (test "lookup in fresh builder" #f (env-builder-lookup b 'k))
    (env-builder-extend! b 'k 42)
    (test "lookup after extend!" 42 (env-builder-lookup b 'k))
    (env-builder-extend! b 'k 99)
    (test "second extend! shadows"  99 (env-builder-lookup b 'k))
    (test "env is a morph-env?" #t (morph-env? (env-builder-env b)))))

(test-group "env-builder: emit and items"
  (let ((b (make-env-builder)))
    (test "items initially empty" #t (null? (env-builder-items b)))
    (env-builder-emit! b 'first)
    (env-builder-emit! b 'second)
    (test "items in emission order" #t (equal? (env-builder-items b) '(first second)))))

(test-group "env-builder: snapshot"
  (let ((b (make-env-builder)))
    (env-builder-extend! b 'x 7)
    (env-builder-emit! b 'a)
    (env-builder-emit! b 'b)
    (call-with-values
      (lambda () (env-builder-snapshot b))
      (lambda (env items)
        (test "snapshot env has key" 7 (morph-env-lookup env 'x))
        (test "snapshot items in order" #t (equal? items '(a b)))))))

(test-group "env-builder: initialised with existing env"
  (let* ((base (morph-env-extend empty-morph-env 'base-key 'base-val))
         (b    (make-env-builder base)))
    (test "inherits from initial env" 'base-val (env-builder-lookup b 'base-key))))


;;; ==========================================================
;;; Layer 3: env-monad
;;; ==========================================================

(test-group "env-monad: env-return"
  (let ((result (run-env-run (env-return 42))))
    (test "result value" 42 (car result))
    (test "env is empty" #t (morph-env? (cadr result)))
    (test "no emitted items" '() (caddr result))))

(test-group "env-monad: env-extend-m + env-lookup-m"
  (let ((result
          (run-env-run
            (env-do
              (env-extend-m 'x 10)
              (v <- (env-lookup-m 'x))
              (env-return v)))))
    (test "bound value retrieved" 10 (car result))
    (test "env retains binding" 10 (morph-env-lookup (cadr result) 'x))))

(test-group "env-monad: env-emit-m"
  (let ((result
          (run-env-run
            (env-do
              (env-emit-m 'alpha)
              (env-emit-m 'beta)
              (env-return 'done)))))
    (test "emitted items in order" #t (equal? (caddr result) '(alpha beta)))
    (test "return value" 'done (car result))))

(test-group "env-monad: env-map-m"
  (let ((result
          (run-env-run
            (env-do
              (env-extend-m 'scale 3)
              (vs <- (env-map-m
                       (lambda (x)
                         (env-do
                           (s <- (env-lookup-m 'scale))
                           (let ((v (* x s)))
                             (env-do
                               (env-emit-m v)
                               (env-return v)))))
                       '(1 2 3)))
              (env-return vs)))))
    (test "mapped values" #t (equal? (car result) '(3 6 9)))
    (test "emitted during map" #t (equal? (caddr result) '(3 6 9)))))

(test-group "env-monad: env-run*"
  (call-with-values
    (lambda ()
      (env-run*
        (env-do
          (env-extend-m 'a 1)
          (env-emit-m 'item)
          (env-return 'ignored))))
    (lambda (env items)
      (test "env-run* env" 1 (morph-env-lookup env 'a))
      (test "env-run* items" #t (equal? items '(item))))))

(test-group "env-monad: shadowing across binds"
  (let ((result
          (run-env-run
            (env-do
              (env-extend-m 'v 1)
              (env-extend-m 'v 2)
              (x <- (env-lookup-m 'v))
              (env-return x)))))
    (test "newest binding wins" 2 (car result))))


(test-exit)
