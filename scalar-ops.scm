;;; scalar-ops.scm
;;;
;;; Scheme side of the scalar-op table (scalar-ops-table.scm): for each op,
;;; its parameters, its symbolic expression and the combiner procedure that
;;; the morphism constructors apply to array elements.  The combiners are
;;; generated from the table at syntax time, so the expression an op is
;;; known by and the procedure that computes it come from the same entry.
;;;
;;;   (scalar-op-combiner 'relu)   => a procedure of one argument
;;;   (scalar-op-expr 'relu)       => (if (> x 0.0) x 0.0)
;;;   (scalar-op-params 'relu)     => (x)

(module array-morphisms-scalar-ops

  (scalar-op?
   scalar-op-names
   scalar-op-arity
   scalar-op-params
   scalar-op-expr
   scalar-op-combiner
   scalar-op-generic?
   scalar-add
   scalar-primitive?
   scalar-primitives)

  (import scheme (chicken base)
          (only (chicken flonum) fp+ fpabs)
          (only srfi-69 make-hash-table hash-table-ref/default hash-table-set!))
  (import-for-syntax scheme (chicken base) array-morphisms-scalar-ops-table)

  (define (scalar-add x y)
    "The sum of x and y, the combiner of the add op.  Two flonums are added
    with fp+, which follows IEEE 754 and so gives -0.0 for -0.0 + -0.0, as
    the compiled element-wise kernels do.  The generic + of some CHICKEN
    versions starts its sum from exact 0 and gives 0.0 there.  Other
    arguments, such as integers, are added with the generic +."
    (if (and (flonum? x) (flonum? y))
        (fp+ x y)
        (+ x y)))

  ;; (define-scalar-ops name) binds name to a list with one vector per table
  ;; entry: #(op params expr combiner generic?).  The combiner is built as
  ;; the entry's props say (see scalar-ops-table.scm).  The table's symbols
  ;; are inserted unrenamed, so the expressions refer to the bindings of
  ;; this module: the standard numeric procedures and fpabs.
  (define-syntax define-scalar-ops
    (er-macro-transformer
     (lambda (form r c)
       (define (generic-name props)
         (let loop ((ps props))
           (cond ((null? ps) #f)
                 ((and (pair? (car ps)) (eq? (caar ps) 'generic)) (cadar ps))
                 (else (loop (cdr ps))))))
       (define (combiner-form params expr props)
         (cond ((generic-name props) => (lambda (name) name))
               ((memq 'exact-args props)
                `(,(r 'lambda) ,params ,expr))
               (else
                `(,(r 'lambda) ,params
                  (,(r 'let) ,(map (lambda (p) `(,p (,(r 'exact->inexact) ,p))) params)
                   ,expr)))))
       (define (entry-form e)
         (unless (and (list? e) (>= (length e) 3) (symbol? (car e))
                      (list? (cadr e)) (pair? (cadr e)))
           (error 'define-scalar-ops "malformed scalar-op table entry" e))
         (let ((op (car e)) (params (cadr e)) (expr (caddr e)) (props (cdddr e)))
           `(,(r 'vector) (,(r 'quote) ,op)
                          (,(r 'quote) ,params)
                          (,(r 'quote) ,expr)
                          ,(combiner-form params expr props)
                          ,(and (generic-name props) #t))))
       `(,(r 'define) ,(cadr form)
         (,(r 'list) ,@(map entry-form scalar-ops-table))))))

  (define-scalar-ops %scalar-op-entries)

  (define %scalar-op-index
    (let ((table (make-hash-table eq?)))
      (for-each (lambda (e) (hash-table-set! table (vector-ref e 0) e))
                %scalar-op-entries)
      table))

  (define (entry who op)
    (or (hash-table-ref/default %scalar-op-index op #f)
        (error who "unknown scalar op" op)))

  (define (scalar-op? op)
    "True when op (a symbol) names an entry of the scalar-op table."
    (and (symbol? op) (hash-table-ref/default %scalar-op-index op #f) #t))

  (define (scalar-op-names)
    "List of the op names of the scalar-op table, in table order."
    (map (lambda (e) (vector-ref e 0)) %scalar-op-entries))

  (define (scalar-op-arity op)
    "Number of arguments op takes."
    (length (vector-ref (entry 'scalar-op-arity op) 1)))

  (define (scalar-op-params op)
    "The parameter names of op's expression, a list of symbols."
    (vector-ref (entry 'scalar-op-params op) 1))

  (define (scalar-op-expr op)
    "The expression that computes op from its parameters, as a datum."
    (vector-ref (entry 'scalar-op-expr op) 2))

  (define (scalar-op-combiner op)
    "The Scheme procedure that computes op on its arguments.  Every call
    returns the same procedure, so a combiner can be recognized with eq?."
    (vector-ref (entry 'scalar-op-combiner op) 3))

  (define (scalar-op-generic? op)
    "True when op's combiner is a standard procedure that also accepts
    exact numbers (the (generic name) prop of the table)."
    (vector-ref (entry 'scalar-op-generic? op) 4))

  ;; The primitive operators of table expressions and segop bodies.  if and
  ;; let* are special forms and are not listed.
  (define %scalar-primitives
    '(+ - * / exp log sqrt sin cos tan floor ceiling expt fpabs < > <= >= =))

  (define (scalar-primitives)
    "List of the primitive operator names allowed in scalar expressions."
    %scalar-primitives)

  (define (scalar-primitive? sym)
    "True when sym is a primitive operator of scalar expressions."
    (and (memq sym %scalar-primitives) #t))

) ;; end module array-morphisms-scalar-ops
