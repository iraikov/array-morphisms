;;; tests/run.scm
;;; Run all test-*.scm suites in a single process and report a summary.
;;; Usage: csi -s tests/run.scm   (from the project root)
;;;        csi -s run.scm         (from within tests/)

(import scheme (chicken base) test)

;;; Override test-exit so included files do not terminate the process early.
;;; test-failure-count is a global parameter in the test egg that accumulates
;;; across all included files; we read it once at the end.
(define test-exit (lambda () (values)))

(include-relative "test-core.scm")
(include-relative "test-index-fn.scm")
(include-relative "test-basic-ops.scm")
(include-relative "test-batch-ops.scm")
(include-relative "test-blas.scm")
(include-relative "test-structural-ops.scm")
(include-relative "test-realization.scm")
(include-relative "test-context.scm")
(include-relative "test-morph-env.scm")
(include-relative "test-im2col.scm")
(include-relative "test-col2im.scm")
(include-relative "test-grad.scm")
(include-relative "test-grad-check.scm")
(include-relative "test-attention.scm")
(include-relative "test-ssa.scm")

(exit (min 255 (test-failure-count)))
