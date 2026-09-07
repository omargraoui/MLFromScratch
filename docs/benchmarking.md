# Benchmark methodology

The benchmark suite checks whether the NumPy implementations solve the intended mathematical
problems. scikit-learn supplies independently implemented estimators, datasets, and evaluation
metrics. The core package does not import it. Runtime is recorded as context; performance
superiority is neither a goal nor a claim.

## Reproduce the report

From the repository root, after `python -m pip install -e ".[dev]"`:

```bash
python -m benchmarks.run_all
python scripts/generate_figures.py
```

To refresh the committed README table from the newly executed comparisons, run
`python -m benchmarks.run_all --update-readme`. This replaces only the section between the
README's benchmark summary markers and fails explicitly if those markers are missing.

Use `--output-dir PATH` on either command to write elsewhere. The benchmark writes:

- [`results.csv`](../benchmarks/results/results.csv): one row per numerical comparison.
- [`results.json`](../benchmarks/results/results.json): the same rows plus environment,
  preprocessing, hyperparameters, training sizes, timings, and convergence information.
- [`results.md`](../benchmarks/results/results.md): the complete human-readable table.
- [`summary.md`](../benchmarks/results/summary.md): selected metrics for the README.

All numbers come from fitting the estimators when the command runs. Nothing reads prior results
to fabricate a new report. A non-finite metric fails the run. Re-execution changes timestamps
and timings; numerical results should agree to ordinary floating-point tolerances in a matching
environment. Published dependency versions are captured in JSON rather than asserted to work
identically across all future releases.

## Six datasets, appropriate tasks

| Dataset | Source and size | Evaluated algorithms | Preparation |
| :-- | :-- | :-- | :-- |
| Synthetic regression | `make_regression`: 600 × 8, 6 informative features, noise SD 10, bias 12 | Linear, GD and normal solvers | Training feature z-scores |
| Diabetes | `load_diabetes`: 442 × 10 | Linear, GD and normal solvers | Training feature z-scores |
| Breast cancer | `load_breast_cancer`: 569 × 30 | Binary logistic | Stratified split, training feature z-scores |
| Iris | `load_iris`: 150 × 4 | K-Means, PCA (2 components) | Whole-data feature z-scores |
| Wine | `load_wine`: 178 × 13 | K-Means, PCA (5 components) | Whole-data feature z-scores |
| Digits | `load_digits`: 1,797 × 64 | PCA (20 components) | Divide pixel intensity by 16 |

All six datasets are generated locally or bundled with scikit-learn. No download, credentials,
or external service is needed. The supervised split uses 75% training data and 25% test data
with `random_state=42`. Classification is stratified. Means and population standard deviations
are calculated with NumPy **after splitting**, using only training samples; a constant feature
gets scale 1. Test samples use the frozen training statistics. Targets are not standardized.

Iris/Wine clustering and PCA are exploratory, in-sample diagnostics. Their preprocessing uses
the full feature matrix and makes no claim about held-out generalization. Known class labels
are used only to evaluate or color plots; they never enter an unsupervised fit. The fixed
digits division by 16 represents the dataset's pixel range and estimates no parameters.

## Matching objectives and interpreting measurements

### Linear regression and gradient descent

Both implementations minimize unregularized MSE with an intercept. The `gd` solver is compared
with scikit-learn's least-squares estimator, so agreement demonstrates optimizer convergence to
the least-squares solution. The `normal` solver provides a second algebraic check through
NumPy least squares. MSE and R² are computed on the held-out test set. Prediction RMSE against
the reference measures implementation agreement independently of dataset prediction difficulty.

The GD learning rate is `0.9 / L`, where `L = 2 ||[X, 1]||₂² / n` bounds the MSE Hessian's
largest eigenvalue. This ensures a conservative fixed step on the scaled training design.
The limit is 100,000 updates with `tol=1e-7`. The numerical learning rate, actual iteration
count, final loss, and convergence flag are recorded for each run. Ill-conditioned features
can still slow fixed-step GD substantially, particularly on diabetes.

### Logistic regression

The shared objective is

$$
\frac{1}{n}\sum_i[\log(1+\exp(z_i))-y_i z_i]
+\frac{\lambda}{2}\|w\|_2^2,\qquad z_i=x_i^T w+b,\quad \lambda=0.01.
$$

The intercept is not penalized. The benchmark sets scikit-learn's
`C = 1 / (n_train * lambda)` to match its regularization convention, rather than comparing
models with mismatched defaults. Our solver uses fixed-step GD and scikit-learn uses L-BFGS.
The step is `0.9 / (0.25 ||[X, 1]||₂²/n + lambda)`; maximum updates are 50,000, with
`tol=1e-8` for GD and `tol=1e-10` for L-BFGS. Different optimizers' tolerances and iteration
counts are not interchangeable.

Accuracy and F1 use threshold 0.5; log loss evaluates probabilities, and maximum absolute
probability disagreement makes small optimization differences visible. Class 1 in the bundled
breast cancer dataset is **benign**. F1 follows that encoding; it is not a clinical sensitivity
claim. This experiment is an algorithm validation exercise, not a medical model evaluation.

### K-Means

Both models use 3 clusters, K-Means++ initialization, 10 restarts, a 300-iteration cap,
`tol=1e-6`, and seed 42. This matches the broad algorithmic configuration, but not the sequence
of random draws: our implementation uses NumPy's `Generator`, while scikit-learn's random
state and greedy K-Means++ initialization differ. Our centroid displacement tolerance and
scikit-learn's variance-scaled tolerance also differ. Different local minima and iteration
counts are legitimate outcomes; they remain visible in the report.

Inertia measures the within-cluster sum of squared distances. Adjusted Rand index (ARI) is
computed both against the known labels and between implementations. Raw labels cannot be
compared: relabeling clusters from `[0, 1, 2]` to `[2, 0, 1]` leaves the same partition.
ARI equals 1 for identical partitions regardless of IDs. A lower inertia need not produce
better agreement with species or wine classes: K-Means optimizes geometry, not those labels.

### PCA

Both models center data, use full SVD, report sample variance with denominator `n - 1`, and
do not whiten. The report includes total retained explained variance ratio, mean squared
reconstruction error, and `||P_ours - P_reference||_F`, where `P = components.T @ components`.
The projector comparison evaluates the retained feature subspace.

A singular vector and its negative represent the same direction; raw signed component
differences are therefore misleading. Projectors and reconstruction avoid this ambiguity.
If equal singular values straddle the selected component cutoff, the chosen subspace itself
can be nonunique, so a projector difference needs interpretation rather than automatic blame.

## Reproducibility and limitations

- The seed is 42 wherever random sampling or initialization is involved.
- JSON records Python, NumPy, SciPy, scikit-learn, threadpoolctl, platform, and CPU information.
- Fits run under `threadpool_limits(limits=1)` to reduce variability from native thread pools.
- `perf_counter` measures one fit per estimator, excluding loading, preprocessing, and metrics.
  The least-squares reference is fitted once per dataset and reused for the two linear solvers.
- These are single measurements, not statistically rigorous speed benchmarks. There are no
  warm-up/repetition confidence intervals, and OS scheduling and BLAS versions affect timings.
- Runtime parity with optimized, compiled production estimators is not expected. Iteration
  counts across distinct optimizers and convergence criteria are not directly comparable.
- Relative difference is absolute difference divided by the absolute reference. A zero
  reference produces JSON `null`, a blank CSV cell, and `—` in Markdown.
- These small datasets exercise correctness across different geometries; they do not establish
  broad real-world predictive performance. There is no hyperparameter search or test-set tuning.
- Fixed-step GD, dense arrays, binary-only logistic regression, and full SVD intentionally
  limit scope. Large-scale sparse, stochastic, and distributed learning are outside this suite.

Figures and notebooks import the core estimators. The logistic figure deliberately uses only
radius and texture to show a true two-dimensional decision boundary; its accuracy is not the
30-feature benchmark result. All other plot annotations are computed from fitted models too.
