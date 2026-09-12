# Mathematics behind the implementations

Each derivation below corresponds to code in [`src/ml_from_scratch`](../src/ml_from_scratch).
Samples are rows of $X\in\mathbb{R}^{n\times d}$; a single-target response is
$y\in\mathbb{R}^n$. All computations use real, dense float64 arrays.

## Gradient descent: following a local slope

An objective $J(\theta)$ assigns a scalar cost to parameters $\theta$.
Its gradient collects partial derivatives. The first-order approximation

$$
J(\theta+\Delta)\approx J(\theta)+\nabla J(\theta)^T\Delta
$$

shows why moving opposite the gradient decreases the objective for a sufficiently
small step. With learning rate $\alpha>0$:

$$
\boxed{\theta_{t+1}=\theta_t-\alpha\nabla J(\theta_t).}
$$

The [optimizer](../src/ml_from_scratch/optimization/gradient_descent.py) takes a callable
returning both the loss and gradient; the estimators supply their own objectives.
There is one fixed learning rate, with no hidden line search or learning-rate schedule.

For a differentiable convex objective with an $L$-Lipschitz gradient, a step
$0<\alpha\leq 1/L$ gives the usual descent guarantee. For a positive-definite quadratic,
$0<\alpha<2/L$ converges; values close to the upper bound can oscillate.
Large differences in feature scales increase curvature differences and slow convergence.
Standardization often helps, but must use training statistics when evaluating held-out data.

The stopping rule is $\|\nabla J(\theta_t)\|_2\leq\texttt{tol}$. This measures stationarity,
not an absolute guarantee of parameter accuracy. For a $\mu$-strongly convex objective,
the objective gap is bounded by $\|\nabla J\|_2^2/(2\mu)$; without such curvature information,
a small gradient alone does not establish closeness to a unique optimum.

`loss_history_` includes the initial objective and every accepted update, so its length
is `n_iter_ + 1`. Exhausting `max_iter` emits `ConvergenceWarning` and leaves
`converged_ = False`. A materially increasing loss or nonfinite computation raises
`FloatingPointError`; lower the learning rate or rescale features. Rounding-sized loss
increases are tolerated. The quadratic notebook checks convergence against an analytical optimum
and least-squares reference instead of treating a library estimator as a generic optimizer.

## Linear regression: least squares and its gradient

The hypothesis is an affine map:

$$
\hat y=Xw+b\mathbf{1},\qquad
J(w,b)=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2.
$$

Write residuals as $r=Xw+b\mathbf{1}-y$. Since $\partial r_i/\partial w_j=X_{ij}$,
the chain rule gives

$$
\frac{\partial J}{\partial w_j}=\frac{2}{n}\sum_i r_iX_{ij},\qquad
\boxed{\nabla_wJ=\frac{2}{n}X^Tr,\quad
\frac{\partial J}{\partial b}=\frac{2}{n}\mathbf{1}^Tr.}
$$

There is no hidden factor $1/2$ in this objective. With the augmented design
$A=[X\ \mathbf{1}]$, its Hessian is $2A^TA/n$; the largest eigenvalue provides
a defensible learning-rate bound. With `fit_intercept=False`, use $A=X$ and fix $b=0$.

The [implementation](../src/ml_from_scratch/linear_model/linear_regression.py) also offers
`solver="normal"`. Setting the gradient to zero gives the normal equations
$A^TA\theta=A^Ty$, but explicitly inverting $A^TA$ squares the condition number and
fails for rank-deficient inputs. The implementation instead uses `np.linalg.lstsq`.
With an intercept, it solves the centered problem and recovers
$b=\bar y-\bar Xw$. This gives the minimum-norm weight solution when the centered
design is rank deficient. The solver name describes the stationary least-squares
solution, not a matrix-inversion implementation.
[NumPy's least-squares contract](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
specifies the minimum-norm behavior.

### Preserving small variations around large feature offsets

With an intercept, translating features by $c$ preserves the fitted weights and
predictions: $w'=w$ and $b'=b-c^Tw$. Floating-point evaluation needs extra care.
For example, the float64 values $10^{16}$ and $10^{16}+2$ are distinct, but their
exact mean $10^{16}+1$ is not representable. Subtracting a rounded mean can change
the least-squares solution, even if another library gives the same result.

The direct solver retains a two-part feature mean. With training reference $a=x_0$,
compute $D=X-a$, $m=\operatorname{mean}(D)$, and solve using $X_c=D-m$. Predictions use

$$
\hat y(x)=((x-a)-m)^Tw+\bar y.
$$

Neither $a+m$ nor the cancellation between a large $x^Tw$ and intercept is needed
for prediction. The public intercept is still computed as
$b=(\bar y-m^Tw)-a^Tw$, but manually evaluating `X @ coef_ + intercept_` can lose
precision compared with `predict`. Training loss uses the same centered evaluation
as prediction. GD and fits without an intercept retain their original evaluation.

Analytic tests use known slopes and predictions, noisy repeated observations, and
a rank-deficient design with known minimum-norm weights. Absolute tolerances are
set at the scale of the small target variations, not the large feature offset.
This preserves information present in the input; it cannot recover differences
already rounded away. It also does not guarantee accuracy for every float64 input:
extreme feature ranges can overflow during translation, and target centering still
uses a float64 mean.

MSE averages squared errors; RMSE returns to target units; MAE averages absolute
errors. The coefficient of determination is

$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
$$

$R^2$ can be negative. It needs at least two samples here. Constant targets receive
1 for perfect predictions and 0 otherwise, an explicit finite convention.

## Binary logistic regression: probabilities and cross entropy

For labels $y_i\in\{0,1\}$, define logits and probabilities

$$
z_i=x_i^Tw+b,\qquad
p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}}.
$$

The Bernoulli likelihood is $p_i^{y_i}(1-p_i)^{1-y_i}$. Taking the negative logarithm
and averaging gives binary cross entropy. Optional L2 regularization yields

$$
J(w,b)=-\frac{1}{n}\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\frac{\lambda}{2}\|w\|_2^2.
$$

Since $\sigma'(z)=\sigma(z)(1-\sigma(z))$, differentiation cancels the probability
denominators and gives $\partial\ell_i/\partial z_i=p_i-y_i$. Therefore

$$
\boxed{\nabla_wJ=\frac{1}{n}X^T(p-y)+\lambda w,\quad
\frac{\partial J}{\partial b}=\frac{1}{n}\mathbf{1}^T(p-y).}
$$

The intercept is not penalized. Here `l2` is $\lambda$, not scikit-learn's inverse
regularization parameter `C`. For unweighted samples, matching the objectives requires
$C=1/(n\lambda)$ when $\lambda>0$.
See the [reference objective](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

The [implementation](../src/ml_from_scratch/linear_model/logistic_regression.py) avoids
computing logarithms of rounded probabilities during optimization. For binary labels,

$$
\ell_i=\log(1+e^{(1-2y_i)z_i})
=\operatorname{logaddexp}(0,(1-2y_i)z_i).
$$

This form also avoids subtracting two large, nearly equal values. The sigmoid computes
only $e^{-|z|}$, avoiding exponential overflow. The gradient needs the same care:
for $y=1$ and $z=40$, float64 evaluates $\sigma(z)-1$ as zero even though the
derivative is approximately $-4.248\times10^{-18}$ and is representable.
Differentiating the signed-logit loss directly gives

$$
s_i=1-2y_i,\qquad
\frac{\partial\ell_i}{\partial z_i}=s_i\sigma(s_i z_i)
=\begin{cases}
\sigma(z_i), & y_i=0,\\
-\sigma(-z_i), & y_i=1.
\end{cases}
$$

The implementation uses this residual in both the weight and intercept gradients;
the L2 term is unchanged. This avoids premature gradient-norm convergence caused by
rounding positive-class probabilities to one. It also preserves the mathematical
symmetry under swapping labels and negating parameters. Extremely small tails can
still underflow to zero when they are outside float64's representable range.
Tests compare individual derivatives to a high-precision decimal reference and
check convergence under label inversion with both strict and loose tolerances.

`predict_proba` returns columns
$[1-p,p]$; `predict` returns 1 when $p\geq\texttt{threshold}$, including ties.
Training requires both classes. The objective is convex, but unregularized strictly
separable data have no finite maximum-likelihood solution: weights can grow while loss
approaches zero. Regularization and explicit convergence reporting matter.

Precision is $TP/(TP+FP)$, recall is $TP/(TP+FN)$, and F1 is
$2TP/(2TP+FP+FN)$. Undefined binary ratios return zero. These are threshold-dependent;
log loss evaluates probabilities and clips only at float64 machine precision in the metric.

## K-Means: alternating minimization

Let $c_i\in\{0,\ldots,k-1\}$ denote assignments and $\mu_j\in\mathbb{R}^d$ centers.
The objective, also called inertia, is

$$
\boxed{J(c,\mu)=\sum_{i=1}^{n}\|x_i-\mu_{c_i}\|_2^2.}
$$

For fixed centers, each sample independently minimizes its distance:

$$c_i\leftarrow\arg\min_j\|x_i-\mu_j\|_2^2.$$

For fixed assignments, differentiating the contribution of a nonempty cluster gives
$2\sum_{i:c_i=j}(\mu_j-x_i)=0$, hence

$$\mu_j\leftarrow\frac{1}{|C_j|}\sum_{i\in C_j}x_i.$$

These steps cannot increase the exact objective. They reach a local stationary partition,
not necessarily a global optimum. The [implementation](../src/ml_from_scratch/cluster/kmeans.py)
performs multiple starts and retains the smallest final inertia.

Random initialization samples observations without replacement. K-Means++ chooses the first
center uniformly and subsequent centers with probability proportional to squared distance
to the nearest chosen center. This is the basic single-candidate construction; scikit-learn's
greedy variant can choose different starts even for the same integer seed.
[Reference clustering documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
describes the reference algorithm and its initialization.

An empty cluster has no mean. It is reseeded at a sample farthest from the occupied updated
centers; distances refresh after each reseed. With fewer distinct observations than $k$,
some centers must coincide, so fewer occupied labels are allowed. Ties go to the first center.
Stopping uses unchanged assignments or absolute Frobenius center movement at most `tol`.
Final labels and inertia are recomputed against the published centers.

Cluster IDs have no ordering or class meaning. For comparison, use inertia or the adjusted
Rand index (ARI). If $n_{ij}$ is a contingency count and $a_i,b_j$ its row/column totals,

$$
\operatorname{ARI}=\frac{\sum_{ij}\binom{n_{ij}}2-E}
{\tfrac12[\sum_i\binom{a_i}2+\sum_j\binom{b_j}2]-E},\qquad
E=\frac{\sum_i\binom{a_i}2\sum_j\binom{b_j}2}{\binom n2}.
$$

This corrects pair agreement for chance and is invariant under relabeling. Identical
degenerate partitions receive 1. Ground-truth class ARI is an external evaluation;
K-Means never sees those labels while fitting.

## PCA: variance maximization and reconstruction

Center the training features: $X_c=X-\mathbf{1}\bar x^T$. The sample covariance is
$S=X_c^TX_c/(n-1)$. For a unit direction $v$, projected sample variance is $v^TSv$.
Maximize it with a Lagrange multiplier:

$$
\mathcal{L}(v,\lambda)=v^TSv-\lambda(v^Tv-1),\qquad
\nabla_v\mathcal{L}=2Sv-2\lambda v=0.
$$

Thus principal directions are covariance eigenvectors. The largest eigenvalue gives
the greatest variance; subsequent directions maximize variance subject to orthogonality.

The [implementation](../src/ml_from_scratch/decomposition/pca.py) uses thin SVD:

$$
X_c=U\Sigma V^T,\qquad S=V\frac{\Sigma^2}{n-1}V^T.
$$

This avoids explicitly forming the covariance matrix. Singular values are returned in
descending order by [NumPy's SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).
The rows of `components_` are the first $k$ rows of $V^T$:

$$
Z=X_cV_k,\qquad \hat X=ZV_k^T+\mathbf{1}\bar x^T,\qquad
\lambda_j=\sigma_j^2/(n-1),\quad
\rho_j=\lambda_j/\sum_\ell\lambda_\ell.
$$

The denominator includes all available components, including discarded ones. Constant
data return zero ratios rather than $0/0$. `n_components=None` retains the thin-SVD basis;
at most $\min(n,d)$ components are available, and the centered rank is at most $n-1$.

The truncated reconstruction minimizes squared Frobenius reconstruction error among
rank-$k$ approximations to $X_c$; the residual sum of squares is
$\sum_{j>k}\sigma_j^2$. With all available directions, training samples reconstruct up
to floating point rounding. A thin basis for wide data need not reconstruct arbitrary
new points exactly.

Both $v$ and $-v$ represent the same axis. Repeated eigenvalues allow whole rotations within
their eigenspaces. Tests compare reconstruction, variance and projectors $V_kV_k^T$, rather
than raw signed coordinates. If a repeated eigenvalue straddles the truncation boundary,
even the selected $k$-dimensional subspace need not be unique.

PCA centers but does not standardize or whiten. Changing feature units changes covariance
and therefore changes the projection; preprocessing is an explicit experimental decision.
