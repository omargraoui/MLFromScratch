"""Regenerate portfolio figures from fitted estimators; no stored or invented results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_regression,
)
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from ml_from_scratch import PCA, GradientDescent, KMeans, LinearRegression, LogisticRegression

SEED = 42
BLUE = "#245d91"
TEAL = "#008c87"
ORANGE = "#d57a25"
COLORS = [BLUE, TEAL, ORANGE]


def configure_style() -> None:
    """Use readable typography, restrained colors, and a consistent export resolution."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#26354a",
            "text.color": "#26354a",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "savefig.dpi": 170,
        }
    )


def save(figure: Figure, output: Path, name: str) -> None:
    """Write an image and release its memory for headless script execution."""
    figure.savefig(output / name, bbox_inches="tight", metadata={"Software": "Matplotlib"})
    plt.close(figure)


def linear_regression_figure(output: Path) -> None:
    """Visualize held-out predictive agreement and the MSE optimization trace."""
    features, targets = make_regression(
        n_samples=600, n_features=8, n_informative=6, noise=10, bias=12, random_state=SEED
    )
    train, test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, random_state=SEED
    )
    mean, scale = train.mean(axis=0), train.std(axis=0)
    train, test = (train - mean) / scale, (test - mean) / scale
    model = LinearRegression(solver="gd", learning_rate=0.1, max_iter=5000, tol=1e-7)
    model.fit(train, y_train)
    reference = SklearnLinearRegression().fit(train, y_train)
    predictions = model.predict(test)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].scatter(y_test, predictions, color=BLUE, alpha=0.75, s=24, label="From scratch")
    bounds = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    axes[0].plot(bounds, bounds, "--", color=ORANGE, label="Perfect prediction")
    axes[0].set(
        xlabel="Observed target",
        ylabel="Predicted target",
        title=f"Held-out predictions · R² = {r2_score(y_test, predictions):.4f}",
    )
    axes[0].legend(frameon=False)
    axes[1].semilogy(model.loss_history_, color=BLUE, linewidth=2, label="Gradient descent")
    reference_mse = np.mean((reference.predict(train) - y_train) ** 2)
    axes[1].axhline(reference_mse, color=ORANGE, linestyle="--", label="sklearn least squares")
    axes[1].set(
        xlabel="Gradient updates", ylabel="Training MSE", title="Convergence to least squares"
    )
    axes[1].legend(frameon=False)
    figure.suptitle("Linear regression | eight features, fixed train/test split", fontsize=14)
    save(figure, output, "linear_regression.png")


def logistic_regression_figure(output: Path) -> None:
    """Show a genuinely two-feature model's probability surface and training objective."""
    dataset = load_breast_cancer()
    features, targets = dataset.data[:, :2], dataset.target
    train, test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25, stratify=targets, random_state=SEED
    )
    mean, scale = train.mean(axis=0), train.std(axis=0)
    train, test = (train - mean) / scale, (test - mean) / scale
    model = LogisticRegression(learning_rate=0.5, max_iter=10_000, tol=1e-7, l2=0.01)
    model.fit(train, y_train)
    horizontal, vertical = np.meshgrid(np.linspace(-2.5, 4, 180), np.linspace(-2.5, 4, 180))
    grid = np.column_stack((horizontal.ravel(), vertical.ravel()))
    probability = model.predict_proba(grid)[:, 1].reshape(horizontal.shape)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    field = axes[0].contourf(
        horizontal, vertical, probability, levels=np.linspace(0, 1, 11), cmap="Blues", alpha=0.65
    )
    axes[0].contour(horizontal, vertical, probability, levels=[0.5], colors=[ORANGE], linewidths=2)
    for label, color in enumerate([ORANGE, BLUE]):
        subset = test[y_test == label]
        axes[0].scatter(
            subset[:, 0],
            subset[:, 1],
            s=23,
            color=color,
            edgecolors="white",
            linewidths=0.5,
            label=dataset.target_names[label],
        )
    axes[0].set(
        xlabel="Mean radius (training z-score)",
        ylabel="Mean texture (training z-score)",
        title=f"Held-out points · accuracy = {accuracy_score(y_test, model.predict(test)):.3f}",
    )
    axes[0].legend(frameon=False, loc="upper right")
    figure.colorbar(field, ax=axes[0], label="P(benign)", shrink=0.8)
    axes[1].plot(model.loss_history_, color=BLUE, linewidth=2)
    axes[1].set(
        xlabel="Gradient updates",
        ylabel="Mean BCE + L2 penalty",
        title="Stable binary objective · λ = 0.01",
    )
    figure.suptitle(
        "Logistic regression | two-feature educational view of breast cancer data", fontsize=13
    )
    save(figure, output, "logistic_regression.png")


def gradient_descent_figure(output: Path) -> None:
    """Compare stable step sizes on an anisotropic convex quadratic."""
    curvature = np.array([1.0, 9.0])
    optimum = np.array([2.0, -1.0])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        displacement = parameters - optimum
        return float(0.5 * np.sum(curvature * displacement**2)), curvature * displacement

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for learning_rate, color in zip((0.02, 0.1, 0.21), COLORS, strict=True):
        optimizer = GradientDescent(learning_rate=learning_rate, max_iter=1500, tol=1e-9)
        optimizer.minimize(objective, np.array([-4.0, 4.0]))
        axes[0].semilogy(
            np.maximum(optimizer.loss_history_, 1e-20),
            color=color,
            linewidth=2,
            label=f"α = {learning_rate}",
        )
    axes[0].set(
        xlabel="Gradient updates",
        ylabel="Objective (log scale)",
        title="Step size controls progress",
    )
    axes[0].legend(frameon=False)
    horizontal, vertical = np.meshgrid(np.linspace(-4.5, 4, 160), np.linspace(-3, 4.5, 160))
    values = 0.5 * ((horizontal - 2) ** 2 + 9 * (vertical + 1) ** 2)
    axes[1].contour(horizontal, vertical, values, levels=[1, 3, 10, 30, 60, 100], cmap="Blues")
    axes[1].scatter([2], [-1], color=ORANGE, s=100, marker="*", label="Analytic minimum (2, −1)")
    axes[1].scatter([-4], [4], color=BLUE, s=40, label="Shared initial parameters")
    axes[1].set(
        xlabel="First parameter", ylabel="Second parameter", title="Unequal curvature: diag(1, 9)"
    )
    axes[1].legend(frameon=False, loc="lower left", fontsize=9)
    figure.suptitle("Gradient descent | same objective, different learning rates", fontsize=14)
    save(figure, output, "gradient_descent.png")


def kmeans_figure(output: Path) -> None:
    """Plot a two-dimensional view while clustering all four standardized Iris features."""
    dataset = load_iris()
    features = (dataset.data - dataset.data.mean(axis=0)) / dataset.data.std(axis=0)
    model = KMeans(n_clusters=3, n_init=10, random_state=SEED, tol=1e-6).fit(features)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for label, color in enumerate(COLORS):
        subset = features[model.labels_ == label]
        axes[0].scatter(
            subset[:, 0], subset[:, 2], color=color, s=30, alpha=0.8, label=f"Cluster {label + 1}"
        )
    axes[0].scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 2],
        marker="X",
        color="black",
        edgecolors="white",
        linewidths=1,
        s=140,
        label="Centroids",
    )
    axes[0].set(
        xlabel="Sepal length (z-score)",
        ylabel="Petal length (z-score)",
        title="Iris · displayed in two of four fitted dimensions",
    )
    axes[0].legend(frameon=False, fontsize=9)
    cluster_counts = range(1, 8)
    inertias = [
        KMeans(n_clusters=count, n_init=10, random_state=SEED).fit(features).inertia_
        for count in cluster_counts
    ]
    axes[1].plot(list(cluster_counts), inertias, "o-", color=BLUE, linewidth=2)
    axes[1].axvline(3, color=ORANGE, linestyle="--", label="k = 3 shown at left")
    axes[1].set(
        xlabel="Number of clusters",
        ylabel="Within-cluster sum of squared distances",
        title="Inertia decreases with model capacity",
        xticks=list(cluster_counts),
    )
    axes[1].legend(frameon=False)
    ari = adjusted_rand_score(dataset.target, model.labels_)
    figure.suptitle(
        f"K-Means | known species used only for evaluation · ARI = {ari:.3f}", fontsize=14
    )
    save(figure, output, "kmeans.png")


def pca_figure(output: Path) -> None:
    """Connect low-dimensional projection, retained variance, and image reconstruction."""
    wine = load_wine()
    features = (wine.data - wine.data.mean(axis=0)) / wine.data.std(axis=0)
    projection = PCA(n_components=2).fit_transform(features)
    digits = load_digits()
    pixels = digits.data / 16.0
    full = PCA().fit(pixels)
    compressed = PCA(n_components=20).fit(pixels)
    reconstruction = compressed.inverse_transform(compressed.transform(pixels))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for label, color in enumerate(COLORS):
        subset = projection[wine.target == label]
        axes[0].scatter(
            subset[:, 0],
            subset[:, 1],
            color=color,
            s=22,
            alpha=0.8,
            label=f"Wine class {label + 1}",
        )
    axes[0].set(
        xlabel="Principal component 1",
        ylabel="Principal component 2",
        title="Wine · unsupervised 2D projection",
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(
        np.arange(1, 65), np.cumsum(full.explained_variance_ratio_), color=BLUE, linewidth=2
    )
    retained = compressed.explained_variance_ratio_.sum()
    axes[1].scatter(
        [20], [retained], color=ORANGE, s=50, zorder=3, label=f"20 components: {retained:.1%}"
    )
    axes[1].set(
        xlabel="Components retained",
        ylabel="Cumulative explained variance ratio",
        title="Digits · variance retained",
        ylim=(0, 1.04),
    )
    axes[1].legend(frameon=False)
    originals = np.concatenate([pixels[i].reshape(8, 8) for i in range(4)], axis=1)
    recovered = np.concatenate([reconstruction[i].reshape(8, 8) for i in range(4)], axis=1)
    mosaic = np.concatenate([originals, np.ones((1, 32)), recovered], axis=0)
    axes[2].imshow(mosaic, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set(title="Digits · original / 20-component reconstruction", xticks=[], yticks=[])
    axes[2].grid(False)
    figure.suptitle("PCA | centering, SVD, and a measurable information tradeoff", fontsize=14)
    save(figure, output, "pca.png")


def main() -> None:
    """Generate all five figures without opening interactive windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    with threadpool_limits(limits=1):
        for generator in (
            linear_regression_figure,
            logistic_regression_figure,
            gradient_descent_figure,
            kmeans_figure,
            pca_figure,
        ):
            generator(args.output_dir)
    print(f"Generated five figures in {args.output_dir}")


if __name__ == "__main__":
    main()
