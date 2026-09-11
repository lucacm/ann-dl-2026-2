"""Exercise 1: Point Clouds, Geometry and Spread in 2D.

Gera as 4 classes gaussianas do enunciado (item A), as versões reescalonadas
(item B) e a visualização de fronteiras por centróide mais próximo (item C).
Salva as figuras em ``figures/`` e imprime os números que alimentam o
relatório e a tabela *Results summary*.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise1_point_clouds.py
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"

CLASSES = {
    0: {"mean": np.array([2.0, 3.0]), "std": np.array([0.8, 2.5])},
    1: {"mean": np.array([5.0, 6.0]), "std": np.array([1.2, 1.9])},
    2: {"mean": np.array([8.0, 1.0]), "std": np.array([0.9, 0.9])},
    3: {"mean": np.array([15.0, 4.0]), "std": np.array([0.5, 2.0])},
}
N_PER_CLASS = 100
SCALES = (0.5, 1.0, 2.0, 4.0)
COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}

RNG = np.random.default_rng(42)  # (1)!

# Ruído-base padrão N(0, I), sorteado uma única vez e reaproveitado em todas
# as escalas: as 4 versões do item B são literalmente as mesmas nuvens mais
# ou menos espalhadas, não sorteios independentes.
BASE_Z = {c: RNG.standard_normal(size=(N_PER_CLASS, 2)) for c in CLASSES}


def generate(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Amostra as 4 classes com os desvios multiplicados por ``scale``."""
    xs, ys = [], []
    for c, params in CLASSES.items():
        xs.append(params["mean"] + scale * params["std"] * BASE_Z[c])
        ys.append(np.full(N_PER_CLASS, c))
    return np.vstack(xs), np.concatenate(ys)


def sigma_bar(c: int) -> float:
    """sigma_bar_k = (sigma_kx + sigma_ky) / 2, a partir dos parâmetros da classe."""
    return float(CLASSES[c]["std"].mean())


def separation_ratios(scale: float = 1.0) -> dict[tuple[int, int], float]:
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) para cada par de classes."""
    ratios = {}
    for i, j in combinations(CLASSES, 2):
        dist = np.linalg.norm(CLASSES[i]["mean"] - CLASSES[j]["mean"])
        denom = scale * (sigma_bar(i) + sigma_bar(j))
        ratios[(i, j)] = float(dist / denom)
    return ratios


def mixing_rate(scale: float) -> float:
    """Fração de pontos cujo centro de classe mais próximo não é o da própria classe."""
    X, y = generate(scale)
    means = np.stack([CLASSES[c]["mean"] for c in CLASSES])
    dists = np.linalg.norm(X[:, None, :] - means[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    return float(np.mean(nearest != y))


def plot_clouds(ax, X, y, means, title):
    for c in CLASSES:
        ax.scatter(*X[y == c].T, s=14, alpha=0.75, color=COLORS[c], label=f"Classe {c}")
    for c in CLASSES:
        ax.scatter(*means[c], marker="X", s=140, color=COLORS[c], edgecolor="black", linewidth=1.2, zorder=5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)


def figure_1() -> None:
    """Figura 1: dispersão das 4 classes em scale=1.0, com centros marcados."""
    X, y = generate(1.0)
    means = np.stack([CLASSES[c]["mean"] for c in CLASSES])
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_clouds(ax, X, y, means, "Nuvens de pontos gaussianas (scale = 1.0)")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-point-clouds.png", dpi=160)
    plt.close(fig)  # (2)!


def figure_1b() -> None:
    """Figura 1b: mesma dispersão com as regiões de centróide mais próximo."""
    X, y = generate(1.0)
    means = np.stack([CLASSES[c]["mean"] for c in CLASSES])

    margin = 1.5
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_dists = np.linalg.norm(grid[:, None, :] - means[None, :, :], axis=2)
    region = grid_dists.argmin(axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.matplotlib.colors.ListedColormap([COLORS[c] for c in CLASSES])
    ax.contourf(xx, yy, region, levels=np.arange(-0.5, 4.5, 1), cmap=cmap, alpha=0.15)
    ax.contour(xx, yy, region, levels=np.arange(0.5, 4, 1), colors="black", linewidths=1.0, linestyles="--")
    plot_clouds(ax, X, y, means, "Fronteiras por centróide mais próximo (scale = 1.0)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01b-decision-boundaries.png", dpi=160)
    plt.close(fig)


def figure_2() -> None:
    """Figura 2: 4 subplots, um por valor de s, com eixos compartilhados."""
    datasets = {s: generate(s) for s in SCALES}
    means = np.stack([CLASSES[c]["mean"] for c in CLASSES])

    max_std = max(CLASSES[c]["std"].max() for c in CLASSES) * max(SCALES)
    margin = 3 * max_std
    x_min = min(CLASSES[c]["mean"][0] for c in CLASSES) - margin
    x_max = max(CLASSES[c]["mean"][0] for c in CLASSES) + margin
    y_min = min(CLASSES[c]["mean"][1] for c in CLASSES) - margin
    y_max = max(CLASSES[c]["mean"][1] for c in CLASSES) + margin

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), SCALES):
        X, y = datasets[s]
        plot_clouds(ax, X, y, means, f"s = {s}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Nuvens de pontos para diferentes fatores de escala do desvio padrão")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-scaled-clouds.png", dpi=160)
    plt.close(fig)


def figure_3(rates: dict[float, float]) -> None:
    """Figura 3: taxa de mistura em função de s."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(SCALES, [rates[s] for s in SCALES], marker="o", color="tab:purple")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("Fator de escala $s$")
    ax.set_ylabel("Taxa de mistura")
    ax.set_title("Taxa de mistura em função do fator de escala do desvio padrão")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-mixing-rate.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    figure_1()
    figure_1b()
    figure_2()

    ratios_s1 = separation_ratios(1.0)
    print("Separation ratio r_ij em s = 1.0:")
    for (i, j), r in ratios_s1.items():
        print(f"  r_{i}{j} = {r:.4f}")
    smallest_pair, smallest_r = min(ratios_s1.items(), key=lambda kv: kv[1])
    print(f"Menor r_ij em s = 1.0: par {smallest_pair} = {smallest_r:.4f}")
    print(f"Mesmo par em s = 2.0 (sem regenerar dados): {smallest_r / 2:.4f}")

    rates = {s: mixing_rate(s) for s in SCALES}
    for s in SCALES:
        print(f"Taxa de mistura em s = {s}: {rates[s]:.4f} ({rates[s] * 100:.2f}%)")

    figure_3(rates)


if __name__ == "__main__":
    main()
