"""Exercise 2: Non-Linearity in Higher Dimensions.

Gera o Dataset I (gaussianas deslocadas em 5D, item A) e o Dataset II (cascas
concêntricas em 5D, item B), projeta ambos com PCA (item C) e avalia uma
função de separação não linear para o Dataset II (item D3). Salva as figuras
em ``figures/`` e imprime os números que alimentam o relatório e a tabela
*Results summary*.

Uso (a partir da raiz do repositório):

    python docs/exercises/data/code/exercise2_high_dim.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"

RNG = np.random.default_rng(42)  # (1)!

N_PER_CLASS = 500

MU_A = np.zeros(5)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(5, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

RADIUS_C = (2.0, 0.4)  # (media, desvio padrao), classe núcleo
RADIUS_D = (5.0, 0.4)  # (media, desvio padrao), classe casca


def dataset_i() -> tuple[np.ndarray, np.ndarray]:
    """Dataset I: duas gaussianas 5D deslocadas (classes A=0, B=1)."""
    XA = RNG.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    XB = RNG.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    X = np.vstack([XA, XB])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


def random_unit_vectors(n: int, dim: int = 5) -> np.ndarray:
    """Direções uniformes na esfera unitária de R^dim (item B1)."""
    v = RNG.standard_normal(size=(n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def dataset_ii() -> tuple[np.ndarray, np.ndarray]:
    """Dataset II: cascas concêntricas 5D (classe C=núcleo=0, D=casca=1)."""
    u_c = random_unit_vectors(N_PER_CLASS)
    rho_c = RNG.normal(RADIUS_C[0], RADIUS_C[1], size=N_PER_CLASS)
    XC = rho_c[:, None] * u_c

    u_d = random_unit_vectors(N_PER_CLASS)
    rho_d = RNG.normal(RADIUS_D[0], RADIUS_D[1], size=N_PER_CLASS)
    XD = rho_d[:, None] * u_d

    X = np.vstack([XC, XD])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])
    return X, y


def figure_4(X1, y1, X2, y2, ev1, ev2) -> None:
    """Figura 4: projeção PCA 2D dos dois datasets, lado a lado."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for c, label, color in ((0, "Classe A", "tab:blue"), (1, "Classe B", "tab:orange")):
        axes[0].scatter(*X1[y1 == c].T, s=10, alpha=0.6, color=color, label=label)
    axes[0].set_title(f"Dataset I (PC1+PC2 = {ev1[:2].sum() * 100:.1f}% da variância)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()

    for c, label, color in ((0, "Classe C (núcleo)", "tab:green"), (1, "Classe D (casca)", "tab:red")):
        axes[1].scatter(*X2[y2 == c].T, s=10, alpha=0.6, color=color, label=label)
    axes[1].set_title(f"Dataset II (PC1+PC2 = {ev2[:2].sum() * 100:.1f}% da variância)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-pca-projections.png", dpi=160)
    plt.close(fig)


def figure_5(X1, y1, X2, y2) -> None:
    """Figura 5: histograma do raio ||x|| em 5D, com as duas classes sobrepostas."""
    r1 = np.linalg.norm(X1, axis=1)
    r2 = np.linalg.norm(X2, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(r1[y1 == 0], bins=30, alpha=0.6, color="tab:blue", label="Classe A")
    axes[0].hist(r1[y1 == 1], bins=30, alpha=0.6, color="tab:orange", label="Classe B")
    axes[0].set_title("Dataset I: raio $\\|x\\|$ em 5D")
    axes[0].set_xlabel("$\\|x\\|$")
    axes[0].set_ylabel("Contagem")
    axes[0].legend()

    axes[1].hist(r2[y2 == 0], bins=30, alpha=0.6, color="tab:green", label="Classe C (núcleo)")
    axes[1].hist(r2[y2 == 1], bins=30, alpha=0.6, color="tab:red", label="Classe D (casca)")
    axes[1].set_title("Dataset II: raio $\\|x\\|$ em 5D")
    axes[1].set_xlabel("$\\|x\\|$")
    axes[1].set_ylabel("Contagem")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-radius-histograms.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    X1, y1 = dataset_i()
    X2, y2 = dataset_ii()

    pca1 = PCA(n_components=2, random_state=42).fit(X1)
    pca2 = PCA(n_components=2, random_state=42).fit(X2)
    P1 = pca1.transform(X1)
    P2 = pca2.transform(X2)
    ev1 = pca1.explained_variance_ratio_
    ev2 = pca2.explained_variance_ratio_

    figure_4(P1, y1, P2, y2, ev1, ev2)
    figure_5(X1, y1, X2, y2)

    print(f"Dataset I : variancia explicada PC1={ev1[0]:.4f} PC2={ev1[1]:.4f} soma={ev1[:2].sum():.4f}")
    print(f"Dataset II: variancia explicada PC1={ev2[0]:.4f} PC2={ev2[1]:.4f} soma={ev2[:2].sum():.4f}")

    centerA, centerB = X1[y1 == 0].mean(axis=0), X1[y1 == 1].mean(axis=0)
    centerC, centerD = X2[y2 == 0].mean(axis=0), X2[y2 == 1].mean(axis=0)
    dist1 = float(np.linalg.norm(centerA - centerB))
    dist2 = float(np.linalg.norm(centerC - centerD))
    print(f"Distancia entre centros, Dataset I: {dist1:.4f}")
    print(f"Distancia entre centros, Dataset II: {dist2:.4f}")

    r2 = np.linalg.norm(X2, axis=1)
    mean_radius_c = float(r2[y2 == 0].mean())
    mean_radius_d = float(r2[y2 == 1].mean())
    print(f"Raio medio, casca interna (C): {mean_radius_c:.4f}")
    print(f"Raio medio, casca externa (D): {mean_radius_d:.4f}")

    # Item D3: proposta de funcao separadora nao linear para o Dataset II,
    # baseada em ||x||^2 = soma dos xi^2 (nunca treinada, so avaliada).
    threshold = ((RADIUS_C[0] + RADIUS_D[0]) / 2) ** 2  # (2)!
    f_x = np.sum(X2 ** 2, axis=1)
    pred = (f_x > threshold).astype(float)
    accuracy = float(np.mean(pred == y2))
    print(f"f(x) = ||x||^2 com limiar {threshold:.4f} separa o Dataset II com acuracia {accuracy:.4f}")


if __name__ == "__main__":
    main()
