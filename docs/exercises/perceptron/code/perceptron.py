"""Exercise 2: Understanding Perceptrons and Their Limitations.

Implementa o perceptron do zero (ativação, predição, regra de atualização e loop de
treino) e o reaproveita, sem alterações, nos dois datasets do enunciado: o separável
(Exercício 1) e o sobreposto (Exercício 2, com o algoritmo *pocket*). Salva as 6 figuras
em ``figures/`` e imprime os números que alimentam o relatório e a tabela
*Results summary*.

Uso (a partir da raiz do repositório):

    python docs/exercises/perceptron/code/perceptron.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --8<-- [start:setup]
RNG = np.random.default_rng(42)  # (1)!

COLORS = {0: "tab:blue", 1: "tab:orange"}
N_PER_CLASS = 1000
# --8<-- [end:setup]


# --8<-- [start:data]
def make_dataset(mean0, cov0, mean1, cov1, n_per_class, rng):
    """Sorteia as duas classes gaussianas e embaralha os pontos uma única vez.

    A ordem resultante fica fixa para todas as épocas de uma chamada de ``train``:
    sem isso, cada época veria primeiro os `n_per_class` pontos da classe 0 e só
    depois os da classe 1, e "re-rodar com eta = 1.0, mudando nada mais" (item D)
    deixaria de ser uma comparação justa.
    """
    X0 = rng.multivariate_normal(mean0, cov0, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov1, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    order = rng.permutation(len(X))
    return X[order], y[order]
# --8<-- [end:data]


# --8<-- [start:perceptron]
def predict(X, w, b):
    """Predição em lote: step(w . x + b), com step(z) = 1 se z >= 0, senao 0."""
    return (X @ w + b >= 0).astype(float)


def accuracy(X, y, w, b):
    return float(np.mean(predict(X, w, b) == y))


def train(X, y, w0, b0, eta, max_epochs=100, pocket=False):
    """Treina um perceptron amostra a amostra com a regra de erro 0/1.

    Para na primeira epoca sem nenhuma atualizacao ou em `max_epochs`. Com
    `pocket=True`, a unica coisa que o loop ganha e' a copia de (w, b) para o
    "bolso" sempre que uma atualizacao produz uma acuracia, no dataset inteiro,
    maior que a melhor ja vista (algoritmo pocket).
    """
    w, b = w0.copy(), b0
    acc_history = []

    pocket_w, pocket_b = w0.copy(), b0
    pocket_acc = accuracy(X, y, w0, b0)
    pocket_epoch = 0
    pocket_history = []

    converged = False
    epochs_run = 0
    updates_history = []
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        n_updates = 0
        for xi, yi in zip(X, y):
            z = w @ xi + b
            yhat = 1.0 if z >= 0 else 0.0
            error = yi - yhat
            if error != 0.0:
                w = w + eta * error * xi
                b = b + eta * error
                n_updates += 1
                if pocket:
                    acc = accuracy(X, y, w, b)
                    if acc > pocket_acc:
                        pocket_acc, pocket_w, pocket_b = acc, w.copy(), b
                        pocket_epoch = epoch

        acc_history.append(accuracy(X, y, w, b))
        updates_history.append(n_updates)
        if pocket:
            pocket_history.append(pocket_acc)
        if n_updates == 0:
            converged = True
            break

    result = {
        "w": w,
        "b": b,
        "epochs": epochs_run,
        "converged": converged,
        "acc_history": np.array(acc_history),
        "updates_history": np.array(updates_history),
        "final_acc": acc_history[-1],
    }
    if pocket:
        result.update(
            pocket_w=pocket_w,
            pocket_b=pocket_b,
            pocket_acc=pocket_acc,
            pocket_epoch=pocket_epoch,
            pocket_history=np.array(pocket_history),
        )
    return result
# --8<-- [end:perceptron]


def plot_scatter(ax, X, y, title):
    for c in (0, 1):
        ax.scatter(*X[y == c].T, s=10, alpha=0.6, color=COLORS[c], label=f"Classe {c}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_boundary(ax, X, y, w, b, label, color, style="-"):
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1 = np.array([x1_min, x1_max])
    x2 = -(w[0] * x1 + b) / w[1]
    ax.plot(x1, x2, style, color=color, linewidth=2, label=label)


def plot_misclassified(ax, X, y, w, b):
    wrong = predict(X, w, b) != y
    ax.scatter(
        *X[wrong].T, s=50, facecolors="none", edgecolors="black",
        linewidths=1.2, label=f"Erros ({wrong.sum()})", zorder=5,
    )


# --8<-- [start:ex1]
MEAN0_EX1 = np.array([1.5, 1.5])
MEAN1_EX1 = np.array([5.0, 5.0])
COV_EX1 = np.array([[0.5, 0.0], [0.0, 0.5]])

X1, y1 = make_dataset(MEAN0_EX1, COV_EX1, MEAN1_EX1, COV_EX1, N_PER_CLASS, RNG)
W0_EX1 = RNG.normal(0, 0.01, size=2)  # (2)!

run_001 = train(X1, y1, W0_EX1, 0.0, eta=0.01, max_epochs=100)
run_100 = train(X1, y1, W0_EX1, 0.0, eta=1.0, max_epochs=100)

dir_001 = run_001["w"] / np.linalg.norm(run_001["w"])
dir_100 = run_100["w"] / np.linalg.norm(run_100["w"])
angle_deg = np.degrees(np.arccos(np.clip(dir_001 @ dir_100, -1.0, 1.0)))
# --8<-- [end:ex1]


# --8<-- [start:ex1_zero]
W0_ZERO = np.zeros(2)
zero_001 = train(X1, y1, W0_ZERO, 0.0, eta=0.01, max_epochs=100)
zero_100 = train(X1, y1, W0_ZERO, 0.0, eta=1.0, max_epochs=100)
zero_ratio_w = zero_100["w"] / zero_001["w"]
zero_ratio_b = zero_100["b"] / zero_001["b"]
# --8<-- [end:ex1_zero]


# --8<-- [start:ex2]
MEAN0_EX2 = np.array([3.0, 3.0])
MEAN1_EX2 = np.array([4.0, 4.0])
COV_EX2 = np.array([[1.5, 0.0], [0.0, 1.5]])

X2, y2 = make_dataset(MEAN0_EX2, COV_EX2, MEAN1_EX2, COV_EX2, N_PER_CLASS, RNG)
W0_EX2 = RNG.normal(0, 0.01, size=2)

run_pocket = train(X2, y2, W0_EX2, 0.0, eta=0.01, max_epochs=100, pocket=True)
# --8<-- [end:ex2]


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Figura 1: dispersão dos dois datasets do Exercício 1.
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_scatter(ax, X1, y1, "Exercício 1: dados separáveis")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-separable-scatter.png", dpi=150)
    plt.close(fig)

    # Figura 2: fronteira de decisão (eta = 0.01) com os erros marcados.
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_scatter(ax, X1, y1, "Exercício 1: fronteira de decisão (η = 0.01)")
    plot_boundary(ax, X1, y1, run_001["w"], run_001["b"], "Fronteira", "black")
    plot_misclassified(ax, X1, y1, run_001["w"], run_001["b"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-separable-boundary.png", dpi=150)
    plt.close(fig)

    # Figura 3: acurácia por época (eta = 0.01).
    fig, ax = plt.subplots(figsize=(5.5, 4))
    epochs = np.arange(1, len(run_001["acc_history"]) + 1)
    ax.plot(epochs, run_001["acc_history"], marker="o", markersize=3, color="tab:blue")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Exercício 1: acurácia × época (η = 0.01)")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # eixo de época só com inteiros
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-separable-accuracy.png", dpi=150)
    plt.close(fig)

    # Figura 4: dispersão do dataset sobreposto do Exercício 2.
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_scatter(ax, X2, y2, "Exercício 2: dados sobrepostos")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-overlapping-scatter.png", dpi=150)
    plt.close(fig)

    # Figura 5: fronteiras final e pocket, com os erros da fronteira final marcados.
    fig, ax = plt.subplots(figsize=(5.5, 5))
    plot_scatter(ax, X2, y2, "Exercício 2: fronteiras final e pocket")
    plot_boundary(ax, X2, y2, run_pocket["w"], run_pocket["b"], "Final", "black", style="--")
    plot_boundary(ax, X2, y2, run_pocket["pocket_w"], run_pocket["pocket_b"], "Pocket", "tab:red")
    plot_misclassified(ax, X2, y2, run_pocket["pocket_w"], run_pocket["pocket_b"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-overlapping-boundary.png", dpi=150)
    plt.close(fig)

    # Figura 6: acurácia atual × acurácia pocket, por época.
    fig, ax = plt.subplots(figsize=(5.5, 4))
    epochs2 = np.arange(1, len(run_pocket["acc_history"]) + 1)
    ax.plot(epochs2, run_pocket["acc_history"], color="black", linestyle="--", label="Atual")
    ax.plot(epochs2, run_pocket["pocket_history"], color="tab:red", label="Pocket")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Exercício 2: acurácia × época (atual vs. pocket)")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-overlapping-accuracy.png", dpi=150)
    plt.close(fig)

    print("=== Exercício 1 ===")
    print(f"w0 (compartilhado) = {W0_EX1}")
    print(f"eta=0.01: w={run_001['w']}, b={run_001['b']:.6f}, "
          f"epochs={run_001['epochs']}, converged={run_001['converged']}, "
          f"acc={run_001['final_acc']:.4f}")
    print(f"eta=1.00: w={run_100['w']}, b={run_100['b']:.6f}, "
          f"epochs={run_100['epochs']}, converged={run_100['converged']}, "
          f"acc={run_100['final_acc']:.4f}")
    print(f"dir(eta=0.01) = {dir_001}")
    print(f"dir(eta=1.00) = {dir_100}")
    print(f"angulo entre as direcoes = {angle_deg:.4f} graus")
    print(f"updates por epoca (eta=0.01) = {run_001['updates_history'].tolist()}")
    print(f"updates por epoca (eta=1.00) = {run_100['updates_history'].tolist()}")

    print("\n=== Exercício 1, partindo de w=0, b=0 ===")
    print(f"eta=0.01: w={zero_001['w']}, b={zero_001['b']:.6f}, epochs={zero_001['epochs']}")
    print(f"eta=1.00: w={zero_100['w']}, b={zero_100['b']:.6f}, epochs={zero_100['epochs']}")
    print(f"razao w(eta=1.0)/w(eta=0.01) = {zero_ratio_w}")
    print(f"razao b(eta=1.0)/b(eta=0.01) = {zero_ratio_b:.4f}")

    print("\n=== Exercício 2 ===")
    print(f"w0 = {W0_EX2}")
    print(f"final:  w={run_pocket['w']}, b={run_pocket['b']:.6f}, "
          f"acc={run_pocket['final_acc']:.4f}")
    print(f"pocket: w={run_pocket['pocket_w']}, b={run_pocket['pocket_b']:.6f}, "
          f"acc={run_pocket['pocket_acc']:.4f}, epoch={run_pocket['pocket_epoch']}")
    print(f"epochs_run={run_pocket['epochs']}, converged={run_pocket['converged']}")
    print(f"updates nas ultimas 5 epocas = {run_pocket['updates_history'][-5:].tolist()}")
    print(f"updates min/max/media (todas as 100 epocas) = "
          f"{run_pocket['updates_history'].min()}/"
          f"{run_pocket['updates_history'].max()}/"
          f"{run_pocket['updates_history'].mean():.1f}")


if __name__ == "__main__":
    main()
