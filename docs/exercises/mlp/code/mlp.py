"""Exercise 3: Multi-Layer Perceptron (MLP).

Implementa um MLP do zero em numpy (ativações, loss, forward, backward e atualização) e o
reaproveita, sem nenhuma alteração, nos Exercícios 2 (classificação binária), 3
(multiclasse) e 4 (profundidade). O Exercício 1 (backpropagation à mão) está resolvido no
próprio relatório. Salva as 6 figuras em ``figures/`` e imprime os números que alimentam o
relatório e a tabela *Results summary*.

Uso (a partir da raiz do repositório):

    python docs/exercises/mlp/code/mlp.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --8<-- [start:setup]
RNG = np.random.default_rng(42)  # (1)!

COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
# --8<-- [end:setup]


# --8<-- [start:mlp]
def init_params(sizes, rng):
    """Inicializa uma rede com camadas ``sizes`` (ex. ``[2, 8, 1]``).

    Pesos de Xavier/Glorot, ``N(0, 1/fan_in)``, adequados ao tanh: mantêm a variância da
    pré-ativação perto de 1 em cada camada, longe da saturação. Vieses começam em zero.
    """
    return [
        (rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out)), np.zeros(n_out))
        for n_in, n_out in zip(sizes[:-1], sizes[1:])
    ]


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def softmax(u):
    e = np.exp(u - u.max(axis=1, keepdims=True))  # subtrair o máximo evita overflow
    return e / e.sum(axis=1, keepdims=True)


OUTPUTS = {"sigmoid": sigmoid, "softmax": softmax}


def forward(X, params, output):
    """Passa o lote ``X`` (N x d) pela rede e devolve ``(y_hat, cache)``.

    Camadas ocultas usam tanh; a camada de saída usa ``output`` ("sigmoid" ou "softmax").
    ``cache`` guarda a entrada de cada camada (``X``, ``h1``, ``h2``, ...), que é tudo o que
    o backward precisa.
    """
    cache = [X]
    h = X
    for W, b in params[:-1]:
        h = np.tanh(h @ W + b)
        cache.append(h)
    W, b = params[-1]
    return OUTPUTS[output](h @ W + b), cache


def loss(y_hat, Y):
    """Entropia cruzada média sobre o lote.

    Com uma coluna de saída (sigmoid) é a BCE, ``-[y log y_hat + (1-y) log(1-y_hat)]``;
    com K colunas (softmax, ``Y`` one-hot) é a entropia cruzada categórica,
    ``-sum_k y_k log y_hat_k``. O ``clip`` evita ``log(0)``.
    """
    p = np.clip(y_hat, 1e-12, 1 - 1e-12)
    if Y.shape[1] == 1:
        return float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))
    return float(-np.mean(np.sum(Y * np.log(p), axis=1)))


def backward(y_hat, Y, cache, params):
    """Gradientes de ``loss`` em relação a todos os pesos e vieses.

    Tanto sigmoid + BCE quanto softmax + entropia cruzada dão o mesmo gradiente na
    pré-ativação da saída, ``dL/du = (y_hat - y) / N``, então uma única função atende os
    dois casos. Daí para trás, cada camada oculta multiplica pela derivada do tanh,
    ``1 - h**2``, calculada a partir da própria saída ``h`` guardada no cache.
    """
    N = Y.shape[0]
    delta = (y_hat - Y) / N  # dL/du da camada de saída
    grads = [None] * len(params)
    for layer in range(len(params) - 1, -1, -1):
        h_in = cache[layer]
        W, _ = params[layer]
        grads[layer] = (h_in.T @ delta, delta.sum(axis=0))
        if layer > 0:
            delta = (delta @ W.T) * (1 - h_in**2)  # dL/dz da camada anterior
    return grads


def update(params, grads, eta):
    """Um passo de gradiente descendente: ``theta <- theta - eta * dL/dtheta``."""
    return [(W - eta * dW, b - eta * db) for (W, b), (dW, db) in zip(params, grads)]


def train(X, Y, sizes, eta, epochs, output, rng):
    """Treina com gradiente descendente em lote completo e registra a loss por época."""
    params = init_params(sizes, rng)
    history = []
    for _ in range(epochs):
        y_hat, cache = forward(X, params, output)
        history.append(loss(y_hat, Y))
        params = update(params, backward(y_hat, Y, cache, params), eta)
    return params, np.array(history)


def predict(X, params, output):
    y_hat, _ = forward(X, params, output)
    if output == "sigmoid":
        return (y_hat[:, 0] >= 0.5).astype(int)
    return y_hat.argmax(axis=1)


def n_params(sizes):
    return sum(n_in * n_out + n_out for n_in, n_out in zip(sizes[:-1], sizes[1:]))
# --8<-- [end:mlp]


def gradient_check(rng):
    """Compara ``backward`` com diferenças finitas centradas numa rede pequena.

    Roda nos dois modos de saída e devolve o maior erro relativo encontrado.
    """
    worst = 0.0
    for output, k in (("sigmoid", 1), ("softmax", 3)):
        X = rng.normal(size=(5, 3))
        labels = rng.integers(0, max(k, 2), size=5)
        Y = labels[:, None].astype(float) if k == 1 else np.eye(k)[labels]
        params = init_params([3, 4, 4, k], rng)
        y_hat, cache = forward(X, params, output)
        grads = backward(y_hat, Y, cache, params)
        eps = 1e-6
        for (W, b), (dW, db) in zip(params, grads):
            for arr, g in ((W, dW), (b, db)):
                num = np.zeros_like(arr)
                for idx in np.ndindex(arr.shape):
                    old = arr[idx]
                    arr[idx] = old + eps
                    lp = loss(forward(X, params, output)[0], Y)
                    arr[idx] = old - eps
                    lm = loss(forward(X, params, output)[0], Y)
                    arr[idx] = old
                    num[idx] = (lp - lm) / (2 * eps)
                rel = np.abs(num - g).max() / max(np.abs(num).max() + np.abs(g).max(), 1e-12)
                worst = max(worst, rel)
    return worst


def plot_loss(curves, title, path):
    """Loss de treino x época; ``curves`` é ``{rótulo: histórico}``."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, hist in curves.items():
        ax.plot(np.arange(1, len(hist) + 1), hist, label=label)
    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss de treino (entropia cruzada)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --8<-- [start:ex2_data]
X2, c2 = make_blobs(
    n_samples=[500, 250, 250],
    centers=[[0, 0], [3, 3], [-3, -3]],
    cluster_std=1.2,
    random_state=42,
)
y2 = (c2 > 0).astype(int)  # (1)!
X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)
# --8<-- [end:ex2_data]


def ex2_figure1():
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for cls in (0, 1):
        m = y2 == cls
        ax.scatter(X2[m, 0], X2[m, 1], s=12, alpha=0.7, color=COLORS[cls],
                   label=f"Classe {cls} ({m.sum()} pontos)")
    ax.set_title("Figura 1. Dataset do Exercício 2 (1000 pontos)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-blobs-scatter.png", dpi=150)
    plt.close(fig)


# --8<-- [start:ex2_baseline]
baseline2 = LogisticRegression().fit(X2_tr, y2_tr)
acc_lin2 = baseline2.score(X2_te, y2_te)
# --8<-- [end:ex2_baseline]

# --8<-- [start:ex2_mlp]
SIZES2, ETA2, EPOCHS2 = [2, 8, 1], 0.5, 2000
params2, hist2 = train(
    X2_tr, y2_tr[:, None].astype(float), SIZES2, ETA2, EPOCHS2, "sigmoid", RNG
)
acc_mlp2 = np.mean(predict(X2_te, params2, "sigmoid") == y2_te)
# --8<-- [end:ex2_mlp]


def ex2_figure3():
    lo, hi = X2.min(axis=0) - 1, X2.max(axis=0) + 1
    g1, g2 = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
    grid = np.c_[g1.ravel(), g2.ravel()]
    region = predict(grid, params2, "sigmoid").reshape(g1.shape)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.contourf(g1, g2, region, levels=[-0.5, 0.5, 1.5],
                colors=[COLORS[0], COLORS[1]], alpha=0.2)
    ax.contour(g1, g2, region, levels=[0.5], colors="k", linewidths=1)
    for cls in (0, 1):
        m = y2_te == cls
        ax.scatter(X2_te[m, 0], X2_te[m, 1], s=16, color=COLORS[cls],
                   edgecolor="k", linewidth=0.3, label=f"Teste, classe {cls}")
    ax.plot([], [], color="k", linewidth=1, label="Fronteira de decisão")
    ax.set_title(f"Figura 3. Fronteira do MLP {SIZES2} sobre o conjunto de teste")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-blobs-boundary.png", dpi=150)
    plt.close(fig)


# --8<-- [start:ex3_data]
X3, y3 = make_classification(
    n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
    n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42,
)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(
    X3, y3, test_size=0.2, random_state=42, stratify=y3
)
Y3_tr = np.eye(3)[y3_tr]  # (1)!
# --8<-- [end:ex3_data]

# --8<-- [start:ex3_mlp]
SIZES3, ETA3, EPOCHS3 = [4, 16, 3], 0.5, 2000
params3, hist3 = train(X3_tr, Y3_tr, SIZES3, ETA3, EPOCHS3, "softmax", RNG)
pred3 = predict(X3_te, params3, "softmax")
acc_mlp3 = np.mean(pred3 == y3_te)
cm3 = confusion_matrix(y3_te, pred3)

acc_lin3 = LogisticRegression().fit(X3_tr, y3_tr).score(X3_te, y3_te)
# --8<-- [end:ex3_mlp]


def ex3_figure5():
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm3, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm3[i, j], ha="center", va="center",
                    color="white" if cm3[i, j] > cm3.max() / 2 else "black")
    ax.set_xticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_yticks(range(3), [f"Classe {k}" for k in range(3)])
    ax.set_title(f"Figura 5. Matriz de confusão no teste\nMLP {SIZES3}")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    fig.colorbar(im, ax=ax, label="Número de amostras")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-confusion-matrix.png", dpi=150)
    plt.close(fig)


# --8<-- [start:ex4_mlp]
SIZES4 = [4, 16, 16, 3]  # só a profundidade muda; ETA3, EPOCHS3 e dados iguais
params4, hist4 = train(X3_tr, Y3_tr, SIZES4, ETA3, EPOCHS3, "softmax", RNG)
acc_mlp4 = np.mean(predict(X3_te, params4, "softmax") == y3_te)
# --8<-- [end:ex4_mlp]

# --8<-- [start:ex4_seeds]
SEEDS = [42, 43, 44]
runs = {}
for sizes in (SIZES3, SIZES4):
    final_loss, test_acc = [], []
    for seed in SEEDS:
        p, h = train(X3_tr, Y3_tr, sizes, ETA3, EPOCHS3, "softmax",
                     np.random.default_rng(seed))  # (1)!
        final_loss.append(h[-1])
        test_acc.append(np.mean(predict(X3_te, p, "softmax") == y3_te))
    runs[len(sizes) - 2] = (np.array(final_loss), np.array(test_acc))
# --8<-- [end:ex4_seeds]


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)

    print(f"Gradient check, maior erro relativo: {gradient_check(np.random.default_rng(0)):.2e}")

    ex2_figure1()
    plot_loss({f"MLP {SIZES2}": hist2},
              f"Figura 2. Loss de treino, Exercício 2 (eta = {ETA2})",
              FIGURES / "fig02-blobs-loss.png")
    ex2_figure3()
    print("\n== Exercício 2 ==")
    print(f"Treino/teste: {len(X2_tr)}/{len(X2_te)}; classe 1 no teste: {y2_te.sum()}")
    print(f"Baseline linear, acurácia de teste: {acc_lin2:.2%}")
    print(f"MLP {SIZES2}, eta={ETA2}, épocas={EPOCHS2}: loss inicial {hist2[0]:.4f}, "
          f"final {hist2[-1]:.4f}, acurácia de teste {acc_mlp2:.2%}")

    plot_loss({f"MLP {SIZES3}": hist3},
              f"Figura 4. Loss de treino, Exercício 3 (eta = {ETA3})",
              FIGURES / "fig04-multiclass-loss.png")
    ex3_figure5()
    print("\n== Exercício 3 ==")
    print(f"Treino/teste: {len(X3_tr)}/{len(X3_te)}; por classe no teste: {np.bincount(y3_te)}")
    print(f"MLP {SIZES3}, eta={ETA3}, épocas={EPOCHS3}: loss final {hist3[-1]:.4f}, "
          f"acurácia de teste {acc_mlp3:.2%}")
    print(f"Matriz de confusão (linhas = verdadeira):\n{cm3}")
    print(f"Baseline logístico, acurácia de teste: {acc_lin3:.2%}")

    plot_loss({f"1 camada oculta {SIZES3}": hist3, f"2 camadas ocultas {SIZES4}": hist4},
              f"Figura 6. Loss de treino, raso x profundo (eta = {ETA3})",
              FIGURES / "fig06-depth-loss.png")
    print("\n== Exercício 4 ==")
    print(f"MLP {SIZES4}: loss final {hist4[-1]:.4f}, acurácia de teste {acc_mlp4:.2%}")
    for depth, sizes in ((1, SIZES3), (2, SIZES4)):
        fl, ta = runs[depth]
        print(f"{depth} camada(s) oculta(s) {sizes}, {n_params(sizes)} parâmetros: "
              f"loss final {np.round(fl, 4)} média {fl.mean():.4f}; "
              f"acurácia {np.round(ta * 100, 2)} média {ta.mean():.2%}")
    diff = runs[2][1].mean() - runs[1][1].mean()
    print(f"Diferença média de acurácia (profundo - raso): {diff * 100:+.2f} p.p.")
