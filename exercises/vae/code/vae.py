"""Exercise 4: Variational Autoencoders (VAE).

Treina dois VAEs no MNIST com a mesma arquitetura, dados, semente, beta e orçamento, mudando
só a dimensão do latente: d = 16 (Exercício 1) e d = 2 (Exercício 2). A reparametrização,
os dois termos da loss (BCE somada sobre os pixels e KL em forma fechada) e a amostragem do
prior são escritos à mão; o PyTorch entra só com camadas lineares, autograd e o Adam.
Salva as 6 figuras em ``figures/`` e imprime os números que alimentam o relatório e a tabela
*Results summary*. O MNIST é baixado na primeira execução para ``code/data/``.

Uso (a partir da raiz do repositório):

    python docs/exercises/vae/code/vae.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
DATA = HERE / "data"

# --8<-- [start:setup]
LR = 1e-3        # Adam
BATCH = 128
EPOCHS = 20
BETA = 1.0       # peso do KL; beta = 1 é o ELBO de verdade
SEED = 42        # torch.manual_seed antes de cada modelo e gerador do split
# --8<-- [end:setup]


# --8<-- [start:data]
def load_mnist(train):
    """MNIST como um tensor (N, 784) com pixels em [0, 1] e um tensor de rótulos (N,).

    ``ToTensor()`` converte cada imagem 28x28 de uint8 para float em [0, 1]; ``view(-1)``
    achata para um vetor de 784. Converter tudo uma vez só, em vez de a cada época dentro
    do DataLoader, deixa o treino várias vezes mais rápido e não muda nenhum valor.
    """
    ds = datasets.MNIST(DATA, train=train, download=True)
    to_tensor = transforms.ToTensor()
    X = torch.stack([to_tensor(img).view(-1) for img, _ in ds])
    return X, ds.targets.clone()


X_full, y_full = load_mnist(train=True)            # 60.000 imagens oficiais de treino
full = TensorDataset(X_full, y_full)
train_set, val_set = random_split(                 # (1)!
    full, [50_000, 10_000], generator=torch.Generator().manual_seed(SEED)
)
# --8<-- [end:data]


# --8<-- [start:model]
class VAE(nn.Module):
    """Encoder 784-512-256 com duas cabeças (mu, log sigma^2) e decoder d-256-512-784."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)   # (1)!
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),  # média de Bernoulli de cada pixel
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def reparameterize(mu, logvar):
    """z = mu + sigma * eps, eps ~ N(0, I).

    A aleatoriedade fica toda em ``eps``, que não depende de nenhum parâmetro; ``z`` vira
    uma função diferenciável de ``mu`` e ``logvar`` e o gradiente passa pela amostragem.
    """
    sigma = torch.exp(0.5 * logvar)   # sigma = exp(log sigma^2 / 2)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps
# --8<-- [end:model]


# --8<-- [start:loss]
def vae_loss_terms(x_hat, x, mu, logvar):
    """Devolve (reconstrução, KL), cada um em nats por imagem, médio sobre o lote.

    Reconstrução: BCE **somada** sobre os 784 pixels (log-verossimilhança de Bernoulli de
    cada pixel), dividida pelo tamanho do lote. ``reduction="mean"`` dividiria também por
    784 e, comparada ao KL somado sobre o latente, equivaleria a usar beta ~ 784.

    KL: forma fechada de KL(N(mu, sigma^2) || N(0, I)), somada sobre as d dimensões do
    latente e média sobre o lote.
    """
    B = x.shape[0]
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / B
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum() / B
    return recon, kl
# --8<-- [end:loss]


# --8<-- [start:train]
@torch.no_grad()
def evaluate(model, dataset, batch=1000):
    """Reconstrução e KL médios por imagem num conjunto inteiro (z amostrado, como no treino)."""
    model.eval()
    recon_sum = kl_sum = 0.0
    for x, _ in DataLoader(dataset, batch_size=batch):
        x_hat, mu, logvar = model(x)
        recon, kl = vae_loss_terms(x_hat, x, mu, logvar)
        recon_sum += recon.item() * len(x)   # desfaz a média do lote para somar por imagem
        kl_sum += kl.item() * len(x)
    n = len(dataset)
    return recon_sum / n, kl_sum / n


def train_vae(d):
    torch.manual_seed(SEED)                          # (1)!
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    hist = {k: [] for k in ("train_recon", "train_kl", "val_recon", "val_kl")}

    for epoch in range(EPOCHS):
        model.train()
        recon_sum = kl_sum = 0.0
        for x, _ in loader:
            x_hat, mu, logvar = model(x)
            recon, kl = vae_loss_terms(x_hat, x, mu, logvar)
            loss = recon + BETA * kl                 # -ELBO por imagem
            opt.zero_grad()
            loss.backward()
            opt.step()
            recon_sum += recon.item() * len(x)       # (2)!
            kl_sum += kl.item() * len(x)
        hist["train_recon"].append(recon_sum / len(train_set))
        hist["train_kl"].append(kl_sum / len(train_set))

        val_recon, val_kl = evaluate(model, val_set)  # (3)!
        hist["val_recon"].append(val_recon)
        hist["val_kl"].append(val_kl)
        print(f"  d={d:2d} época {epoch + 1:2d}: treino recon {hist['train_recon'][-1]:7.2f} "
              f"KL {hist['train_kl'][-1]:6.2f} | validação recon {val_recon:7.2f} KL {val_kl:6.2f}")
    return model, hist
# --8<-- [end:train]


# --------------------------------------------------------------------------- figuras
def show_row(axes, images, titles=None):
    """Desenha imagens de 784 pixels numa linha de eixos, sem marcas nos eixos."""
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if titles is not None:
            ax.set_title(titles[i], fontsize=8)


def fig1_loss_curves(hist, path):
    epochs = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, term, name in ((axes[0], "recon", "Reconstrução (BCE)"), (axes[1], "kl", "KL")):
        ax.plot(epochs, hist[f"train_{term}"], "o-", ms=3, label="Treino")
        ax.plot(epochs, hist[f"val_{term}"], "s--", ms=3, label="Validação")
        ax.set_title(f"{name} por época (d = 16)")
        ax.set_xlabel("Época")
        ax.set_ylabel("nats por imagem")
        ax.set_xticks(epochs[::2])
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figura 1. Os dois termos da loss, treino e validação")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --8<-- [start:fig2]
@torch.no_grad()
def fig2_reconstructions(model, X_test, y_test, path):
    idx = [int((y_test == k).nonzero()[0]) for k in range(10)]  # primeiro de cada dígito
    x = X_test[idx]
    mu, _ = model.encode(x)
    x_hat = model.decode(mu)                                    # decodifica mu, não z
    fig, axes = plt.subplots(2, 10, figsize=(11, 2.8))
    show_row(axes[0], x, titles=[f"idx {i}" for i in idx])
    show_row(axes[1], x_hat)
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("decode(μ)")
    fig.suptitle("Figura 2. Primeira imagem de teste de cada dígito e sua reconstrução (d = 16)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return idx
# --8<-- [end:fig2]


# --8<-- [start:fig3]
@torch.no_grad()
def fig3_prior_samples(model, path):
    torch.manual_seed(0)
    z = torch.randn(64, model.d)          # z ~ N(0, I), amostrado à mão
    x = model.decode(z)
    grid = x.reshape(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
    fig, ax = plt.subplots(figsize=(6, 6.3))
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Figura 3. 64 amostras do prior z ~ N(0, I), decodificadas (d = 16)")
    ax.set_xlabel("coluna da grade (8 amostras)")
    ax.set_ylabel("linha da grade (8 amostras)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return x
# --8<-- [end:fig3]


# --8<-- [start:fig4]
def fig4_latent_scatter(mu, y, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for k in range(10):
        m = y == k
        ax.scatter(mu[m, 0], mu[m, 1], s=3, alpha=0.6, color=cmap(k), label=str(k))
    ax.set_title("Figura 4. μ das 10.000 imagens de teste no latente de 2 dimensões")
    ax.set_xlabel("μ₁")
    ax.set_ylabel("μ₂")
    ax.grid(alpha=0.3)
    ax.legend(title="Dígito", markerscale=4, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
# --8<-- [end:fig4]


# --8<-- [start:fig5]
@torch.no_grad()
def fig5_latent_grid(model, path, n=15):
    q = np.linspace(0.05, 0.95, n)
    z_vals = norm.ppf(q)                  # Phi^{-1}(q): quantis do N(0, 1)
    z_vals[np.isclose(z_vals, 0)] = 0.0   # Phi^{-1}(0.5) sai -2.8e-17; evita o rótulo "-0.00"
    # linha i da figura (de cima para baixo) usa z2 decrescente, para o eixo crescer para cima
    zz = np.array([[z1, z2] for z2 in z_vals[::-1] for z1 in z_vals], dtype=np.float32)
    x = model.decode(torch.from_numpy(zz))
    grid = x.reshape(n, n, 28, 28).permute(0, 2, 1, 3).reshape(n * 28, n * 28)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ticks = np.arange(n) * 28 + 14
    ax.set_xticks(ticks, [f"{v:.2f}" for v in z_vals], rotation=90, fontsize=7)
    ax.set_yticks(ticks, [f"{v:.2f}" for v in z_vals[::-1]], fontsize=7)
    ax.set_xlabel("z₁ = Φ⁻¹(q),  q de 0.05 a 0.95")
    ax.set_ylabel("z₂ = Φ⁻¹(q),  q de 0.05 a 0.95")
    ax.set_title("Figura 5. Plano latente decodificado numa grade 15 × 15 de quantis (d = 2)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return z_vals
# --8<-- [end:fig5]


# --8<-- [start:fig6]
IDX_A, IDX_B = 4, 2   # primeiro "4" e primeiro "1" do conjunto de teste


@torch.no_grad()
def fig6_interpolation(model, X_test, y_test, path):
    t = torch.linspace(0, 1, 10).unsqueeze(1)
    xa, xb = X_test[IDX_A], X_test[IDX_B]
    pixel = (1 - t) * xa + t * xb                     # interpolação no espaço de pixels
    mu, _ = model.encode(torch.stack([xa, xb]))
    latent = model.decode((1 - t) * mu[0] + t * mu[1])  # interpolação em mu (d = 16)
    fig, axes = plt.subplots(2, 10, figsize=(11, 2.9))
    show_row(axes[0], pixel, titles=[f"t={v:.2f}" for v in t.squeeze(1).tolist()])
    show_row(axes[1], latent)
    axes[0, 0].set_ylabel("pixels")
    axes[1, 0].set_ylabel("latente")
    fig.suptitle(f"Figura 6. De um {int(y_test[IDX_A])} (idx {IDX_A}) para um "
                 f"{int(y_test[IDX_B])} (idx {IDX_B}): pixels (cima) x latente d = 16 (baixo)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pixel, latent
# --8<-- [end:fig6]


def grey_fraction(x, lo=0.1, hi=0.9):
    """Fração de pixels 'cinzas' (nem fundo nem traço): mede quão borrada/translúcida é a imagem."""
    return ((x > lo) & (x < hi)).float().mean().item()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows (cp1252) não imprime σ, μ
    FIGURES.mkdir(exist_ok=True)
    print(f"Split: treino {len(train_set)}, validação {len(val_set)} "
          f"(primeiros índices de treino: {train_set.indices[:5]})")

    # ---------------------------------------------------------------- Exercício 1
    print("\n== Exercício 1: d = 16 ==")
    vae16, hist16 = train_vae(16)
    fig1_loss_curves(hist16, FIGURES / "fig01-loss-curves.png")

    # ---------------------------------------------------------------- Exercício 2
    print("\n== Exercício 2: d = 2 ==")
    vae2, hist2 = train_vae(2)

    # -------------------------------- só agora o conjunto de teste entra em cena
    X_test, y_test = load_mnist(train=False)
    test_set = TensorDataset(X_test, y_test)
    torch.manual_seed(SEED)
    te16 = evaluate(vae16, test_set)
    torch.manual_seed(SEED)
    te2 = evaluate(vae2, test_set)

    idx_digits = fig2_reconstructions(vae16, X_test, y_test, FIGURES / "fig02-reconstructions.png")
    samples = fig3_prior_samples(vae16, FIGURES / "fig03-prior-samples.png")
    with torch.no_grad():
        mu2, _ = vae2.encode(X_test)
        mu2_tr, _ = vae2.encode(X_full[train_set.indices])
    fig4_latent_scatter(mu2.numpy(), y_test.numpy(), FIGURES / "fig04-latent-scatter.png")
    z_vals = fig5_latent_grid(vae2, FIGURES / "fig05-latent-grid.png")
    pixel, latent = fig6_interpolation(vae16, X_test, y_test, FIGURES / "fig06-interpolation.png")

    print("\n== Números de teste (nats por imagem) ==")
    for d, (r, k) in ((16, te16), (2, te2)):
        print(f"d={d:2d}: reconstrução {r:.2f}, KL {k:.2f}, -ELBO {r + k:.2f}")
    print(f"d=16: KL de validação época 1 = {hist16['val_kl'][0]:.2f}, "
          f"época {EPOCHS} = {hist16['val_kl'][-1]:.2f}")
    print(f"d=16: reconstrução de validação época 1 = {hist16['val_recon'][0]:.2f}, "
          f"época {EPOCHS} = {hist16['val_recon'][-1]:.2f}")
    print(f"d= 2: KL de validação época 1 = {hist2['val_kl'][0]:.2f}, "
          f"época {EPOCHS} = {hist2['val_kl'][-1]:.2f}")

    # ------------------------------------------- números de apoio para as análises
    print("\n== Apoio às análises ==")
    with torch.no_grad():
        mu16, logvar16 = vae16.encode(X_test)
        sigma16 = torch.exp(0.5 * logvar16)
        # 1E.3: decodificar mu x decodificar um z amostrado
        rec_mu = F.binary_cross_entropy(vae16.decode(mu16), X_test, reduction="sum").item() / len(X_test)
        torch.manual_seed(SEED)
        rec_z = F.binary_cross_entropy(vae16.decode(reparameterize(mu16, logvar16)), X_test,
                                       reduction="sum").item() / len(X_test)
        # 1E.2a: posterior agregada x prior, por dimensão
        kl_dim = 0.5 * (mu16.pow(2) + logvar16.exp() - 1 - logvar16).mean(0)
        agg_var = mu16.var(0) + sigma16.pow(2).mean(0)          # variância da mistura por dimensão
    print(f"σ médio do encoder (d=16, teste): {sigma16.mean():.3f}  (prior: 1)")
    print(f"Reconstrução de teste decodificando μ: {rec_mu:.2f} | decodificando z amostrado: {rec_z:.2f}")
    print(f"KL por dimensão (d=16): {np.round(kl_dim.numpy(), 2)}")
    print(f"Dimensões ativas (KL > 0.1 nat): {(kl_dim > 0.1).sum().item()} de 16")
    print(f"Variância da posterior agregada por dimensão: {np.round(agg_var.numpy(), 2)}")
    print(f"  mín {agg_var.min():.2f}, máx {agg_var.max():.2f}")
    with torch.no_grad():
        rec16 = vae16.decode(mu16)
    print(f"Fração de pixels cinzas (0.1-0.9): teste real {grey_fraction(X_test):.3f}, "
          f"reconstruções {grey_fraction(rec16):.3f}, amostras do prior {grey_fraction(samples):.3f}")

    # 2E.1: sobreposição entre dígitos no plano de mu (k-NN treinado no mu de treino)
    knn = KNeighborsClassifier(n_neighbors=15).fit(mu2_tr.numpy(), y_full[train_set.indices].numpy())
    pred = knn.predict(mu2.numpy())
    cm = confusion_matrix(y_test.numpy(), pred)
    print(f"k-NN (k=15) sobre μ de d=2, acurácia de teste: {(pred == y_test.numpy()).mean():.2%}")
    pairs = sorted(((cm[i, j] + cm[j, i], i, j) for i in range(10) for j in range(i + 1, 10)),
                   reverse=True)
    print("Pares mais confundidos:", [(f"{i}-{j}", int(n)) for n, i, j in pairs[:6]])
    print(f"Matriz de confusão (linhas = verdadeira):\n{cm}")
    centers = np.array([mu2.numpy()[y_test.numpy() == k].mean(0) for k in range(10)])
    print("Centróide de μ por dígito (d=2):")
    for k in range(10):
        print(f"  {k}: ({centers[k, 0]:+.2f}, {centers[k, 1]:+.2f})")
    print(f"Valores de z da grade 15x15: {np.round(z_vals, 2)}")

    # 2D / 2E.3
    print(f"Interpolação: idx {IDX_A} (dígito {int(y_test[IDX_A])}) -> "
          f"idx {IDX_B} (dígito {int(y_test[IDX_B])})")
    print("Fração de pixels cinzas ao longo de t, pixels: "
          f"{[round(grey_fraction(p), 3) for p in pixel]}")
    print("Fração de pixels cinzas ao longo de t, latente: "
          f"{[round(grey_fraction(p), 3) for p in latent]}")
    print(f"Primeiro índice de teste de cada dígito: {idx_digits}")
