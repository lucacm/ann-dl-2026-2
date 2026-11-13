"""Projeto, entrega 2 (Regressão): MLP em PyTorch para o preço dos carros CarDekho.

Gera as figuras de ``figures/`` e imprime todos os números citados no relatório, na
ordem dos passos do enunciado (0 ... 7). Dados, split e pré-processamento vêm, sem
alteração, do módulo da Entrega 1 (``docs/projects/eda/code/cardekho.py``).

Uso (a partir da raiz do repositório, com os CSVs em docs/projects/eda/code/data/):

    python docs/projects/regression/code/regression.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eda" / "code"))

from cardekho import (CATEGORICAL, NUMERIC, SEED, build_preprocess,  # noqa: E402
                      load_dataset, split)

FIGURES = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)

torch.set_num_threads(1)  # resultados idênticos entre execuções na CPU

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "font.size": 9, "axes.axisbelow": True,
})
COLORS = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a55c9"]


def plain_log(ax, axis="both"):
    """Eixo log com rótulos legíveis (100.000, 200.000, ...) e sem rótulos menores."""
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", "."))
    for name in (["x", "y"] if axis == "both" else [axis]):
        ax_ = getattr(ax, f"{name}axis")
        ax_.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax_.set_major_formatter(fmt)
        ax_.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"  -> figures/{name}")


def header(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def rupees(v):
    return f"{v:,.0f} ₹".replace(",", ".")


# =============================================================================
# 0. Ponto de partida
# =============================================================================
header("0. Ponto de partida")

# --8<-- [start:setup]
# Mesmos dados, mesmo split e mesma seed da Entrega 1. O alvo já vem em log(preço).
df = load_dataset()
X_train_full, X_test, y_train_full, y_test, src_train_full, src_test = split(df)

# Validação: 20% do TREINO (o teste fica guardado até o passo 5).
X_tr, X_val, y_tr, y_val, src_tr, src_val = train_test_split(
    X_train_full, y_train_full, src_train_full,
    test_size=0.2, random_state=SEED, stratify=src_train_full)

# Pipeline do EDA: aprende medianas, quantis, médias e categorias SÓ no treino.
prep = build_preprocess().fit(X_tr)
Z_tr, Z_val, Z_test = (prep.transform(X) for X in (X_tr, X_val, X_test))
# --8<-- [end:setup]

print(f"dataset: {df.shape[0]} linhas | treino EDA {len(X_train_full)} | teste {len(X_test)}")
print(f"treino {len(X_tr)} | validação {len(X_val)} | teste {len(X_test)}")
print(f"features após o pipeline: {Z_tr.shape[1]}")
print(f"média de Z_tr (deve ser ~0 nas padronizadas): "
      f"{Z_tr[:, 1:9].mean():.4f} | média de Z_val: {Z_val[:, 1:9].mean():.4f}")


# --8<-- [start:metrics]
def metrics(y_log, pred_log):
    """Métricas na escala ORIGINAL (rúpias): desfaz o log antes de comparar."""
    y, p = np.exp(np.asarray(y_log)), np.exp(np.asarray(pred_log))
    err = y - p
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE": np.mean(np.abs(err) / y) * 100,
        "R2": 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2),
    }


CRITERION = 0.15  # a rede precisa de MAE de teste >= 15% menor que o melhor baseline
# --8<-- [end:metrics]


def fmt_metrics(m):
    return (f"MAE {rupees(m['MAE'])} | RMSE {rupees(m['RMSE'])} | "
            f"MAPE {m['MAPE']:.1f}% | R² {m['R2']:.3f}")


print(f"preço mínimo do dataset: {rupees(np.exp(df['price'].pipe(np.log)).min())}"
      " (longe de zero: MAPE é seguro)")

# =============================================================================
# 1. Baselines
# =============================================================================
header("1. Baselines (validação)")

# --8<-- [start:baselines]
baselines = {
    "Dummy (mediana)": DummyRegressor(strategy="median"),
    "Ridge (alpha=1)": Ridge(alpha=1.0),
}
for name, model in baselines.items():
    model.fit(Z_tr, y_tr)  # mesmo pipeline, mesmo treino da rede
# --8<-- [end:baselines]

baseline_val = {}
for name, model in baselines.items():
    baseline_val[name] = metrics(y_val, model.predict(Z_val))
    print(f"{name:16s} {fmt_metrics(baseline_val[name])}")
best_baseline = min(baseline_val, key=lambda k: baseline_val[k]["MAE"])
print(f"melhor baseline na validação: {best_baseline}")
print(f"preço previsto pelo Dummy: {rupees(np.exp(baselines['Dummy (mediana)'].predict(Z_val[:1])[0]))}")

# =============================================================================
# 2. A rede
# =============================================================================
header("2. A rede")


# --8<-- [start:model]
def make_mlp(n_in, hidden=(64, 32), dropout=0.0):
    """MLP: [Linear -> ReLU -> Dropout] por camada oculta, e 1 saída linear."""
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = h
    layers.append(nn.Linear(width, 1))  # saída linear: log(preço) é um número real
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # He, feita para ReLU
            nn.init.zeros_(m.bias)
    return net
# --8<-- [end:model]


def n_params(net):
    return sum(p.numel() for p in net.parameters())


torch.manual_seed(SEED)
base_net = make_mlp(Z_tr.shape[1])
print(base_net)
print(f"parâmetros: {n_params(base_net)} | linhas de treino: {len(Z_tr)} | "
      f"razão {n_params(base_net) / len(Z_tr):.2f}")

# =============================================================================
# 3. Treino
# =============================================================================
header("3. Treino")

T = {k: torch.tensor(np.asarray(v), dtype=torch.float32)
     for k, v in {"Ztr": Z_tr, "Zval": Z_val, "Ztest": Z_test,
                  "ytr": y_tr, "yval": y_val}.items()}


# --8<-- [start:train]
def train(hidden=(64, 32), dropout=0.0, weight_decay=0.0, seed=SEED,
          lr=1e-3, batch_size=64, max_epochs=300, patience=20):
    """Treina com AdamW + MSE em log(preço); early stopping na perda de validação."""
    torch.manual_seed(seed)
    net = make_mlp(T["Ztr"].shape[1], hidden, dropout)
    # A saída começa na média do alvo (~13 em log); sem isso a rede gasta dezenas de
    # épocas só subindo o bias de 0 até 13.
    nn.init.constant_(net[-1].bias, float(T["ytr"].mean()))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)

    history = {"train": [], "val": []}
    best_val, best_state, best_epoch, waited = np.inf, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        net.train()  # liga o dropout
        order = torch.randperm(len(T["Ztr"]), generator=gen)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(T["Ztr"][idx]).squeeze(1), T["ytr"][idx])
            loss.backward()
            opt.step()

        net.eval()  # desliga o dropout para medir
        with torch.no_grad():
            tr = loss_fn(net(T["Ztr"]).squeeze(1), T["ytr"]).item()
            va = loss_fn(net(T["Zval"]).squeeze(1), T["yval"]).item()
        history["train"].append(tr)
        history["val"].append(va)

        if va < best_val:  # melhorou: guarda uma cópia destes pesos
            best_val, best_epoch, waited = va, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:  # 20 épocas sem melhorar: para
                break
    net.load_state_dict(best_state)  # volta para a melhor época
    net.eval()
    return net, history, best_epoch


def predict(net, Z):
    with torch.no_grad():
        return net(Z).squeeze(1).numpy()
# --8<-- [end:train]


t0 = time.time()
net0, hist0, best0 = train()
elapsed = time.time() - t0
n_ep = len(hist0["val"])
print(f"config base (64,32), sem regularização: {n_ep} épocas em {elapsed:.1f}s "
      f"({elapsed / n_ep * 1000:.0f} ms/época), melhor época {best0}")
print(f"MSE log na melhor época: treino {hist0['train'][best0 - 1]:.4f} | "
      f"validação {hist0['val'][best0 - 1]:.4f}")
print(f"MSE log na última época: treino {hist0['train'][-1]:.4f} | "
      f"validação {hist0['val'][-1]:.4f}")
print(f"MSE log na época 1: treino {hist0['train'][0]:.4f} | validação {hist0['val'][0]:.4f}")
for ep in (5, 10, 20, 50):
    if ep <= n_ep:
        print(f"  época {ep}: treino {hist0['train'][ep - 1]:.4f} | validação {hist0['val'][ep - 1]:.4f}")
print(f"base na validação: {fmt_metrics(metrics(y_val, predict(net0, T['Zval'])))}")

fig, ax = plt.subplots(figsize=(7, 4))
epochs = np.arange(1, n_ep + 1)
ax.plot(epochs, hist0["train"], color=COLORS[0], label="treino")
ax.plot(epochs, hist0["val"], color=COLORS[1], label="validação")
ax.axvline(best0, color="gray", ls=":", label=f"melhor época ({best0})")
ax.set_yscale("log")
ax.set(title="Figura 1 — Perda por época, configuração base (64-32, sem regularização)",
       xlabel="época", ylabel="MSE em log(preço) (escala log)")
ax.legend()
save(fig, "fig01-curves.png")

# =============================================================================
# 4. Regularização e busca
# =============================================================================
header("4. Regularização e busca de hiperparâmetros (validação)")

# --8<-- [start:search]
GRID = [dict(hidden=h, dropout=d, weight_decay=wd)
        for h in [(32, 16), (64, 32), (128, 64)]
        for d in [0.0, 0.2]
        for wd in [0.0, 1e-3]]  # 3 x 2 x 2 = 12 configurações

results, histories = [], {}
GRID_VALUE = lambda v: v if isinstance(v, tuple) else float(v)  # noqa: E731
for cfg in GRID:
    net, hist, best_ep = train(**cfg)
    m = metrics(y_val, predict(net, T["Zval"]))
    results.append({**cfg, **m, "best_epoch": best_ep, "params": n_params(net)})
    histories[(cfg["hidden"], cfg["dropout"], cfg["weight_decay"])] = hist

table = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
best_cfg = {k: GRID_VALUE(table.loc[0, k]) for k in ("hidden", "dropout", "weight_decay")}
# --8<-- [end:search]

with pd.option_context("display.width", 200, "display.max_columns", 20):
    show = table.copy()
    show["MAE"] = show["MAE"].round(0)
    show["RMSE"] = show["RMSE"].round(0)
    show["MAPE"] = show["MAPE"].round(2)
    show["R2"] = show["R2"].round(4)
    print(show.to_string())
print(f"escolhida: {best_cfg}")

print("\nEfeito da regularização (MAE de validação, média sobre as 3 larguras):")
for (d, wd), g in table.groupby(["dropout", "weight_decay"]):
    print(f"  dropout {d} | weight decay {wd}: MAE {rupees(g['MAE'].mean())} | "
          f"MAPE {g['MAPE'].mean():.2f}% | época de parada média {g['best_epoch'].mean():.0f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
variants = [((64, 32), 0.0, 0.0, "sem regularização"),
            ((64, 32), 0.0, 1e-3, "weight decay 1e-3"),
            ((64, 32), 0.2, 0.0, "dropout 0,2"),
            ((64, 32), 0.2, 1e-3, "dropout 0,2 + weight decay")]
for (h, d, wd, label), c in zip(variants, COLORS):
    hist = histories[(h, d, wd)]
    ep = np.arange(1, len(hist["val"]) + 1)
    axes[0].plot(ep, hist["val"], color=c, label=label)
    gap = np.array(hist["val"]) - np.array(hist["train"])
    axes[1].plot(ep, gap, color=c, label=label)
axes[0].set_yscale("log")
axes[0].set(title="Perda de validação", xlabel="época", ylabel="MSE em log(preço) (escala log)")
axes[1].axhline(0, color="gray", lw=0.8)
axes[1].set(title="Distância validação − treino", xlabel="época", ylabel="Δ MSE em log(preço)")
axes[0].legend()
fig.suptitle("Figura 2 — Regularização na rede 64-32", fontweight="bold")
save(fig, "fig02-regularization.png")

# =============================================================================
# 5. Avaliação no teste (única vez em que o teste é usado)
# =============================================================================
header("5. Avaliação no teste")

# --8<-- [start:test]
SEEDS = [42, 43, 44]
runs = []
for s in SEEDS:
    net, hist, best_ep = train(**best_cfg, seed=s)
    runs.append({
        "seed": s, "net": net, "best_epoch": best_ep,
        "train": metrics(y_tr, predict(net, T["Ztr"])),
        "val": metrics(y_val, predict(net, T["Zval"])),
        "test": metrics(y_test, predict(net, T["Ztest"])),  # o teste, pela 1ª vez
    })

baseline_test = {name: metrics(y_test, model.predict(Z_test))
                 for name, model in baselines.items()}
# --8<-- [end:test]

for r in runs:
    print(f"seed {r['seed']}: parou na época {r['best_epoch']} | teste {fmt_metrics(r['test'])}")


def mean_std(split_name, key):
    vals = np.array([r[split_name][key] for r in runs])
    return vals.mean(), vals.std(ddof=1)


print("\nRede, média ± desvio (3 seeds):")
for split_name in ("train", "val", "test"):
    parts = []
    for key in ("MAE", "RMSE", "MAPE", "R2"):
        mu, sd = mean_std(split_name, key)
        parts.append(f"{key} {mu:,.4f} ± {sd:,.4f}" if key == "R2" else
                     f"{key} {mu:,.2f} ± {sd:,.2f}")
    print(f"  {split_name:5s} " + " | ".join(parts))

print("\nBaselines no teste:")
for name, m in baseline_test.items():
    print(f"  {name:16s} {fmt_metrics(m)}")

net_mae, net_mae_sd = mean_std("test", "MAE")
ref = baseline_test[best_baseline]["MAE"]
gain = 1 - net_mae / ref
print(f"\nCritério: MAE da rede {rupees(net_mae)} vs {best_baseline} {rupees(ref)} -> "
      f"redução de {gain * 100:.1f}% (meta >= {CRITERION * 100:.0f}%): "
      f"{'ATINGIDO' if gain >= CRITERION else 'NÃO ATINGIDO'}")
print(f"redução de MAE em relação ao Dummy: "
      f"{(1 - net_mae / baseline_test['Dummy (mediana)']['MAE']) * 100:.1f}%")

# Figuras e análise de erro usam a rede da seed 42.
net = runs[0]["net"]
pred_log = predict(net, T["Ztest"])
y_true, y_pred = np.exp(y_test.to_numpy()), np.exp(pred_log)
resid = y_true - y_pred

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for src, c in zip(["v3", "v4"], COLORS):
    mask = (src_test == src).to_numpy()
    ax.scatter(y_true[mask], y_pred[mask], s=8, alpha=0.5, color=c, label=f"origem {src}")
lims = [min(y_true.min(), y_pred.min()) * 0.8, max(y_true.max(), y_pred.max()) * 1.2]
ax.plot(lims, lims, color="black", lw=1, label="previsão perfeita (y = x)")
ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
       title="Figura 3 — Previsto × real (teste)",
       xlabel="preço real (₹, escala log)", ylabel="preço previsto (₹, escala log)")
plain_log(ax)
ax.tick_params(axis="x", rotation=45)
ax.legend()
save(fig, "fig03-pred-vs-actual.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_pred, resid, s=8, alpha=0.5, color=COLORS[0], label="carro do teste")
axes[0].axhline(0, color="black", lw=1, label="resíduo zero")
axes[0].set(xscale="log", title="Resíduo em rúpias",
            xlabel="preço previsto (₹, escala log)", ylabel="real − previsto (₹)")
plain_log(axes[0], "x")
axes[0].tick_params(axis="x", rotation=45)
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
axes[0].legend()
axes[1].scatter(y_pred, y_test.to_numpy() - pred_log, s=8, alpha=0.5, color=COLORS[1],
                label="carro do teste")
axes[1].axhline(0, color="black", lw=1, label="resíduo zero")
axes[1].set(xscale="log", title="Resíduo em log (erro relativo)",
            xlabel="preço previsto (₹, escala log)", ylabel="log(real) − log(previsto)")
plain_log(axes[1], "x")
axes[1].tick_params(axis="x", rotation=45)
axes[1].legend()
fig.suptitle("Figura 4 — Resíduos × previsto (teste)", fontweight="bold")
save(fig, "fig04-residuals.png")

r_log = y_test.to_numpy() - pred_log
print(f"resíduo em log: média {r_log.mean():.4f} | desvio {r_log.std():.4f}")
for lo, hi in [(0, 3e5), (3e5, 1e6), (1e6, 1e7), (1e7, 1e9)]:
    m = (y_pred >= lo) & (y_pred < hi)
    print(f"  previsto em [{rupees(lo)}, {rupees(hi)}): n={m.sum():4d} | "
          f"desvio do resíduo ₹ {rupees(resid[m].std())} | desvio do resíduo log {r_log[m].std():.3f}")

# =============================================================================
# 6. Análise de erro
# =============================================================================
header("6. Análise de erro (teste, sem escolher nada a partir daqui)")

# --8<-- [start:errors]
err = X_test.copy()
err["source"] = src_test
err["real"], err["previsto"] = y_true, y_pred
err["erro_abs"] = np.abs(resid)
err["erro_rel_%"] = np.abs(resid) / y_true * 100
err["faixa_preco"] = pd.qcut(y_true, 5, labels=["Q1 (barato)", "Q2", "Q3", "Q4", "Q5 (caro)"])


def by_group(col):
    return err.groupby(col, observed=True).agg(
        n=("real", "size"), MAE=("erro_abs", "mean"), MAPE=("erro_rel_%", "mean"))
# --8<-- [end:errors]


groups = {"faixa_preco": "Faixa de preço (quintil)", "source": "Origem",
          "transmission": "Câmbio", "fuel": "Combustível"}
group_tables = {}
for col, title in groups.items():
    group_tables[col] = by_group(col)
    print(f"\n{title}:")
    print(group_tables[col].round(1).to_string())
print("\nlimites dos quintis de preço (₹):",
      [rupees(v) for v in np.quantile(y_true, [0, .2, .4, .6, .8, 1])])

age = pd.cut(err["year"], [1990, 2010, 2015, 2018, 2020, 2023],
             labels=["≤2010", "2011–15", "2016–18", "2019–20", "2021–22"])
print("\nAno de fabricação:")
print(err.groupby(age, observed=True).agg(n=("real", "size"), MAE=("erro_abs", "mean"),
                                          MAPE=("erro_rel_%", "mean")).round(1).to_string())
print("\nowner = 0 vs resto:")
print(err.groupby(err["owner"] == 0).agg(n=("real", "size"), MAE=("erro_abs", "mean"),
                                         MAPE=("erro_rel_%", "mean")).round(1).to_string())

worst_group = max(((col, idx, row["MAPE"]) for col, t in group_tables.items()
                   for idx, row in t.iterrows() if row["n"] >= 20), key=lambda x: x[2])
print(f"\nsubgrupo (n>=20) com maior MAPE: {worst_group[0]} = {worst_group[1]}, "
      f"MAPE {worst_group[2]:.1f}%")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
for ax, (col, title) in zip(axes, groups.items()):
    t = group_tables[col]
    bars = ax.bar(t.index.astype(str), t["MAPE"], color=COLORS[0], label="MAPE")
    for b, n in zip(bars, t["n"]):
        ax.annotate(f"n={n}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
    ax.set(title=title, xlabel="")
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("erro percentual médio (MAPE, %)")
axes[0].legend()
fig.suptitle("Figura 5 — Erro relativo por subgrupo (teste, rede seed 42)", fontweight="bold")
save(fig, "fig05-error-by-group.png")

cols = ["brand", "year", "km", "fuel", "transmission", "power_bhp", "owner", "source",
        "real", "previsto", "erro_rel_%"]
worst = err.sort_values("erro_rel_%", ascending=False).head(10)[cols]
print("\n10 piores (erro relativo):")
print(worst.round(1).to_string())
print("\n10 piores (erro absoluto ₹):")
print(err.sort_values("erro_abs", ascending=False).head(10)[cols].round(1).to_string())

# --8<-- [start:permutation]
rng = np.random.default_rng(SEED)
base_mae = metrics(y_test, pred_log)["MAE"]
importance = {}
for col in NUMERIC + CATEGORICAL:  # as 11 colunas ORIGINAIS, antes do pipeline
    increases = []
    for _ in range(5):
        X_perm = X_test.copy()
        X_perm[col] = rng.permutation(X_perm[col].to_numpy())  # quebra a relação col-preço
        Z_perm = torch.tensor(prep.transform(X_perm), dtype=torch.float32)
        increases.append(metrics(y_test, predict(net, Z_perm))["MAE"] - base_mae)
    importance[col] = (np.mean(increases), np.std(increases))
# --8<-- [end:permutation]

imp = pd.DataFrame(importance, index=["aumento_MAE", "desvio"]).T.sort_values("aumento_MAE")
print(f"\nImportância por permutação (MAE base {rupees(base_mae)}):")
print(imp.iloc[::-1].round(0).to_string())

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.barh(imp.index, imp["aumento_MAE"], xerr=imp["desvio"], color=COLORS[0],
        label="média de 5 permutações (± desvio)")
ax.set(title="Figura 6 — Importância por permutação (teste)",
       xlabel="aumento do MAE ao embaralhar a coluna (₹)", ylabel="feature")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
ax.legend(loc="lower right")
save(fig, "fig06-permutation.png")

# =============================================================================
# 7. Resumo
# =============================================================================
header("7. Results summary")
h = best_cfg["hidden"]
mape_mu, mape_sd = mean_std("test", "MAPE")
r2_mu, r2_sd = mean_std("test", "R2")
print(f"1. CarDekho v3+v4 | regressão | price (₹), treinado em log | métrica principal MAE ₹")
print(f"2. treino {len(X_tr)} | validação {len(X_val)} | teste {len(X_test)}")
print(f"3. melhor baseline {best_baseline}: MAE teste {rupees(ref)}")
print(f"4. {Z_tr.shape[1]} → {' → '.join(map(str, h))} → 1 | "
      f"{n_params(runs[0]['net'])} parâmetros | "
      f"dropout {best_cfg['dropout']} | weight decay {best_cfg['weight_decay']}")
print(f"5. AdamW lr 1e-3, batch 64, épocas de parada {[r['best_epoch'] for r in runs]}")
print(f"6. MAE teste {rupees(net_mae)} ± {rupees(net_mae_sd)} | MAPE {mape_mu:.1f}% ± {mape_sd:.1f} "
      f"| R² {r2_mu:.3f} ± {r2_sd:.3f}")
print(f"7. critério {'atingido' if gain >= CRITERION else 'não atingido'} ({gain * 100:.1f}%)")
print(f"8. maior erro: {worst_group[0]} = {worst_group[1]}, MAPE {worst_group[2]:.1f}% "
      f"(MAE {rupees(group_tables[worst_group[0]].loc[worst_group[1], 'MAE'])})")
