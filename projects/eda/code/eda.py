"""Projeto, entrega 1 (EDA): análise exploratória do dataset CarDekho (v3 + v4).

Gera as figuras de ``figures/`` e imprime todos os números citados no relatório, na
ordem dos itens do enunciado (1A ... 5). A harmonização, o split e o pipeline vêm de
``cardekho.py``.

Uso (a partir da raiz do repositório, com os CSVs em docs/projects/eda/code/data/):

    python docs/projects/eda/code/eda.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardekho import (CATEGORICAL, NUMERIC, SEED, build_preprocess,  # noqa: E402
                      deduplicate, harmonize, load_dataset, load_raw, split)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# --8<-- [start:style]
RNG = np.random.default_rng(SEED)  # única fonte de aleatoriedade fora do sklearn/UMAP

SOURCE_COLORS = {"v3": "#2a78d6", "v4": "#eb6834"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#cfe0f5", "#2a78d6", "#0d2f5c"])
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "font.size": 9, "axes.axisbelow": True,
})
# --8<-- [end:style]


def plain_log(ax, axis="both"):
    """Eixo log com rótulos legíveis (100, 200, 500...) e sem rótulos menores."""
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


# =============================================================================
# 1A. Seleção das fontes e dicionário
# =============================================================================
header("1A. Seleção das fontes")

# --8<-- [start:sources]
PRICE = {"car data": ("Selling_Price", 1e5),  # car data vem em lakhs
         "FROM DEKHO": ("selling_price", 1), "v3": ("selling_price", 1),
         "v4": ("Price", 1)}
YEAR = {"car data": "Year", "FROM DEKHO": "year", "v3": "year", "v4": "Year"}
BANDS = [0, 3e5, 6e5, 1e6, 2e6, np.inf]
BAND_LABELS = ["< 3 lakh", "3–6 lakh", "6–10 lakh", "10–20 lakh", "> 20 lakh"]

raw = {k: load_raw(k) for k in PRICE}
profile, coverage = {}, {}
for k, df in raw.items():
    col, mult = PRICE[k]
    p = df[col] * mult
    profile[k] = {
        "linhas": len(df), "colunas": df.shape[1],
        "duplicatas": int(df.duplicated().sum()),
        "linhas c/ ausente": int(df.isna().any(axis=1).sum()),
        "p10": p.quantile(0.10), "mediana": p.median(),
        "média": p.mean(), "p90": p.quantile(0.90),
        "anos": f"{df[YEAR[k]].min()}–{df[YEAR[k]].max()}",
    }
    coverage[k] = (pd.cut(p, BANDS, labels=BAND_LABELS)
                   .value_counts(normalize=True).reindex(BAND_LABELS) * 100)
profile = pd.DataFrame(profile)
coverage = pd.DataFrame(coverage).T
# --8<-- [end:sources]
print(profile.to_string())
print("\nCobertura de preço (% das linhas de cada fonte):")
print(coverage.round(1).to_string())
print(f"\n% acima de 10 lakh: v3 {coverage.loc['v3', BAND_LABELS[3:]].sum():.1f}%  "
      f"v4 {coverage.loc['v4', BAND_LABELS[3:]].sum():.1f}%")

fig, ax = plt.subplots(figsize=(8, 3.2))
left = np.zeros(len(coverage))
band_colors = SEQ(np.linspace(0.05, 1, len(BAND_LABELS)))
for band, color in zip(BAND_LABELS, band_colors):
    vals = coverage[band].to_numpy()
    ax.barh(coverage.index, vals, left=left, color=color, label=band,
            edgecolor="white", linewidth=2)
    for y, (l, v) in enumerate(zip(left, vals)):
        if v >= 6:
            ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if band in BAND_LABELS[2:] else "#0b0b0b")
    left += vals
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("% dos anúncios da fonte")
ax.set_ylabel("Arquivo")
ax.set_title("Figura 1. Cobertura de faixas de preço por arquivo")
ax.legend(title="Faixa de preço", bbox_to_anchor=(1.01, 1), loc="upper left")
save(fig, "fig01-price-coverage.png")

# Matriz de presença: o mesmo conceito, com o nome que ele tem em cada arquivo.
# --8<-- [start:presence]
CONCEPTS = {
    "preço": ["Selling_Price", "selling_price", "selling_price", "Price"],
    "ano": ["Year", "year", "year", "Year"],
    "km": ["Kms_Driven", "km_driven", "km_driven", "Kilometer"],
    "combustível": ["Fuel_Type", "fuel", "fuel", "Fuel Type"],
    "câmbio": ["Transmission", "transmission", "transmission", "Transmission"],
    "vendedor": ["Seller_Type", "seller_type", "seller_type", "Seller Type"],
    "dono": ["Owner", "owner", "owner", "Owner"],
    "marca": [None, "name", "name", "Make"],
    "motor (cc)": [None, None, "engine", "Engine"],
    "potência": [None, None, "max_power", "Max Power"],
    "torque": [None, None, "torque", "Max Torque"],
    "assentos": [None, None, "seats", "Seating Capacity"],
    "consumo": [None, None, "mileage", None],
    "preço do novo": ["Present_Price", None, None, None],
    "tração": [None, None, None, "Drivetrain"],
    "dimensões (3)": [None, None, None, "Length"],
    "tanque": [None, None, None, "Fuel Tank Capacity"],
    "cidade": [None, None, None, "Location"],
    "cor": [None, None, None, "Color"],
}
presence = pd.DataFrame(
    {k: [np.nan if cols[i] is None else 100 * raw[k][cols[i]].notna().mean()
         for cols in CONCEPTS.values()] for i, k in enumerate(PRICE)},
    index=list(CONCEPTS))
common_all = presence.dropna().index.tolist()
common_v3v4 = presence[["v3", "v4"]].dropna().index.tolist()
# --8<-- [end:presence]
print("\nPresença de colunas (% preenchido; NaN = coluna não existe):")
print(presence.round(1).to_string())
print(f"\nConceitos comuns às 4 fontes ({len(common_all)}): {common_all}")
print("  (marca não entra: car data só traz o modelo, 'ritz', 'sx4')")
print(f"Conceitos comuns a v3 e v4 ({len(common_v3v4)}): {common_v3v4}")

fig, ax = plt.subplots(figsize=(6.2, 6.4))
sns.heatmap(presence, annot=presence.map(lambda v: "" if np.isnan(v) else f"{v:.0f}"),
            fmt="", cmap=SEQ, vmin=90, vmax=100, linewidths=2, linecolor="white",
            cbar_kws={"label": "% de linhas preenchidas"}, ax=ax)
ax.grid(False)  # as bordas brancas das células fazem o papel de grade
ax.set_facecolor("#e4e3df")
ax.set_title("Figura 2. Presença de cada coluna nos 4 arquivos\n"
             "(cinza = a coluna não existe no arquivo)")
ax.set_xlabel("Arquivo")
ax.set_ylabel("Coluna (conceito harmonizado)")
save(fig, "fig02-column-presence.png")

# Conflito FROM DEKHO x v3: o mesmo carro (nome, ano, km) com outro preço.
old, v3raw, v4raw = raw["FROM DEKHO"], raw["v3"], raw["v4"]
key = ["name", "year", "km_driven"]
same_car = old[key].drop_duplicates().merge(v3raw[key].drop_duplicates())
same_price = (old[key + ["selling_price"]].drop_duplicates()
              .merge(v3raw[key + ["selling_price"]].drop_duplicates()))
print(f"\nFROM DEKHO x v3: {len(same_car)} carros (nome, ano, km) em comum, "
      f"{len(same_price)} com o mesmo preço")
print(f"FROM DEKHO: colunas que o v3 não tem = "
      f"{sorted(set(old.columns) - set(v3raw.columns))}")

# Harmonização: quantas linhas cada regra afetou
header("1A. Harmonização v3 + v4")
v3d, v4d = deduplicate(v3raw, "selling_price"), deduplicate(v4raw, "Price")
num3 = pd.to_numeric(v3raw["torque"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
kgm3 = v3raw["torque"].str.contains("kgm", case=False, na=False)
print(f"brand 'Land'->'Land Rover' (v3): {(v3raw['name'].str.split().str[0] == 'Land').sum()}"
      f" | 'Ashok'->'Ashok Leyland' (v3): {(v3raw['name'].str.split().str[0] == 'Ashok').sum()}"
      f" | 'Maruti Suzuki'->'Maruti' (v4): {(v4raw['Make'] == 'Maruti Suzuki').sum()}")
print("fuel v4:", v4raw["Fuel Type"].value_counts().to_dict())
print("owner v3:", v3raw["owner"].value_counts().to_dict())
print("owner v4:", v4raw["Owner"].value_counts().to_dict())
print("seller v3:", v3raw["seller_type"].value_counts().to_dict())
print("seller v4:", v4raw["Seller Type"].value_counts().to_dict())
print(f"torque v3 em kgm convertido: {(kgm3 & (num3 < 100)).sum()} | "
      f"marcado kgm mas já em Nm (>=100): {(kgm3 & (num3 >= 100)).sum()}")
print(f"power '0' (v3): {(pd.to_numeric(v3raw['max_power'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce') == 0).sum()}")
v4_unitless = v4raw["Max Torque"].str.match(r"^\s*[\d.]+\s*@", na=False).sum()
print(f"torque v4 sem unidade (lido como Nm): {v4_unitless}")

df = load_dataset()
print(f"\nDataset final: {df.shape[0]} linhas x {df.shape[1]} colunas "
      f"(origem: {df['source'].value_counts().to_dict()})")
print(f"  {len(NUMERIC)} numéricas {NUMERIC}")
print(f"  {len(CATEGORICAL)} categóricas {CATEGORICAL}  + alvo 'price' + 'source' (só EDA)")

# =============================================================================
# 1B. Qualidade
# =============================================================================
header("1B. Qualidade")

# --8<-- [start:quality]
h_raw = harmonize(v3raw, v4raw)  # antes de deduplicar, para medir a ausência estrutural
mileage_missing = 100 * len(v4raw) / (len(v3raw) + len(v4raw))
v4_only_missing = 100 * len(v3raw) / (len(v3raw) + len(v4raw))

missing = pd.DataFrame({
    "ausentes": df.isna().sum(),
    "%": 100 * df.isna().mean(),
    "% no v3": 100 * df[df.source == "v3"].isna().mean(),
    "% no v4": 100 * df[df.source == "v4"].isna().mean(),
}).query("ausentes > 0")
tech = ["engine_cc", "power_bhp", "torque_nm", "seats"]
pattern = df[tech].isna().value_counts()
# --8<-- [end:quality]
print(f"Ausência estrutural se mantivesse colunas exclusivas: mileage {mileage_missing:.1f}%, "
      f"colunas só do v4 {v4_only_missing:.1f}%")
print(missing.round(2).to_string())
print(f"Linhas com algum ausente: {df.isna().any(axis=1).sum()} "
      f"({100 * df.isna().any(axis=1).mean():.2f}%)")
print("Padrão de ausência (engine, power, torque, seats):")
print(pattern.to_string())
na_rows = df[df[tech].isna().all(axis=1)]
print(f"Linhas sem nenhuma das 4 técnicas: {len(na_rows)}; preço mediano "
      f"{na_rows['price'].median():.0f} vs {df['price'].median():.0f} geral; "
      f"ano mediano {na_rows['year'].median():.0f} vs {df['year'].median():.0f}; "
      f"origem {na_rows['source'].value_counts().to_dict()}")
print(f"Ausentes por marca (top 5): "
      f"{na_rows['brand'].value_counts().head(5).to_dict()}")

v3_dup = int(v3raw.duplicated().sum())
v4_dup = int(v4raw.duplicated().sum())
v3_nd = v3raw.drop_duplicates()
keys3 = [c for c in v3raw.columns if c != "selling_price"]
grp = v3_nd[v3_nd.duplicated(subset=keys3, keep=False)].groupby(keys3, dropna=False)
ratio = grp["selling_price"].max() / grp["selling_price"].min()
keys4 = [c for c in v4raw.columns if c != "Price"]
v4_conf = v4raw[v4raw.duplicated(subset=keys4, keep=False)]
print(f"\nDuplicatas exatas: v3 {v3_dup}, v4 {v4_dup}")
print(f"v3 mesmas colunas, preço diferente: {grp.ngroups} grupos, "
      f"{int(grp.size().sum())} linhas; razão max/min mediana {ratio.median():.3f}, "
      f"máx {ratio.max():.3f}")
print(f"v4 mesmas colunas, preço diferente: {v4_conf.groupby(keys4, dropna=False).ngroups} "
      f"grupos, {len(v4_conf)} linhas")
feat = [c for c in h_raw.columns if c != "source"]
cross = harmonize(v3d, v4d)
cross_dup = cross[cross.duplicated(subset=feat, keep=False)]
print(f"Após harmonizar: {cross.duplicated(subset=feat).sum()} linhas iguais a outra, "
      f"pares entre fontes: "
      f"{int((cross_dup.groupby(feat, dropna=False)['source'].nunique() > 1).sum())}")
print(f"Linhas removidas no total: {len(v3raw) + len(v4raw) - len(df)}")

print(f"\nValores impossíveis: km == 0: {(df.km == 0).sum()}; km > 500 mil: "
      f"{(df.km > 5e5).sum()} (máx {df.km.max():.0f}); "
      f"torque/cc > 0,4 anulado: "
      f"{int((h_raw.torque_nm.isna() & h_raw.engine_cc.notna() & (h_raw.source == 'v3') & v3raw.torque.reindex(h_raw.index).notna()).sum())}")
print(f"seats: {df.seats.value_counts().sort_index().to_dict()}")
print(f"ano: {df.year.min()}–{df.year.max()}; anos > 2020 (só v4): {(df.year > 2020).sum()}")

fig, ax = plt.subplots(figsize=(7, 3.4))
cols = missing.index.tolist()
x = np.arange(len(cols))
for i, src in enumerate(["v3", "v4"]):
    vals = missing[f"% no {src}"].to_numpy()
    bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=SOURCE_COLORS[src],
                  label=f"{src} (n = {(df.source == src).sum()})", edgecolor="white",
                  linewidth=2)
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
ax.set_xticks(x, cols)
ax.set_ylabel("% de linhas ausentes")
ax.set_xlabel("Coluna")
ax.set_title("Figura 3. Ausentes por coluna e origem (dataset final)")
ax.legend(title="Origem")
ax.set_ylim(0, missing[["% no v3", "% no v4"]].max().max() * 1.25)
save(fig, "fig03-missing.png")

# =============================================================================
# 1C. Alvo
# =============================================================================
header("1C. Alvo (price)")
# --8<-- [start:target]
y = df["price"]
logy = np.log(y)
target_stats = {"média": y.mean(), "mediana": y.median(), "dp": y.std(),
                "mín": y.min(), "máx": y.max(), "assimetria": y.skew(),
                "assimetria log": logy.skew(), "curtose": y.kurt()}
# --8<-- [end:target]
for k, v in target_stats.items():
    print(f"  {k:15s} {v:,.3f}")
print("por origem:", df.groupby("source")["price"].agg(["mean", "median"]).round(0)
      .to_dict())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for src in ["v3", "v4"]:
    s = df[df.source == src]
    axes[0].hist(s["price"] / 1e5, bins=np.linspace(0, 60, 61), alpha=0.65,
                 color=SOURCE_COLORS[src], label=src)
    axes[1].hist(np.log10(s["price"]), bins=50, alpha=0.65, color=SOURCE_COLORS[src],
                 label=src)
axes[0].axvline(y.median() / 1e5, color="#0b0b0b", ls="--", lw=1.2,
                label=f"mediana ({y.median() / 1e5:.2f} lakh)")
axes[0].axvline(y.mean() / 1e5, color="#52514e", ls=":", lw=1.5,
                label=f"média ({y.mean() / 1e5:.2f} lakh)")
axes[0].set(title=f"Preço bruto (assimetria {y.skew():.2f}; eixo cortado em 60 lakh)",
            xlabel="Preço (lakh ₹)", ylabel="Número de anúncios")
axes[1].set(title=f"log10(preço) (assimetria {logy.skew():.2f})",
            xlabel="log10(preço em ₹)", ylabel="Número de anúncios")
for a in axes:
    a.legend(title="Origem")
fig.suptitle("Figura 4. Distribuição do alvo, antes e depois do log", fontweight="bold")
fig.tight_layout()
save(fig, "fig04-target.png")
print(f"Anúncios acima de 60 lakh (fora do eixo): {(y > 60e5).sum()}")

# =============================================================================
# 1D. Split
# =============================================================================
header("1D. Split treino/teste")
# --8<-- [start:do_split]
X_train, X_test, y_train, y_test, src_train, src_test = split(df)
# --8<-- [end:do_split]
print(f"treino {X_train.shape}, teste {X_test.shape}")
print("origem treino:", src_train.value_counts(normalize=True).round(4).to_dict())
print("origem teste :", src_test.value_counts(normalize=True).round(4).to_dict())
print(f"log(preço) médio treino {y_train.mean():.4f}, teste {y_test.mean():.4f}")
print(f"brands só no teste: {sorted(set(X_test.brand) - set(X_train.brand))}")
baseline = np.full(len(y_test), y_train.median())
print(f"Baseline (mediana do treino = {np.exp(y_train.median()):.0f} ₹): "
      f"RMSE log {np.sqrt(np.mean((y_test - baseline) ** 2)):.4f}, "
      f"MAE log {np.mean(np.abs(y_test - baseline)):.4f}, "
      f"MAE ₹ {np.mean(np.abs(np.exp(y_test) - np.exp(baseline))):.0f}")

# A partir daqui as figuras usam o dataset inteiro (o EDA pode ver tudo); toda
# estatística APRENDIDA (imputação, escala, PCA...) sai só do treino.
NUM_CONT = ["year", "km", "engine_cc", "power_bhp", "torque_nm", "seats"]

# =============================================================================
# 2A. Univariada numérica
# =============================================================================
header("2A. Numéricas")
# --8<-- [start:describe]
desc = df[NUMERIC + ["price"]].describe(percentiles=[0.25, 0.5, 0.75]).T
desc["assimetria"] = df[NUMERIC + ["price"]].skew()
iqr = desc["75%"] - desc["25%"]
desc["outliers IQR"] = [((df[c] < q1 - 1.5 * i) | (df[c] > q3 + 1.5 * i)).sum()
                        for c, q1, q3, i in zip(desc.index, desc["25%"], desc["75%"], iqr)]
# --8<-- [end:describe]
print(desc.round(2).to_string())

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
units = {"year": "ano", "km": "km rodados", "engine_cc": "cilindrada (cc)",
         "power_bhp": "potência (bhp)", "torque_nm": "torque (Nm)", "seats": "assentos"}
for a, c in zip(axes.flat, NUM_CONT):
    s = df[c].dropna()
    if c == "seats":
        vc = s.value_counts().sort_index()
        a.bar(vc.index.astype(int).astype(str), vc.values, color="#2a78d6",
              edgecolor="white", linewidth=2, label="anúncios")
        a.set_yscale("log")
    else:
        bins = np.linspace(s.min(), s.quantile(0.995), 40)
        a.hist(s.clip(upper=s.quantile(0.995)), bins=bins, color="#2a78d6",
               edgecolor="white", linewidth=0.5, label="anúncios")
        f = (lambda v: f"{v:.0f}") if c == "year" else \
            (lambda v: f"{v:,.0f}".replace(",", "."))
        a.axvline(s.median(), color="#0b0b0b", ls="--", lw=1.2, label=f"mediana {f(s.median())}")
        a.axvline(s.mean(), color="#52514e", ls=":", lw=1.5, label=f"média {f(s.mean())}")
    a.set_title(f"{c} (assimetria {s.skew():.2f})")
    a.set_xlabel(units[c])
    a.set_ylabel("anúncios" + (" (escala log)" if c == "seats" else ""))
    a.legend(fontsize=7)
fig.suptitle("Figura 5. Distribuição das numéricas contínuas (cauda > p99,5 agrupada "
             "no último bin)", fontweight="bold")
fig.tight_layout()
save(fig, "fig05-numeric-hist.png")

LOGF = ["km", "engine_cc", "power_bhp", "torque_nm"]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for a, c in zip(axes, LOGF):
    s = np.log1p(df[c].dropna())
    a.hist(s, bins=40, color="#2a78d6", edgecolor="white", linewidth=0.5,
           label="anúncios")
    a.set_title(f"log1p({c})\nassimetria {df[c].skew():.2f} → {s.skew():.2f}")
    a.set_xlabel(f"log1p({units[c]})")
    a.set_ylabel("anúncios")
    a.legend(fontsize=7)
fig.suptitle("Figura 6. As quatro numéricas assimétricas depois de log1p",
             fontweight="bold")
fig.tight_layout()
save(fig, "fig06-numeric-log.png")
print("assimetria após log1p:", {c: round(np.log1p(df[c].dropna()).skew(), 3) for c in LOGF})
print(f"km < 1000: {(df.km < 1000).sum()}; km > 500 mil: {(df.km > 5e5).sum()}; "
      f"top 3 km: {df.km.nlargest(3).tolist()}")

# =============================================================================
# 2B. Univariada categórica
# =============================================================================
header("2B. Categóricas")
for c in ["fuel", "transmission", "owner", "is_individual", "brand"]:
    vc = df[c].value_counts()
    print(f"{c}: cardinalidade {vc.size}; "
          f"{ {k: f'{v} ({100 * v / len(df):.1f}%)' for k, v in vc.head(6).items()} }")
brand_vc = df["brand"].value_counts()
rare = brand_vc[brand_vc < 20]
print(f"brand: {brand_vc.size} marcas; top 3 = {100 * brand_vc.head(3).sum() / len(df):.1f}% "
      f"das linhas; {rare.size} marcas com < 20 anúncios ({rare.sum()} linhas, "
      f"{100 * rare.sum() / len(df):.2f}%): {rare.index.tolist()}")
brand_src = pd.crosstab(df.brand, df.source)
print(f"marcas só no v3: {brand_src.index[brand_src['v4'] == 0].tolist()}")
print("owner = 0 por origem e marca:",
      df[df.owner == 0].groupby("source")["brand"].value_counts().to_dict())
print(f"seats == 5: {100 * (df.seats == 5).sum() / df.seats.notna().sum():.1f}% das linhas com seats")
print("maior potência:", df.loc[df.power_bhp.idxmax(), ["brand", "power_bhp", "price"]].to_dict())
print(f"marcas só no v4: {brand_src.index[brand_src['v3'] == 0].tolist()}")

fig = plt.figure(figsize=(12, 7))
gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])
ab = fig.add_subplot(gs[:, 0])
bars = brand_vc.sort_values()
ab.barh(bars.index, bars.values,
        color=["#c8c7c1" if v < 20 else "#2a78d6" for v in bars.values], height=0.75)
ab.axvline(20, color="#0b0b0b", ls="--", lw=1.2, label="limiar de raridade (20)")
ab.set_xscale("log")
ab.set(title=f"brand ({brand_vc.size} marcas; cinza = rara)",
       xlabel="anúncios (escala log)", ylabel="marca")
ab.tick_params(axis="y", labelsize=7)
ab.legend(loc="lower right")
for i, (c, lab) in enumerate([("fuel", "combustível"), ("transmission", "câmbio"),
                              ("owner", "nº de donos anteriores (0 = sem dono)")]):
    a = fig.add_subplot(gs[i, 1])
    vc = df[c].value_counts().sort_index() if c == "owner" else df[c].value_counts()
    b = a.bar(vc.index.astype(str), vc.values, color="#2a78d6", edgecolor="white",
              linewidth=2, label="anúncios")
    a.bar_label(b, labels=[f"{100 * v / len(df):.1f}%" for v in vc.values], fontsize=7,
                padding=1)
    a.set(title=c, xlabel=lab, ylabel="anúncios")
    a.set_ylim(0, vc.max() * 1.2)
    a.legend(fontsize=7, loc="upper right")
fig.suptitle("Figura 7. Frequência das categóricas", fontweight="bold")
fig.tight_layout()
save(fig, "fig07-categorical.png")

# =============================================================================
# 3A. Numérica x numérica
# =============================================================================
header("3A. Correlações")
# --8<-- [start:corr]
corr_cols = NUM_CONT + ["owner", "price"]
spear = df[corr_cols].corr(method="spearman")
pear = df[corr_cols].corr(method="pearson")
# --8<-- [end:corr]
print(spear.round(3).to_string())
pairs = (spear.where(np.triu(np.ones(spear.shape, bool), 1)).stack()
         .rename("rho").reset_index())
pairs = pairs[(pairs.level_0 != "price") & (pairs.level_1 != "price")]
pairs = pairs.reindex(pairs.rho.abs().sort_values(ascending=False).index)
print("pares mais correlacionados (sem o alvo):")
print(pairs.head(5).round(3).to_string(index=False))
print("Pearson vs Spearman com o alvo:",
      {c: (round(pear.loc[c, 'price'], 3), round(spear.loc[c, 'price'], 3))
       for c in corr_cols[:-1]})
print("Spearman com log(price) é igual (monótona):",
      round(df[["power_bhp"]].assign(lp=np.log(df.price)).corr("spearman").iloc[0, 1], 3))

# Simpson: a correlação muda dentro dos grupos?
print("\nSpearman com o preço por origem:")
for s, g in df.groupby("source"):
    print(f"  {s}: " + str({c: round(g[[c, 'price']].corr('spearman').iloc[0, 1], 3)
                            for c in ["year", "km", "engine_cc", "power_bhp", "owner"]}))
print("km x preço, geral vs dentro do mesmo ano:")
km_within = []
for yr, g in df.groupby("year"):
    if len(g) >= 50:
        km_within.append((yr, len(g), g[["km", "price"]].corr("spearman").iloc[0, 1]))
km_within = pd.DataFrame(km_within, columns=["ano", "n", "rho"])
print(f"  geral {spear.loc['km', 'price']:.3f}; dentro do ano (média ponderada) "
      f"{np.average(km_within.rho, weights=km_within.n):.3f}; "
      f"faixa {km_within.rho.min():.3f} a {km_within.rho.max():.3f}")
print("torque/potência (Nm/bhp), mediana por combustível:",
      (df.torque_nm / df.power_bhp).groupby(df.fuel).median().round(2).to_dict())
print("engine_cc x preço, por câmbio:",
      {t: round(g[['engine_cc', 'price']].corr('spearman').iloc[0, 1], 3)
       for t, g in df.groupby('transmission')})

fig, ax = plt.subplots(figsize=(7, 5.8))
mask = np.triu(np.ones_like(spear, dtype=bool), 1)
sns.heatmap(spear, mask=mask, annot=True, fmt=".2f", cmap=DIV, vmin=-1, vmax=1,
            linewidths=2, linecolor="white", square=True,
            cbar_kws={"label": "ρ de Spearman"}, ax=ax, annot_kws={"size": 8})
ax.grid(False)  # as bordas brancas das células fazem o papel de grade
ax.set_title("Figura 8. Correlação de Spearman (numéricas e alvo)")
ax.set_xlabel("variável")
ax.set_ylabel("variável")
save(fig, "fig08-spearman.png")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for a, (xc, yc) in zip(axes, [("engine_cc", "torque_nm"), ("power_bhp", "torque_nm"),
                              ("engine_cc", "power_bhp")]):
    for src in ["v3", "v4"]:
        s = df[df.source == src]
        a.scatter(s[xc], s[yc], s=8, alpha=0.35, color=SOURCE_COLORS[src], label=src,
                  edgecolors="none")
    a.set_xscale("log")
    a.set_yscale("log")
    plain_log(a)
    a.set(title=f"{xc} × {yc}  (ρ = {spear.loc[xc, yc]:.2f})",
          xlabel=f"{units[xc]} (log)", ylabel=f"{units[yc]} (log)")
    a.legend(title="Origem", markerscale=2)
fig.suptitle("Figura 9. Os três pares redundantes do bloco técnico", fontweight="bold")
fig.tight_layout()
save(fig, "fig09-redundant-pairs.png")

# =============================================================================
# 3B. Categórica x alvo
# =============================================================================
header("3B. Categóricas x preço")
d = df.assign(logp=np.log10(df.price))
for c in ["fuel", "transmission", "owner", "is_individual"]:
    g = d.groupby(c)["price"].agg(["size", "median"])
    print(f"{c}: " + str({k: (int(r['size']), int(r['median'])) for k, r in g.iterrows()}))

fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
for a, (c, lab) in zip(axes, [("fuel", "combustível"), ("transmission", "câmbio"),
                              ("owner", "nº de donos anteriores"),
                              ("is_individual", "vendedor (1 = particular)")]):
    order = sorted(d[c].unique()) if c in ("owner", "is_individual") else \
        d.groupby(c)["logp"].median().sort_values().index.tolist()
    sns.boxplot(data=d, x=c, y="logp", order=order, ax=a, color="#9cc0eb",
                fliersize=1.5, linewidth=1)
    meds = d.groupby(c)["logp"].median().reindex(order)
    a.plot(range(len(order)), meds.values, "o", color="#0b0b0b", ms=4, label="mediana")
    a.set_xticks(range(len(order)),
                 [f"{o}\nn={int((d[c] == o).sum())}" for o in order], fontsize=7)
    a.set(title=c, xlabel=lab, ylabel="log10(preço em ₹)")
    a.legend(fontsize=7, loc="upper left")
fig.suptitle("Figura 10. log10(preço) por categoria", fontweight="bold")
fig.tight_layout()
save(fig, "fig10-price-by-category.png")

top_brands = brand_vc[brand_vc >= 20].index
bmed = d[d.brand.isin(top_brands)].groupby("brand")["price"].median().sort_values()
print("preço mediano por marca (lakh):", (bmed / 1e5).round(1).to_dict())
print(f"preço mediano por marca (>= 20 anúncios): mín {bmed.index[0]} {bmed.iloc[0]:.0f}, "
      f"máx {bmed.index[-1]} {bmed.iloc[-1]:.0f}, razão {bmed.iloc[-1] / bmed.iloc[0]:.1f}x")
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=d[d.brand.isin(top_brands)], y="brand", x="logp", order=bmed.index,
            ax=ax, color="#9cc0eb", fliersize=1.5, linewidth=1, orient="h")
ax.plot(np.log10(bmed.values), range(len(bmed)), "o", color="#0b0b0b", ms=4,
        label="mediana")
ax.set(title=f"Figura 11. log10(preço) por marca ({len(top_brands)} marcas com ≥ 20 "
             f"anúncios)", xlabel="log10(preço em ₹)", ylabel="marca")
ax.legend(loc="lower right")
save(fig, "fig11-price-by-brand.png")

# O efeito da origem sobrevive quando se comparam carros parecidos?
# --8<-- [start:source_effect]
d["year_bin"] = pd.cut(d.year, [1980, 2010, 2013, 2016, 2018, 2023])
cells = (d.groupby(["brand", "year_bin", "transmission", "source"], observed=True)["logp"]
          .agg(["median", "size"]).unstack("source"))
cells = cells[(cells[("size", "v3")] >= 3) & (cells[("size", "v4")] >= 3)]
diff = cells[("median", "v4")] - cells[("median", "v3")]
w = cells[("size", "v3")] + cells[("size", "v4")]
raw_gap = d[d.source == "v4"].logp.median() - d[d.source == "v3"].logp.median()
matched_gap = np.average(diff, weights=w)
# --8<-- [end:source_effect]
print(f"Diferença v4 - v3 em log10(preço): bruta {raw_gap:.3f} ({10 ** raw_gap:.2f}x); "
      f"pareada por marca x faixa de ano x câmbio {matched_gap:.3f} "
      f"({10 ** matched_gap:.2f}x), em {len(cells)} células, {int(w.sum())} anúncios; "
      f"células com v4 > v3: {(diff > 0).mean() * 100:.0f}%")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, tr in zip(axes, ["Manual", "Automatic"]):
    for src in ["v3", "v4"]:
        s = d[(d.source == src) & (d.transmission == tr)]
        g = s.groupby("year")["logp"].agg(["median", "size"])
        g = g[g["size"] >= 10]
        a.plot(g.index, g["median"], "-o", color=SOURCE_COLORS[src], lw=2, ms=4,
               label=f"{src} (n = {len(s)})")
    a.set(title=f"Câmbio {tr}", xlabel="ano do carro",
          ylabel="mediana de log10(preço em ₹)")
    a.legend(title="Origem")
fig.suptitle("Figura 12. Mesmo ano e mesmo câmbio: o v4 continua mais caro? "
             "(pontos com ≥ 10 anúncios)", fontweight="bold")
fig.tight_layout()
save(fig, "fig12-source-effect.png")

# =============================================================================
# 3C. Numérica x categórica
# =============================================================================
header("3C. Numéricas x categóricas")
for c, g in [("power_bhp", "transmission"), ("km", "fuel"), ("year", "owner")]:
    s = df.groupby(g)[c].agg(["median", lambda v: v.quantile(.75) - v.quantile(.25)])
    s.columns = ["mediana", "IQR"]
    print(f"{c} por {g}:\n{s.round(1).to_string()}")
print("power_bhp por câmbio e origem (mediana):",
      df.groupby(["transmission", "source"])["power_bhp"].median().to_dict())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
sns.boxplot(data=df, x="transmission", y="power_bhp", hue="source",
            palette=SOURCE_COLORS, ax=axes[0], fliersize=1.5, linewidth=1)
axes[0].set_yscale("log")
plain_log(axes[0], "y")
axes[0].set(title="power_bhp por câmbio e origem", xlabel="câmbio",
            ylabel="potência (bhp, log)")
axes[0].legend(title="Origem")
fo = df.groupby("fuel")["km"].median().sort_values().index
sns.boxplot(data=df, x="fuel", y="km", order=fo, ax=axes[1], color="#9cc0eb",
            fliersize=1.5, linewidth=1)
axes[1].set_yscale("log")
axes[1].plot(range(len(fo)), df.groupby("fuel")["km"].median().reindex(fo), "o",
             color="#0b0b0b", ms=4, label="mediana")
axes[1].set(title="km por combustível", xlabel="combustível", ylabel="km rodados (log)")
axes[1].legend()
sns.boxplot(data=df, x="owner", y="year", ax=axes[2], color="#9cc0eb", fliersize=1.5,
            linewidth=1)
axes[2].plot(range(5), df.groupby("owner")["year"].median(), "o", color="#0b0b0b", ms=4,
             label="mediana")
axes[2].set(title="ano por nº de donos", xlabel="nº de donos anteriores (0 = sem dono)",
            ylabel="ano do carro")
axes[2].legend()
fig.suptitle("Figura 13. Numéricas agrupadas por categoria", fontweight="bold")
fig.tight_layout()
save(fig, "fig13-numeric-by-category.png")

# =============================================================================
# 4A / 4C. Pipeline
# =============================================================================
header("4A/4C. Pipeline")
# --8<-- [start:pipeline]
preprocess = build_preprocess()
Z_train = preprocess.fit_transform(X_train)   # aprende E transforma (só treino)
Z_test = preprocess.transform(X_test)         # SÓ transforma
names = preprocess.get_feature_names_out()
# --8<-- [end:pipeline]
print(f"Z_train {Z_train.shape}, Z_test {Z_test.shape}; NaN: "
      f"{int(np.isnan(Z_train).sum())} / {int(np.isnan(Z_test).sum())}")
print(f"features ({len(names)}): {list(names)}")
log_pipe = preprocess.named_transformers_["log"]
num_pipe = preprocess.named_transformers_["num"]
print("medianas aprendidas no treino (log):",
      dict(zip(["km", "engine_cc", "power_bhp", "torque_nm"],
               log_pipe.named_steps["impute"].statistics_.round(1))))
print("medianas do dataset inteiro         :",
      df[["km", "engine_cc", "power_bhp", "torque_nm"]].median().round(1).to_dict())
clip = log_pipe.named_steps["clip"]
print("limites de corte (treino):",
      {c: (round(lo, 1), round(hi, 1))
       for c, lo, hi in zip(["km", "engine_cc", "power_bhp", "torque_nm"],
                            clip.low_, clip.high_)})
Xtr_imp = log_pipe.named_steps["impute"].transform(X_train[["km", "engine_cc",
                                                            "power_bhp", "torque_nm"]])
clipped = ((Xtr_imp < clip.low_) | (Xtr_imp > clip.high_))
print(f"linhas do treino afetadas pelo corte: {int(clipped.any(axis=1).sum())} "
      f"({100 * clipped.any(axis=1).mean():.2f}%); por coluna "
      f"{dict(zip(['km', 'engine_cc', 'power_bhp', 'torque_nm'], clipped.sum(axis=0)))}")
ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
print("categorias infrequentes agrupadas:",
      {c: list(i) if i is not None else [] for c, i in
       zip(CATEGORICAL, ohe.infrequent_categories_)})
cont = slice(1, 7)  # log__* e num__year, num__seats (coluna 0 = indicador de ausência)
print(f"indicador de ausência: treino {int(Z_train[:, 0].sum())}, teste {int(Z_test[:, 0].sum())}")
print(f"média das contínuas, treino: {np.abs(Z_train[:, cont].mean(axis=0)).max():.2e} (máx |.|)")
print(f"média das contínuas, teste : {Z_test[:, cont].mean(axis=0).round(3)}")
print(f"dp das contínuas, treino   : {Z_train[:, cont].std(axis=0).round(3)}")

# =============================================================================
# 4B. Redução de dimensionalidade
# =============================================================================
header("4B. PCA, t-SNE, UMAP")
import umap  # noqa: E402  (import tardio: demora alguns segundos)

# --8<-- [start:pca]
pca = PCA(random_state=SEED).fit(Z_train)
cum = np.cumsum(pca.explained_variance_ratio_)
A_train = pca.transform(Z_train)[:, :2]
# --8<-- [end:pca]
print("variância explicada (primeiras 8):", pca.explained_variance_ratio_[:8].round(4))
print("acumulada (primeiras 8):", cum[:8].round(4))
print(f"PC1 + PC2 = {cum[1]:.4f}; componentes para 80%: {np.searchsorted(cum, 0.8) + 1}, "
      f"90%: {np.searchsorted(cum, 0.9) + 1}; componentes ~0 (< 1e-6): "
      f"{(pca.explained_variance_ratio_ < 1e-6).sum()}")
for i in range(2):
    load = pd.Series(pca.components_[i], index=names)
    top = load.reindex(load.abs().sort_values(ascending=False).index).head(6)
    print(f"PC{i + 1} loadings top 6: {top.round(3).to_dict()}")
print(f"corr(PC1, log preço) treino: {np.corrcoef(A_train[:, 0], y_train)[0, 1]:.3f}; "
      f"corr(PC2, log preço): {np.corrcoef(A_train[:, 1], y_train)[0, 1]:.3f}")

# Amostra do treino para t-SNE/UMAP (e para comparar os 3 na mesma base)
# --8<-- [start:sample]
N_SAMPLE = 3000
idx = RNG.choice(len(Z_train), size=N_SAMPLE, replace=False)
Zs, ys = Z_train[idx], y_train.to_numpy()[idx] / np.log(10)  # log10, como nas figuras


def price_coherence(E, y, k=10):
    """Desvio do log(preço) entre os k vizinhos no mapa / desvio global.

    Mede se o mapa põe carros de preço parecido perto uns dos outros: 1 = vizinhança
    tão variada quanto o dataset inteiro (nenhuma organização por preço), 0 = vizinhos
    com o mesmo preço. Só vizinhos mais próximos, nenhum modelo é ajustado.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    _, ind = nn.kneighbors(E)
    return float(np.mean(y[ind[:, 1:]].std(axis=1)) / y.std())
# --8<-- [end:sample]


maps, metrics = {}, []


def evaluate(name, E, seconds):
    maps[name] = E
    tw5 = trustworthiness(Zs, E, n_neighbors=5)
    tw30 = trustworthiness(Zs, E, n_neighbors=30)
    coh = price_coherence(E, ys)
    metrics.append((name, tw5, tw30, coh, seconds))
    print(f"  {name:22s} trust k=5 {tw5:.3f}  k=30 {tw30:.3f}  coerência {coh:.3f}  "
          f"({seconds:.1f} s)")


evaluate("PCA", pca.transform(Zs)[:, :2], 0.0)
# --8<-- [start:nonlinear]
for perp in [15, 50]:
    t0 = time.time()
    E = TSNE(2, perplexity=perp, init="pca", random_state=SEED).fit_transform(Zs)
    evaluate(f"t-SNE perp={perp}", E, time.time() - t0)
for nn in [15, 50]:
    t0 = time.time()
    E = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=SEED).fit_transform(Zs)
    evaluate(f"UMAP nn={nn}", E, time.time() - t0)
# --8<-- [end:nonlinear]

# Controle: as mesmas colunas, embaralhadas uma a uma (marginais intactas, relações
# destruídas). Se o mapa do ruído parecer igualmente organizado, o que se viu é o método.
# --8<-- [start:control]
Z_shuf = np.column_stack([RNG.permutation(col) for col in Zs.T])
E_shuf = TSNE(2, perplexity=50, init="pca", random_state=SEED).fit_transform(Z_shuf)
coh_shuf = price_coherence(E_shuf, ys)
coh_raw = price_coherence(Zs, ys)
# --8<-- [end:control]
print(f"  controle (colunas embaralhadas, t-SNE perp=50): coerência {coh_shuf:.3f}")
print(f"  referência: coerência no espaço original de {Zs.shape[1]} dims = {coh_raw:.3f}")
# O que forma as ilhas? Fração dos 10 vizinhos no mapa com o mesmo rótulo, contra o
# acaso (soma dos quadrados das proporções).
Xs = X_train.iloc[idx]
_, ind = NearestNeighbors(n_neighbors=11).fit(maps["UMAP nn=50"]).kneighbors(maps["UMAP nn=50"])
for label, lab in [("transmission", Xs.transmission), ("fuel", Xs.fuel),
                   ("transmission+fuel", Xs.transmission + "_" + Xs.fuel),
                   ("brand", Xs.brand)]:
    lab = lab.to_numpy()
    purity = (lab[ind[:, 1:]] == lab[:, None]).mean()
    chance = (pd.Series(lab).value_counts(normalize=True) ** 2).sum()
    print(f"  UMAP nn=50, vizinhos com mesmo {label}: {purity:.3f} (acaso {chance:.3f})")
top = ys > np.quantile(ys, 0.9)
print(f"  10% mais caros da amostra: {100 * (Xs.transmission[top] == 'Automatic').mean():.1f}% "
      f"automáticos (amostra toda: {100 * (Xs.transmission == 'Automatic').mean():.1f}%)")
print(f"  tem .transform()? PCA {hasattr(pca, 'transform')}, "
      f"t-SNE {hasattr(TSNE(), 'transform')}, UMAP {hasattr(umap.UMAP(), 'transform')}")


def scatter_map(a, E, title, xl, yl):
    sc = a.scatter(E[:, 0], E[:, 1], c=ys, cmap=SEQ, s=5, alpha=0.9, edgecolors="none",
                   vmin=np.percentile(ys, 1), vmax=np.percentile(ys, 99))
    a.set(title=title, xlabel=xl, ylabel=yl)
    return sc


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
k = np.arange(1, len(cum) + 1)
axes[0].bar(k, pca.explained_variance_ratio_, color="#9cc0eb", label="por componente")
axes[0].plot(k, cum, "-o", color="#2a78d6", ms=3, lw=2, label="acumulada")
axes[0].axhline(0.8, color="#52514e", ls=":", lw=1.2, label="80%")
axes[0].set(title=f"Variância explicada (PC1+PC2 = {100 * cum[1]:.1f}%)",
            xlabel="componente", ylabel="fração da variância", xlim=(0, 25))
axes[0].legend()
sc = scatter_map(axes[1], maps["PCA"], "PC1 × PC2 (amostra de 3000 do treino)",
                 f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)",
                 f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
fig.colorbar(sc, ax=axes[1], label="log10(preço em ₹)", extend="both")
fig.suptitle("Figura 14. PCA das features padronizadas do treino", fontweight="bold")
fig.tight_layout()
save(fig, "fig14-pca.png")

for fig_n, prefix, keys in [(15, "t-SNE", ["t-SNE perp=15", "t-SNE perp=50"]),
                            (16, "UMAP", ["UMAP nn=15", "UMAP nn=50"])]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, key in zip(axes, keys):
        m = [r for r in metrics if r[0] == key][0]
        sc = scatter_map(a, maps[key], f"{key}\ntrust k=5 {m[1]:.3f} · coerência {m[3]:.2f}",
                         f"{prefix} 1", f"{prefix} 2")
    fig.colorbar(sc, ax=axes, label="log10(preço em ₹)", extend="both")
    fig.suptitle(f"Figura {fig_n}. {prefix} com dois valores do parâmetro de vizinhança "
                 f"(amostra de 3000 do treino)", fontweight="bold", y=1.04)
    save(fig, f"fig{fig_n}-{prefix.lower().replace('-', '')}.png")

# =============================================================================
# 5. Resumo
# =============================================================================
header("5. Results summary")
miss_top = missing["%"].idxmax()
print(f"1  Dataset: CarDekho v3 + v4, regressão, alvo price (modelado em log)")
print(f"2  {len(df)} instâncias x {len(NUMERIC) + len(CATEGORICAL)} features "
      f"({len(NUMERIC)} numéricas / {len(CATEGORICAL)} categóricas)")
print(f"3  Mais ausente: {miss_top} {missing.loc[miss_top, '%']:.2f}%")
print("4  Descartadas: name/Model (identificador), mileage e 7 colunas só do v4 "
      "(ausência estrutural), source (artefato da coleta)")
print(f"5  Alvo: média {y.mean():.0f}, mediana {y.median():.0f}")
print(f"6  Treino {len(X_train)}, teste {len(X_test)}")
print(f"7  Par mais correlacionado: {pairs.iloc[0].level_0} x {pairs.iloc[0].level_1} "
      f"ρ = {pairs.iloc[0].rho:.3f}")
print(f"8  Linhas afetadas pelo corte de outliers (treino): {int(clipped.any(axis=1).sum())}")
print(f"9  PC1 + PC2 = {100 * cum[1]:.2f}%")
print(f"10 Shape após pipeline: treino {Z_train.shape}, teste {Z_test.shape}")
