"""Exercise 3: Preparing Real-World Data for a Neural Network.

Carrega o Spaceship Titanic (``train.csv``), descreve o dataset bruto (item
A), faz o split treino/teste (item B), aplica o pipeline de pré-processamento
(imputação, one-hot, TotalSpend, log1p, escalonamento, ajustado só no
treino) no item C e reporta as verificações finais (item D). Salva a figura em
``figures/`` e imprime os números que alimentam o relatório e a tabela
*Results summary*.

Uso (a partir da raiz do repositório), com o dataset já baixado do Kaggle
(https://www.kaggle.com/competitions/spaceship-titanic/data) em
``docs/exercises/data/code/data/train.csv``:

    python docs/exercises/data/code/exercise3_spaceship_titanic.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

FIGURES = Path(__file__).resolve().parents[1] / "figures"
DATA_PATH = Path(__file__).resolve().parent / "data" / "train.csv"

SEED = 42
TARGET = "Transported"
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC_COLS = ["Age"] + SPEND_COLS
CATEGORICAL_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP_COLS = ["Cabin", "Name", "PassengerId"]


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset nao encontrado. Baixe train.csv em "
            "https://www.kaggle.com/competitions/spaceship-titanic/data "
            f"e salve em {DATA_PATH}"
        )
    return pd.read_csv(DATA_PATH)


def describe_raw(df: pd.DataFrame) -> None:
    """Item A: descrição do dataset bruto, antes de qualquer split ou transformação."""
    balance = df[TARGET].value_counts(normalize=True)
    print("Balanco de classes (Transported):")
    print(balance.to_string())

    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print("\nValores ausentes por coluna (contagem e %):")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
    print(f"Total de colunas com valores ausentes: {int((missing > 0).sum())}")

    print("\nEstatisticas das colunas de gasto (dados brutos, antes do split):")
    print(df[SPEND_COLS].agg(["mean", "median", "max"]).T.to_string())


def figure_6(before: pd.Series, after: pd.Series) -> None:
    """Figura 6: FoodCourt antes/depois de log1p, no treino."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(before, bins=40, color="tab:blue", alpha=0.8)
    axes[0].set_title("FoodCourt, antes (treino, apos imputacao)")
    axes[0].set_xlabel("FoodCourt")
    axes[0].set_ylabel("Contagem")

    axes[1].hist(after, bins=40, color="tab:orange", alpha=0.8)
    axes[1].set_title("FoodCourt, depois de log1p (treino)")
    axes[1].set_xlabel("log1p(FoodCourt)")
    axes[1].set_ylabel("Contagem")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-foodcourt-before-after.png", dpi=160)
    plt.close(fig)  # (2)!


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print(f"Total de passageiros: {len(df)}")
    describe_raw(df)

    # B: Split ANTES de qualquer imputacao/encoding/escalonamento.
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )  # (1)!
    X_train, X_test = X_train.copy(), X_test.copy()
    print(f"\nAmostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}")

    # C1: Imputacao, mediana (numericas) e moda (categoricas), ajustadas so no treino.
    medians = X_train[NUMERIC_COLS].median()
    for col in NUMERIC_COLS:
        X_train[col] = X_train[col].fillna(medians[col])
        X_test[col] = X_test[col].fillna(medians[col])

    modes = {col: X_train[col].mode(dropna=True).iloc[0] for col in CATEGORICAL_COLS}
    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].fillna(modes[col])
        X_test[col] = X_test[col].fillna(modes[col])

    print(
        f"\nFoodCourt no treino, apos imputacao e antes de transformar: "
        f"media={X_train['FoodCourt'].mean():.4f}, mediana={X_train['FoodCourt'].median():.4f}"
    )

    # C3: Feature engineering, TotalSpend = soma das 5 colunas de gasto.
    X_train["TotalSpend"] = X_train[SPEND_COLS].sum(axis=1)
    X_test["TotalSpend"] = X_test[SPEND_COLS].sum(axis=1)
    spend_and_total = SPEND_COLS + ["TotalSpend"]

    # C4: Cauda pesada, log1p nas colunas de gasto e no TotalSpend.
    foodcourt_before = X_train["FoodCourt"].copy()
    for col in spend_and_total:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])
    foodcourt_after = X_train["FoodCourt"].copy()
    figure_6(foodcourt_before, foodcourt_after)

    # C2: One-hot ajustado so no treino; categorias novas no teste viram tudo-zero.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS])
    cat_train = encoder.transform(X_train[CATEGORICAL_COLS])
    cat_test = encoder.transform(X_test[CATEGORICAL_COLS])
    cat_names = encoder.get_feature_names_out(CATEGORICAL_COLS)

    # C5: Escalonamento para [-1, 1], ajustado so no treino.
    numeric_final = NUMERIC_COLS + ["TotalSpend"]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train[numeric_final])
    num_train = scaler.transform(X_train[numeric_final])
    num_test = scaler.transform(X_test[numeric_final])

    X_train_final = np.hstack([num_train, cat_train])
    X_test_final = np.hstack([num_test, cat_test])
    feature_names = list(numeric_final) + list(cat_names)

    # D2: Verificacoes finais.
    print(f"\nFeatures apos encoding: {len(feature_names)}")
    print(f"NaN restantes, treino: {int(np.isnan(X_train_final).sum())}, "
          f"teste: {int(np.isnan(X_test_final).sum())}")
    print(f"Formato final da matriz de treino: {X_train_final.shape}")
    print(f"Formato final da matriz de teste: {X_test_final.shape}")
    print(f"Treino: min={X_train_final.min():.4f}, max={X_train_final.max():.4f}")
    print(f"Teste : min={X_test_final.min():.4f}, max={X_test_final.max():.4f}")


if __name__ == "__main__":
    main()
