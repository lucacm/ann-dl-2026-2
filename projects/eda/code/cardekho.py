"""Projeto, entrega 1 (EDA): dados CarDekho, harmonização v3 + v4 e pipeline.

Módulo importável com tudo que as entregas seguintes reaproveitam: leitura dos CSVs
brutos, harmonização dos dois arquivos escolhidos num esquema comum, split treino/teste
e o ``ColumnTransformer`` de pré-processamento. Os números e figuras do relatório saem de
``eda.py``, que importa daqui.

Os CSVs não são versionados. Baixe o dataset em
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho e descompacte os
4 arquivos em ``docs/projects/eda/code/data/``.

Uso (a partir da raiz do repositório):

    import sys; sys.path.insert(0, "docs/projects/eda/code")
    from cardekho import load_dataset, split, build_preprocess
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parent / "data"

# --8<-- [start:setup]
SEED = 42

FILES = {
    "car data": "car data.csv",
    "FROM DEKHO": "CAR DETAILS FROM CAR DEKHO.csv",
    "v3": "Car details v3.csv",
    "v4": "car details v4.csv",
}

TARGET = "price"
# --8<-- [end:setup]


def load_raw(key):
    """Lê um dos 4 CSVs brutos, sem nenhuma alteração."""
    path = DATA / FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Baixe o dataset do Kaggle "
            "(nehalbirla/vehicle-dataset-from-cardekho) e descompacte em "
            f"{DATA}/"
        )
    return pd.read_csv(path)


# --8<-- [start:harmonize]
# Marca: primeira palavra do nome no v3, coluna Make no v4. Três grafias não batem.
BRAND_FIX = {"Land": "Land Rover", "Ashok": "Ashok Leyland", "Maruti Suzuki": "Maruti"}

# Combustível: o v4 tem misturas e elétricos/híbridos que o v3 não tem.
FUEL_V4 = {"CNG + CNG": "CNG", "Petrol + CNG": "CNG", "Petrol + LPG": "LPG",
           "Electric": "Other", "Hybrid": "Other"}

# Dono: as duas escalas viram o mesmo ordinal. "Test Drive Car" e "UnRegistered Car"
# são carros sem dono anterior: 0.
OWNER_V3 = {"Test Drive Car": 0, "First Owner": 1, "Second Owner": 2,
            "Third Owner": 3, "Fourth & Above Owner": 4}
OWNER_V4 = {"UnRegistered Car": 0, "First": 1, "Second": 2, "Third": 3,
            "Fourth": 4, "4 or More": 4}


def first_number(s):
    """Primeiro número de uma string ("1248 CC", "74 bhp", "190Nm@ 2000rpm")."""
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+\.?\d*)")[0],
                         errors="coerce").astype(float)


def torque_nm(s):
    """Torque em Nm. No v3, ~500 linhas vêm em kgm (1 kgm = 9,80665 Nm).

    Algumas strings marcadas como kgm já trazem o número em Nm ("115@ 2,500(kgm@ rpm)"
    num Tata Sumo): nenhum carro passa de ~60 kgm, então só converte abaixo de 100.
    """
    value = first_number(s)
    is_kgm = s.astype("string").str.contains("kgm", case=False, na=False) & (value < 100)
    return value.where(~is_kgm, value * 9.80665)


def harmonize(v3, v4):
    """Leva v3 e v4 ao esquema comum: só as colunas que existem nos dois arquivos."""
    a = pd.DataFrame({
        "price": v3["selling_price"],
        "year": v3["year"],
        "km": v3["km_driven"],
        "fuel": v3["fuel"],
        "transmission": v3["transmission"],
        "owner": v3["owner"].map(OWNER_V3),
        "is_individual": (v3["seller_type"] == "Individual").astype(int),
        "brand": v3["name"].str.split().str[0].replace(BRAND_FIX),
        "engine_cc": first_number(v3["engine"]),
        "power_bhp": first_number(v3["max_power"]),
        "torque_nm": torque_nm(v3["torque"]),
        "seats": v3["seats"],
        "source": "v3",
    })
    b = pd.DataFrame({
        "price": v4["Price"],
        "year": v4["Year"],
        "km": v4["Kilometer"],
        "fuel": v4["Fuel Type"].replace(FUEL_V4),
        "transmission": v4["Transmission"],
        "owner": v4["Owner"].map(OWNER_V4),
        "is_individual": (v4["Seller Type"] == "Individual").astype(int),
        "brand": v4["Make"].replace(BRAND_FIX),
        "engine_cc": first_number(v4["Engine"]),
        "power_bhp": first_number(v4["Max Power"]),
        "torque_nm": torque_nm(v4["Max Torque"]),
        "seats": v4["Seating Capacity"],
        "source": "v4",
    })
    df = pd.concat([a, b], ignore_index=True)
    # Potência 0 bhp é impossível (6 linhas do v3): é ausência disfarçada.
    df.loc[df["power_bhp"] == 0, "power_bhp"] = np.nan
    # Torque/cilindrada acima de 0,4 Nm/cc não existe em motor de rua (o máximo real
    # aqui é 0,33): é o "789Nm" do Maruti Zen D de 58 bhp, erro de digitação na fonte.
    df.loc[df["torque_nm"] / df["engine_cc"] > 0.4, "torque_nm"] = np.nan
    for c in ["fuel", "transmission", "brand", "source"]:
        df[c] = df[c].astype(object)
    return df
# --8<-- [end:harmonize]


# --8<-- [start:dedup]
def deduplicate(raw, price_col):
    """Remove duplicatas no arquivo bruto, antes da harmonização.

    1. Linhas idênticas em todas as colunas: cópia, fica uma.
    2. Linhas idênticas em tudo menos o preço: o mesmo carro reanunciado (diferença
       mediana de ~11%). Viram uma linha com o preço mediano do grupo, senão o mesmo
       carro pode cair no treino e no teste.

    Deduplicar depois de harmonizar seria errado: sem ``name``/``mileage``, versões
    diferentes do mesmo modelo passariam por cópias.
    """
    raw = raw.drop_duplicates()
    keys = [c for c in raw.columns if c != price_col]
    return (raw.groupby(keys, dropna=False, sort=False)[price_col].median()
               .reset_index()[raw.columns])


def load_dataset():
    """Dataset final do projeto: v3 + v4 deduplicados e harmonizados."""
    v3 = deduplicate(load_raw("v3"), "selling_price")
    v4 = deduplicate(load_raw("v4"), "Price")
    return harmonize(v3, v4)
# --8<-- [end:dedup]


# --8<-- [start:split]
NUMERIC = ["year", "km", "engine_cc", "power_bhp", "torque_nm", "seats",
           "owner", "is_individual"]
CATEGORICAL = ["fuel", "transmission", "brand"]


def split(df, test_size=0.2):
    """Split 80/20 estratificado por origem x quintil de log(preço).

    Estratificar só por origem deixaria a faixa de preço ao acaso; só por preço, a
    proporção v3/v4 (que difere em preço, câmbio e potência) poderia mudar entre os
    lados. Os quintis servem só para o sorteio e não entram no modelo.
    """
    strata = df["source"] + "_" + pd.qcut(np.log(df[TARGET]), 5, labels=False).astype(str)
    train, test = train_test_split(df, test_size=test_size, random_state=SEED,
                                   stratify=strata)
    X_train, X_test = train[NUMERIC + CATEGORICAL], test[NUMERIC + CATEGORICAL]
    y_train, y_test = np.log(train[TARGET]), np.log(test[TARGET])
    return X_train, X_test, y_train, y_test, train["source"], test["source"]
# --8<-- [end:split]


# --8<-- [start:preprocess]
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Limita cada coluna aos quantis [q, 1-q] aprendidos no treino."""

    def __init__(self, q=0.005):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.q, axis=0)
        self.high_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# Assimetria forte (km, potência, torque, cilindrada): log antes de padronizar.
LOG_FEATURES = ["km", "engine_cc", "power_bhp", "torque_nm"]
LINEAR_FEATURES = [c for c in NUMERIC if c not in LOG_FEATURES]


def build_preprocess(min_brand_frequency=20):
    """ColumnTransformer do projeto. Toda estatística é aprendida no ``fit`` (treino)."""
    log_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper(q=0.005)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    linear_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_brand_frequency,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        # A ausência das colunas técnicas não é aleatória (carros mais velhos e mais
        # baratos): a imputação apagaria isso, então uma coluna 0/1 guarda o fato.
        ("missing", MissingIndicator(features="all"), ["engine_cc"]),
        ("log", log_pipe, LOG_FEATURES),
        ("num", linear_pipe, LINEAR_FEATURES),
        ("cat", cat_pipe, CATEGORICAL),
    ])
# --8<-- [end:preprocess]
