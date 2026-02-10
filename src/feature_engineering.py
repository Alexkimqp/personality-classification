from __future__ import annotations

import pandas as pd
from typing import Mapping, Any, Sequence
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def payload_to_frame(payload: Mapping[str, Any], feature_names: Sequence[str]) -> pd.DataFrame:
    """
    payload: {"social_energy": 6.1, "alone_time_preference": 3.2, ...}
    Возвращает DataFrame(1 row) строго в нужном порядке колонок.
    """
    # защита от дубликатов в списке фич
    if len(set(feature_names)) != len(feature_names):
        dupes = pd.Series(list(feature_names)).value_counts()
        dupes = dupes[dupes > 1].index.tolist()
        raise ValueError(f"Duplicate feature names in feature_names: {dupes}")

    feature_set = set(feature_names)

    missing = [c for c in feature_names if c not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    extra = [k for k in payload.keys() if k not in feature_set]
    if extra:
        raise ValueError(f"Unexpected features: {extra}")

    row = {}
    for c in feature_names:
        try:
            row[c] = float(payload[c])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Feature '{c}' must be numeric, got {payload[c]!r}") from e

    return pd.DataFrame([row], columns=list(feature_names))


def get_highly_correlated_pairs(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Возвращает пары признаков с корреляцией >= threshold (по модулю).
    """
    corr = df.corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "corr"})
    )
    return pairs[pairs["corr"] >= threshold].sort_values("corr", ascending=False)


    


def compare_pairs_by_target_mi(
    pairs: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Для каждой пары сильно коррелирующих признаков
    сравнивает их связь с таргетом через Mutual Information
    и рекомендует, какой признак оставить.
    """


    mi = mutual_info_classif(
        X,
        y,
        random_state=random_state,
        discrete_features="auto",
    )

    mi_series = pd.Series(mi, index=X.columns)

    result = pairs.copy()

    result["mi_f1"] = result["feature_1"].map(mi_series)
    result["mi_f2"] = result["feature_2"].map(mi_series)

    result["keep_feature"] = np.where(
        result["mi_f1"] >= result["mi_f2"],
        result["feature_1"],
        result["feature_2"],
    )

    result["drop_feature"] = np.where(
        result["mi_f1"] < result["mi_f2"],
        result["feature_1"],
        result["feature_2"],
    )

    return result.sort_values(
        ["mi_f1", "mi_f2"], ascending=False
    )