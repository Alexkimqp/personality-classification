from __future__ import annotations

import pandas as pd
from typing import Mapping, Any, Sequence



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