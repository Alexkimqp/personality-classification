from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_from_root(path_value: Union[str, Path]) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return project_root() / p


def load_raw_data(
    file_path: Optional[Union[str, Path]] = None,
    data_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Загружает датасет из CSV файла.
    Если file_path=None, берёт data/raw/personality_synthetic_dataset.csv от корня проекта.
    """
    if file_path is None:
        p = project_root() / data_dir / "personality_synthetic_dataset.csv"
    else:
        p = _resolve_from_root(file_path)

    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")

    logger.info(f"Загрузка данных из {p}")
    df = pd.read_csv(p)

    if df.empty:
        raise ValueError("Загруженный датасет пуст")

    logger.info(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
    return df


def save_processed_data(df: pd.DataFrame, file_path: Union[str, Path] = "data/processed/data.csv") -> Path:
    p = _resolve_from_root(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Обработанные данные сохранены в {p}")
    return p


def load_processed_data(file_path: Union[str, Path] = "data/processed/data.csv") -> pd.DataFrame:
    p = _resolve_from_root(file_path)
    return pd.read_csv(p)


def load_split_csv(filename: str, split_dir: str = "data/processed") -> pd.DataFrame:
    return pd.read_csv(project_root() / split_dir / filename)


def load_artifact(filename: str, split_dir: str = "data/artifacts") -> Any:
    return joblib.load(project_root() / split_dir / filename)


def load_model(filename: str, model_dir: str = "models") -> Any:
    return joblib.load(project_root() / model_dir / filename)

def load_splits(
    split_dir: str = "data/processed",
    scaled: bool = True,
    encoded_y: bool = True,
) -> Dict[str, Any]:
    xtr = "X_train_scaled.csv" if scaled else "X_train.csv"
    xte = "X_test_scaled.csv" if scaled else "X_test.csv"
    ytr = "y_train_enc.csv" if encoded_y else "y_train.csv"
    yte = "y_test_enc.csv" if encoded_y else "y_test.csv"

    X_train = load_split_csv(xtr, split_dir)
    X_test = load_split_csv(xte, split_dir)

    y_train = pd.read_csv(project_root() / split_dir / ytr).iloc[:, 0]
    y_test = pd.read_csv(project_root() / split_dir / yte).iloc[:, 0]

    scaler = load_artifact("standard_scaler.joblib", split_dir="data/artifacts")
    label_encoder = load_artifact("label_encoder.joblib", split_dir="data/artifacts")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": list(X_train.columns),
    }