import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def load_raw_data(
    file_path: Optional[str] = None,
    data_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Загружает датасет из CSV файла.
    
    Args:
        file_path: Полный путь к файлу. Если None, используется стандартный путь.
        data_dir: Директория с данными (по умолчанию data/raw)
    
    Returns:
        pd.DataFrame: Загруженный датасет
    
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если данные пустые или некорректные
    """
    if file_path is None:
        # Определяем путь относительно корня проекта
        project_root = Path(__file__).parent.parent
        file_path = project_root / data_dir / "personality_synthetic_dataset.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    logger.info(f"Загрузка данных из {file_path}")
    df = pd.read_csv(file_path)
    
    if df.empty:
        raise ValueError("Загруженный датасет пуст")
    
    logger.info(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
    return df


def save_processed_data(df: pd.DataFrame, file_path: str = "data/processed/data.csv"):
    """
    Сохраняет обработанные данные в CSV файл.
    
    Args:
        df: pd.DataFrame - обработанные данные
        file_path: Путь к файлу для сохранения
    """
    df.to_csv(file_path, index=False)
    logger.info(f"Обработанные данные сохранены в {file_path}")

def load_processed_data(file_path: str = "data/processed/data.csv") -> pd.DataFrame:
    """
    Загружает обработанные данные из CSV файла.
    
    Args:
        file_path: Путь к файлу для загрузки
    """
    return pd.read_csv(file_path)