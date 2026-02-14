from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import FEATURE_NAMES, Predictor


def main() -> None:
    predictor = Predictor.load()
    payload = {k: 5.0 for k in FEATURE_NAMES}
    proba = predictor.predict_proba(payload)
    print(proba)


if __name__ == "__main__":
    main()

