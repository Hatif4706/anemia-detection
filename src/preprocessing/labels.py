"""Pelabelan Anemia / Non-anemia dari kadar Hb laboratorium (WHO; Kemenkes RI, 2018)."""
from typing import Optional

import numpy as np


def normalize_sex(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    v = str(value).strip().lower()
    if v in {"f", "female", "w", "woman", "perempuan", "p", "donna", "femmina"}:
        return "female"
    if v in {"m", "male", "man", "laki-laki", "l", "uomo", "maschio"}:
        return "male"
    return None


def hb_threshold(sex: Optional[str], age: Optional[float], cfg: dict) -> float:
    if age is not None and not (isinstance(age, float) and np.isnan(age)) and float(age) < 5:
        return cfg["child_under_5"]
    if sex == "female":
        return cfg["female"]
    if sex == "male":
        return cfg["male"]
    return cfg["unknown"]


def anemia_label(hb: float, sex: Optional[str], age: Optional[float], cfg: dict) -> int:
    """1 = Anemia (Hb di bawah batas), 0 = Non-anemia."""
    return int(float(hb) < hb_threshold(sex, age, cfg))
