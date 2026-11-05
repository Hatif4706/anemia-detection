"""Pembaca dataset Eyes-defy-anemia (Dimauro et al., 2023; IEEE DataPort, DOI 10.21227/t5s2-4j73).

Struktur yang diharapkan (sesuai deskripsi dataset):
    eyes_defy_anemia/
        Italy/  Italy.xlsx   1/ 2/ ... 123/
        India/  India.xlsx (atau .xls)   1/ ... 95/
    Tiap folder pasien: <citra asli>.jpg, palpebral.png, forniceal.png, forniceal_palpebral.png

Nama kolom spreadsheet tidak dibakukan oleh penyedia, sehingga kolom dicari secara fleksibel.
"""
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.preprocessing.labels import anemia_label, normalize_sex

MASK_FILES = {"palpebral.png", "forniceal.png", "forniceal_palpebral.png"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {c: "".join(ch for ch in str(c).lower() if ch.isalnum()) for c in df.columns}
    for cand in candidates:
        for c, n in norm.items():
            if n == cand:
                return c
    for cand in candidates:
        for c, n in norm.items():
            if cand in n:
                return c
    return None


def _read_sheet(country_dir: Path) -> pd.DataFrame:
    files = sorted(list(country_dir.glob("*.xlsx")) + list(country_dir.glob("*.xls")) + list(country_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"Spreadsheet tidak ditemukan di {country_dir}")
    f = files[0]
    df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
    num = _find_col(df, ["number", "num", "no", "id", "patient", "folder"])
    hb = _find_col(df, ["hgb", "hb", "hemoglobin", "haemoglobin", "hbgdl"])
    sex = _find_col(df, ["gender", "sex", "jeniskelamin"])
    age = _find_col(df, ["age", "eta", "umur", "usia"])
    if num is None or hb is None:
        raise ValueError(f"Kolom Number/Hb tidak dikenali di {f.name}. Kolom tersedia: {list(df.columns)}")
    out = pd.DataFrame({
        "number": pd.to_numeric(df[num], errors="coerce"),
        "hb": pd.to_numeric(df[hb].astype(str).str.replace(",", "."), errors="coerce"),
        "sex": df[sex].map(normalize_sex) if sex else None,
        "age": pd.to_numeric(df[age], errors="coerce") if age else np.nan,
    })
    return out.dropna(subset=["number"])


def _original_image(folder: Path) -> Optional[Path]:
    cands = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT and p.name.lower() not in MASK_FILES]
    if not cands:
        return None
    jpgs = [p for p in cands if p.suffix.lower() in {".jpg", ".jpeg"}]
    return sorted(jpgs or cands)[0]


def build_metadata(root: str, label_cfg: dict, mask_name: str = "palpebral.png") -> pd.DataFrame:
    root = Path(root)
    rows, skipped = [], []
    for country in ["Italy", "India"]:
        cdir = root / country
        if not cdir.exists():
            raise FileNotFoundError(f"Folder {cdir} tidak ada. Ekstrak dataset ke {root}")
        sheet = _read_sheet(cdir).set_index("number")
        for folder in sorted([p for p in cdir.iterdir() if p.is_dir()], key=lambda p: (len(p.name), p.name)):
            try:
                number = int(folder.name)
            except ValueError:
                continue
            img = _original_image(folder)
            mask = folder / mask_name
            if img is None or not mask.exists():
                skipped.append(f"{country}/{folder.name}: citra/mask tidak lengkap")
                continue
            if number not in sheet.index or pd.isna(sheet.loc[number, "hb"]):
                skipped.append(f"{country}/{folder.name}: nilai Hb tidak ada")
                continue
            r = sheet.loc[number]
            sex = r["sex"] if isinstance(r["sex"], str) else None
            age = None if pd.isna(r["age"]) else float(r["age"])
            rows.append({
                "sample_id": f"{country.lower()}_{number:03d}",
                "subject_id": f"{country.lower()}_{number:03d}",
                "country": country,
                "image_path": str(img),
                "mask_path": str(mask),
                "hb": float(r["hb"]),
                "sex": sex,
                "age": age,
                "anemia_label": anemia_label(float(r["hb"]), sex, age, label_cfg),
            })
    df = pd.DataFrame(rows)
    df.attrs["skipped"] = skipped
    return df
