import os
import sys
import argparse
import pandas as pd
from pathlib import Path

def setup_dataset_structure(data_dir="data"):
    Path(f"{data_dir}/raw/images").mkdir(parents=True, exist_ok=True)
    Path(f"{data_dir}/raw/masks").mkdir(parents=True, exist_ok=True)
    Path(f"{data_dir}/processed").mkdir(parents=True, exist_ok=True)
    Path(f"{data_dir}/splits").mkdir(parents=True, exist_ok=True)
    print(f"[OK] Direktori dataset terstruktur di {data_dir}/")

def print_dataset_instructions():
    print("""
========================================================================
PANDUAN PENYIAPAN DATASET (KAGGLE / DIMAURO ET AL. / DATASET KLINIS)
========================================================================
1. Unduh dataset konjungtiva anemia (misal dari Kaggle atau Dimauro et al.).
2. Letakkan file citra pada:
   data/raw/images/<id_pasien>.jpg
3. Letakkan file masker konjungtiva ground truth pada:
   data/raw/masks/<id_pasien>.png (grayscale, piksel 255 untuk konjungtiva)
4. Buat / perbarui file data/raw/metadata.csv dengan kolom:
   sample_id, image_path, mask_path, anemia_label, hemoglobin_est
   - anemia_label: 1 jika Anemia, 0 jika Non-Anemia
   - hemoglobin_est: nilai kadar Hb g/dL (opsional jika ada)

Jika Anda ingin menguji seluruh pipeline dengan data simulasi cepat:
   python scripts/run_benchmark.py
========================================================================
""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persiapan Dataset Anemia")
    parser.add_argument("--init", action="store_true", help="Buat folder dataset")
    args = parser.parse_args()

    setup_dataset_structure()
    print_dataset_instructions()
