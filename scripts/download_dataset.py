"""Panduan penyiapan dataset Eyes-defy-anemia dan pemeriksaan strukturnya.

    python scripts/download_dataset.py          # cek apakah dataset sudah di tempat yang benar
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, load_config

GUIDE = """
Dataset Eyes-defy-anemia (Dimauro et al., 2023)
  Sumber : https://ieee-dataport.org/documents/eyes-defy-anemia  (DOI 10.21227/t5s2-4j73, perlu login)
  Isi    : 218 citra (Italy 123, India 95), nilai Hb laboratorium, mask palpebral.png buatan ahli
Langkah:
  1. Login ke IEEE DataPort lalu unduh berkas dataset.
  2. Ekstrak ke: {target}
     sehingga ada {target}/Italy/Italy.xlsx dan {target}/India/India.xlsx (atau .xls)
  3. Jalankan: python scripts/prepare_data.py
"""


def main():
    cfg = load_config()
    target = ROOT / cfg["dataset"]["eyes_defy_anemia_dir"]
    print(GUIDE.format(target=target))
    for country, n in [("Italy", 123), ("India", 95)]:
        d = target / country
        if not d.exists():
            print(f"[BELUM] {d} tidak ditemukan")
            continue
        folders = [p for p in d.iterdir() if p.is_dir() and p.name.isdigit()]
        sheets = list(d.glob("*.xls*"))
        print(f"[ADA]   {country}: {len(folders)}/{n} folder pasien, spreadsheet: {[s.name for s in sheets] or 'TIDAK ADA'}")


if __name__ == "__main__":
    main()
