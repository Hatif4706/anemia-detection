import os
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

class DatasetLoader:
    """
    Manajer pemuatan, validasi, dan pembagian dataset konjungtiva palpebra.
    Menjamin partisi data (train, val, test) yang identik untuk seluruh model
    demi perbandingan yang setara (fair benchmark).
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        splits_dir: str = "data/splits",
        random_seed: int = 42
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.splits_dir = splits_dir
        self.random_seed = random_seed

    def create_stratified_splits(
        self,
        metadata_df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_col: str = "anemia_label"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Membagi metadata ke train, val, test secara stratified berdasarkan label anemia.
        Menyimpan file csv di data/splits/ untuk memastikan reproducibilitas absolut.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Rasio harus berjumlah 1.0"
        os.makedirs(self.splits_dir, exist_ok=True)

        # Split 1: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            metadata_df,
            test_size=temp_ratio,
            stratify=metadata_df[target_col],
            random_state=self.random_seed
        )

        # Split 2: Val vs Test
        val_relative_ratio = val_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_relative_ratio),
            stratify=temp_df[target_col],
            random_state=self.random_seed
        )

        train_path = os.path.join(self.splits_dir, "train.csv")
        val_path = os.path.join(self.splits_dir, "val.csv")
        test_path = os.path.join(self.splits_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"[DatasetLoader] Split tersimpan:")
        print(f" - Train: {len(train_df)} sampel ({train_path})")
        print(f" - Val:   {len(val_df)} sampel ({val_path})")
        print(f" - Test:  {len(test_df)} sampel ({test_path})")

        return train_df, val_df, test_df

    def generate_synthetic_benchmark_dataset(self, num_samples: int = 40):
        """
        Menghasilkan sampel citra dan mask sintetis untuk keperluan
        pengujian verifikasi pipeline secara instan sebelum dataset publik diunduh.
        """
        images_dir = os.path.join(self.raw_dir, "images")
        masks_dir = os.path.join(self.raw_dir, "masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        records = []
        np.random.seed(self.random_seed)

        for i in range(num_samples):
            img_id = f"sample_{i:03d}"
            # Tentukan status anemia: 1 = Anemia (pucat), 0 = Normal (merah segar)
            is_anemic = 1 if (i % 2 == 0) else 0

            # Buat citra dasar wajah/mata sintetis (480 x 640)
            img = np.full((480, 640, 3), fill_value=180, dtype=np.uint8)
            # Area wajah / kulit
            cv2.ellipse(img, (320, 240), (200, 230), 0, 0, 360, (190, 205, 225), -1)

            # Gambar dua mata (sclera)
            cv2.ellipse(img, (240, 220), (50, 25), 0, 0, 360, (245, 245, 245), -1)
            cv2.ellipse(img, (400, 220), (50, 25), 0, 0, 360, (245, 245, 245), -1)

            # Iris
            cv2.circle(img, (240, 220), 16, (50, 40, 30), -1)
            cv2.circle(img, (400, 220), 16, (50, 40, 30), -1)

            # Mask ground truth konjungtiva palpebra (kelopak mata bawah yang tertarik)
            mask = np.zeros((480, 640), dtype=np.uint8)
            left_conj = np.array([[200, 235], [240, 260], [280, 235], [240, 245]], dtype=np.int32)
            right_conj = np.array([[360, 235], [400, 260], [440, 235], [400, 245]], dtype=np.int32)
            cv2.fillPoly(mask, [left_conj, right_conj], 255)

            # Warnai area konjungtiva sesuai label anemia
            if is_anemic:
                # Pucat (pallor): Rona kemerahan rendah, lebih kekuningan/putih
                conj_color = (175, 175, 215) # BGR
            else:
                # Normal: Rona merah pekat (vaskularisasi baik)
                conj_color = (80, 90, 210)  # BGR

            # Terapkan warna konjungtiva pada citra
            img[mask == 255] = conj_color

            img_path = os.path.join(images_dir, f"{img_id}.jpg")
            mask_path = os.path.join(masks_dir, f"{img_id}.png")

            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)

            records.append({
                "sample_id": img_id,
                "image_path": img_path,
                "mask_path": mask_path,
                "anemia_label": is_anemic,
                "hemoglobin_est": 8.5 + np.random.uniform(0, 2.0) if is_anemic else 13.5 + np.random.uniform(0, 2.5)
            })

        df = pd.DataFrame(records)
        meta_path = os.path.join(self.raw_dir, "metadata.csv")
        df.to_csv(meta_path, index=False)
        print(f"[DatasetLoader] Generated {num_samples} sample data di {self.raw_dir}")

        return df
