import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

class EvaluationMetrics:
    """
    Kumpulan fungsi evaluasi terstandarisasi untuk Level 1 (Segmentasi/ROI) 
    dan Level 2 (Klasifikasi Medis Anemia).
    """

    # ==========================
    # LEVEL 1: SEGMENTATION ROI
    # ==========================
    @staticmethod
    def calculate_segmentation_metrics(
        pred_mask: np.ndarray, 
        gt_mask: np.ndarray, 
        eps: float = 1e-7
    ) -> Dict[str, float]:
        """
        Menghitung IoU (Jaccard Index), Dice Coefficient, Precision, dan Recall antara mask prediksi dan ground truth.
        """
        pred_b = (pred_mask > 0).astype(np.uint8)
        gt_b = (gt_mask > 0).astype(np.uint8)

        intersection = np.sum((pred_b == 1) & (gt_b == 1))
        union = np.sum((pred_b == 1) | (gt_b == 1))
        pred_total = np.sum(pred_b == 1)
        gt_total = np.sum(gt_b == 1)

        iou = float(intersection + eps) / float(union + eps)
        dice = float(2.0 * intersection + eps) / float(pred_total + gt_total + eps)
        precision = float(intersection + eps) / float(pred_total + eps)
        recall = float(intersection + eps) / float(gt_total + eps)

        return {
            "iou": iou,
            "dice_score": dice,
            "precision": precision,
            "recall": recall
        }

    # ==========================
    # LEVEL 2: MEDICAL CLASSIFICATION
    # ==========================
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Menghitung metrik diagnostik klinis:
        - Sensitivity / Recall: Kemampuan mendeteksi pasien positif anemia (TP / (TP + FN))
        - Specificity: Kemampuan mendeteksi subjek non-anemia yang benar (TN / (TN + FP))
        - Precision, F1-Score, Accuracy, ROC-AUC, dan Confusion Matrix.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0)) # Sensitivity
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Confusion Matrix [TN, FP], [FN, TP]
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        roc_auc = 0.0
        if y_proba is not None:
            try:
                # Pastikan format probabilitas kelas positif (1)
                if y_proba.ndim == 2:
                    pos_proba = y_proba[:, 1]
                else:
                    pos_proba = y_proba
                roc_auc = float(roc_auc_score(y_true, pos_proba))
            except Exception:
                roc_auc = 0.0

        return {
            "accuracy": acc,
            "sensitivity": rec,
            "specificity": specificity,
            "precision": prec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist()
        }
