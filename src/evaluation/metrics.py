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
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
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


def mcnemar_exact(y_true, pred_a, pred_b) -> Dict[str, Any]:
    """Uji McNemar eksak (binomial) pada prediksi berpasangan dua pipeline."""
    from scipy.stats import binomtest
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))  # A benar, B salah
    c = int(np.sum(~a_ok & b_ok))  # A salah, B benar
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"a_correct_b_wrong": b, "a_wrong_b_correct": c, "p_value": p}


def bootstrap_ci(y_true, y_pred, y_prob, n_iter: int = 1000, alpha: float = 0.05, seed: int = 42) -> Dict[str, Any]:
    """Selang kepercayaan bootstrap untuk sensitivity, specificity, dan ROC-AUC."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, y_prob = map(np.asarray, (y_true, y_pred, y_prob))
    vals = {"sensitivity": [], "specificity": [], "roc_auc": []}
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        t, p, s = y_true[idx], y_pred[idx], y_prob[idx]
        if t.sum() > 0:
            vals["sensitivity"].append(np.mean(p[t == 1] == 1))
        if (t == 0).sum() > 0:
            vals["specificity"].append(np.mean(p[t == 0] == 0))
        if 0 < t.sum() < n:
            vals["roc_auc"].append(roc_auc_score(t, s))
    out = {}
    for k, v in vals.items():
        out[k] = [round(float(np.quantile(v, alpha / 2)), 4), round(float(np.quantile(v, 1 - alpha / 2)), 4)] if v else None
    return out
