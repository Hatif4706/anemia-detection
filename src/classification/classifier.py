import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb

from src.evaluation.metrics import EvaluationMetrics

class AnemiaClassifierBenchmark:
    """
    Downstream Classifier Engine terstandarisasi untuk memastikan
    komparasi yang setara (fair comparison) antara berbagai metode ekstraksi ROI.
    """

    SUPPORTED_MODELS = ["xgboost", "svm_rbf", "random_forest"]

    def __init__(self, model_name: str = "xgboost", random_state: int = 42):
        self.model_name = model_name.lower()
        self.random_state = random_state
        self.pipeline = self._build_model_pipeline()
        self.feature_names: List[str] = []

    def _build_model_pipeline(self) -> Pipeline:
        if self.model_name == "xgboost":
            clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss"
            )
        elif self.model_name == "svm_rbf":
            clf = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=self.random_state
            )
        elif self.model_name == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Model {self.model_name} tidak didukung. Pilih dari {self.SUPPORTED_MODELS}")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])

    def evaluate_cv(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Melakukan evaluasi Stratified K-Fold Cross Validation bebas data leakage.
        
        Args:
            X: DataFrame berisi fitur numerik (Erythema index, a*, HSV, GLCM, dll)
            y: Array label biner (1 = Anemia, 0 = Non-Anemia)
            n_splits: Jumlah fold cross-validation
            
        Returns:
            Dictionary hasil evaluasi metrik diagnostik.
        """
        self.feature_names = list(X.columns)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Prediksi out-of-fold probabilities dan kelas
        y_proba = cross_val_predict(self.pipeline, X, y, cv=skf, method="predict_proba")
        y_pred = np.argmax(y_proba, axis=1)

        metrics = EvaluationMetrics.calculate_classification_metrics(y, y_pred, y_proba)
        metrics["model_name"] = self.model_name
        metrics["n_samples"] = len(y)
        metrics["n_features"] = X.shape[1]

        # Latih model final pada seluruh data untuk feature importance
        self.pipeline.fit(X, y)
        metrics["feature_importance"] = self._get_feature_importances()

        return metrics

    def _get_feature_importances(self) -> Dict[str, float]:
        clf = self.pipeline.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            return {
                feat: float(imp) 
                for feat, imp in sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
            }
        return {}
