import sys
import pandas as pd
from typing import Optional

from src.exception import MyException
from src.logger import logging
from src.utils.preprocessing_utils import PreprocessingUtils


class MyModel:
    """
    FINAL inference wrapper for both warning and confirmation models
    """

    def __init__(
        self,
        trained_model_object: object,
        decision_threshold: float,
        model_type: str = "warning"  # "warning" or "confirmation"
    ):
        self.trained_model_object = trained_model_object
        self.decision_threshold = decision_threshold
        self.model_type = model_type

    def predict_proba(self, dataframe: pd.DataFrame):
        try:
            # 🔥 SAME preprocessing as training
            df_processed = PreprocessingUtils.apply_preprocessing_transformations(
                dataframe
            )

            prob = self.trained_model_object.predict_proba(
                df_processed.values
            )[:, 1]

            return prob

        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        try:
            prob = float(self.predict_proba(dataframe)[0])
            pred = int(prob >= self.decision_threshold)

            if self.model_type == "warning":
                risk_label = "⚠️ WARNING - Potential Sepsis Risk" if pred == 1 else "✅ LOW RISK (Screening)"
            else:
                risk_label = "🔴 CONFIRMED - High Sepsis Risk" if pred == 1 else "🟡 NOT CONFIRMED"

            logging.info(
                f"Inference | Model: {self.model_type} | "
                f"prob={prob:.4f} | threshold={self.decision_threshold:.3f} | "
                f"label={pred} | {risk_label}"
            )

            return {
                "prediction": pred,
                "sepsis_risk_score": round(prob, 4),
                "risk_label": risk_label,
                "decision_threshold": round(self.decision_threshold, 4),
                "model_type": self.model_type
            }

        except Exception as e:
            raise MyException(e, sys)