import sys
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.utils.preprocessing_utils import PreprocessingUtils


class MyModel:
    """
    FINAL inference wrapper (NO sklearn transformer)

    - Uses same preprocessing utils as training
    - Uses trained XGBoost model
    - Uses learned decision threshold
    """

    def __init__(
        self,
        trained_model_object: object,
        decision_threshold: float
    ):
        self.trained_model_object = trained_model_object
        self.decision_threshold = decision_threshold

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

            logging.info(
                f"Inference | prob={prob:.4f} | "
                f"threshold={self.decision_threshold:.3f} | "
                f"label={pred}"
            )

            return {
                "prediction": pred,
                "sepsis_risk_score": round(prob, 4),
                "risk_label": (
                    "HIGH RISK OF SEPSIS (within 12h)"
                    if pred == 1 else
                    "LOW RISK OF SEPSIS (within 12h)"
                ),
                "decision_threshold": round(self.decision_threshold, 4)
            }

        except Exception as e:
            raise MyException(e, sys)
