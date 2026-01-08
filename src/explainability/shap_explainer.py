import sys
import shap
import numpy as np
import pandas as pd

from src.exception import MyException
from src.logger import logging


class SepsisSHAPExplainer:
    """
    Doctor-facing SHAP explainability for Sepsis Prediction
    Works with XGBoost models
    """

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_global(self, X_sample: np.ndarray):
        """
        Global explanation:
        Shows which features contribute most to sepsis risk overall
        """
        try:
            logging.info("Generating GLOBAL SHAP explanation")
            shap_values = self.explainer.shap_values(X_sample)

            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                show=False
            )

            logging.info("Global SHAP explanation generated successfully")

        except Exception as e:
            raise MyException(e, sys)

    def explain_patient(self, X_sample: np.ndarray, patient_index: int):
        """
        Patient-level explanation:
        Shows why ONE patient was predicted high/low risk
        """
        try:
            logging.info(f"Generating SHAP explanation for patient {patient_index}")

            shap_values = self.explainer.shap_values(X_sample)

            shap.force_plot(
                self.explainer.expected_value,
                shap_values[patient_index],
                X_sample[patient_index],
                feature_names=self.feature_names,
                matplotlib=True
            )

            logging.info("Patient-level SHAP explanation generated")

        except Exception as e:
            raise MyException(e, sys)

    def get_shap_dataframe(self, X_sample: np.ndarray) -> pd.DataFrame:
        """
        Returns SHAP values as DataFrame (for logging / audit / saving)
        """
        try:
            shap_values = self.explainer.shap_values(X_sample)

            shap_df = pd.DataFrame(
                shap_values,
                columns=self.feature_names
            )

            return shap_df

        except Exception as e:
            raise MyException(e, sys)
