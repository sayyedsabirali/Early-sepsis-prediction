import sys
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import S3ModelEstimator


# ============================================================
# Patient Input Wrapper
# ============================================================
class SepsisPatientData:
    def __init__(
        self,
        hr: float,
        map: float,
        rr: float,
        temp_c: float,
        spo2: float,
        creatinine: float,
        bilirubin: float,
        platelets: float,
        lactate: float,
        urine_output_6h_ml: float,
        anchor_age: int,
        gender: int,
    ):
        self.hr = hr
        self.map = map
        self.rr = rr
        self.temp_c = temp_c
        self.spo2 = spo2
        self.creatinine = creatinine
        self.bilirubin = bilirubin
        self.platelets = platelets
        self.lactate = lactate
        self.urine_output_6h_ml = urine_output_6h_ml
        self.anchor_age = anchor_age
        self.gender = gender

    def get_input_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "hr": [self.hr],
            "map": [self.map],
            "rr": [self.rr],
            "temp_c": [self.temp_c],
            "spo2": [self.spo2],
            "creatinine": [self.creatinine],
            "bilirubin": [self.bilirubin],
            "platelets": [self.platelets],
            "lactate": [self.lactate],
            "urine_output_6h_ml": [self.urine_output_6h_ml],
            "anchor_age": [self.anchor_age],
            "gender": [self.gender],
        })

        logging.info(f"Sepsis input dataframe created | shape={df.shape}")
        return df


# ============================================================
# Predictor
# ============================================================
class SepsisRiskPredictor:
    """
    Loads trained MyModel from S3
    Uses model.learned decision threshold
    """

    CLIP_LIMITS = {
        "hr": (30, 220),
        "map": (30, 130),
        "rr": (8, 60),
        "temp_c": (34, 42),
        "spo2": (50, 100),
        "creatinine": (0.2, 15),
        "bilirubin": (0.1, 20),
        "platelets": (5, 1000),
        "lactate": (0.5, 20),
        "urine_output_6h_ml": (0, 3000),
        "anchor_age": (18, 100),
    }

    def __init__(self, config: ModelPusherConfig = ModelPusherConfig()):
        try:
            self.estimator = S3ModelEstimator(
                bucket_name=config.bucket_name,
                model_path=config.model_registry_key
            )

            logging.info("Loading sepsis model from S3")
            self.model = self.estimator.load_model()

            logging.info(
                f"Model loaded | decision_threshold={self.model.decision_threshold:.4f}"
            )

        except Exception as e:
            raise MyException(e, sys)

    def _clip_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (low, high) in self.CLIP_LIMITS.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)
        return df

    def predict(self, input_df: pd.DataFrame) -> dict:
        """
        FINAL prediction method
        RETURNS DICT (single source of truth)
        """
        try:
            logging.info("Starting Sepsis prediction")

            input_df = self._clip_inputs(input_df)

            # 🔥 delegate to MyModel
            result = self.model.predict(input_df)

            return {
                "sepsis_risk_score": result["sepsis_risk_score"],
                "risk_label": result["risk_label"],
                "decision_threshold": result["decision_threshold"],
                "clinical_note": (
                    "This is an early warning score, not a diagnosis. "
                    "Always interpret with clinical judgment."
                )
            }

        except Exception as e:
            raise MyException(e, sys)
