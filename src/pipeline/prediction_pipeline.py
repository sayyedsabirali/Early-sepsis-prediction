import sys
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object
from src.utils.preprocessing_utils import PreprocessingUtils
from src.constants import (
    WARNING_THRESHOLD,
    CONFIRMATION_THRESHOLD,
    MODERATE_THRESHOLD
)



@dataclass
class SepsisPatientData:
    hr: float
    map: float
    rr: float
    temp_c: float
    spo2: float
    creatinine: float
    bilirubin: float
    platelets: float
    lactate: float
    urine_output_6h_ml: float
    anchor_age: int
    gender: int
    
    def get_input_dataframe(self) -> pd.DataFrame:
        """Convert patient data to dataframe"""
        try:
            patient_dict = {
                'hr': [self.hr],
                'map': [self.map],
                'rr': [self.rr],
                'temp_c': [self.temp_c],
                'spo2': [self.spo2],
                'creatinine': [self.creatinine],
                'bilirubin': [self.bilirubin],
                'platelets': [self.platelets],
                'lactate': [self.lactate],
                'urine_output_6h_ml': [self.urine_output_6h_ml],
                'anchor_age': [self.anchor_age],
                'gender': [self.gender]
            }
            
            return pd.DataFrame(patient_dict)
            
        except Exception as e:
            raise MyException(e, sys)


class DualModelPredictor:
    """
    Loads and uses both warning and confirmation models
    """
    
    def __init__(self, warning_model_path: str, confirmation_model_path: str):
        try:
            logging.info("Loading dual sepsis prediction models...")
            
            # Load both models
            self.warning_model = load_object(warning_model_path)
            self.confirmation_model = load_object(confirmation_model_path)
            
            logging.info(f"Warning Model threshold: {self.warning_model.decision_threshold:.4f}")
            logging.info(f"Confirmation Model threshold: {self.confirmation_model.decision_threshold:.4f}")
            logging.info("Both models loaded successfully")
            
        except Exception as e:
            raise MyException(e, sys)
    
    def _preprocess_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to input data"""
        try:
            # Apply feature engineering
            processed_df = PreprocessingUtils.apply_complete_feature_engineering(input_df)
            
            # Apply preprocessing transformations
            processed_df = PreprocessingUtils.apply_preprocessing_transformations(processed_df)
            
            return processed_df
            
        except Exception as e:
            raise MyException(e, sys)


class SepsisRiskPredictor:
    """
    Main predictor that uses both models for risk assessment
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
    
    def __init__(self):
        try:
            logging.info("Initializing SepsisRiskPredictor with dual models...")
            
            # Load dual model predictor
            self.dual_predictor = DualModelPredictor(
                warning_model_path="artifacts/model_trainer/trained_model/warning_model.pkl",
                confirmation_model_path="artifacts/model_trainer/trained_model/confirmation_model.pkl"
            )
            
            logging.info("Dual model predictor initialized successfully")
            
        except Exception as e:
            raise MyException(e, sys)
    
    def _clip_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip input values to physiological ranges"""
        for col, (low, high) in self.CLIP_LIMITS.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)
        return df
    
    def _get_risk_level(self, warning_score: float, confirmation_score: float) -> Tuple[str, str]:
        """Determine risk level based on both model scores"""
        
        warning_threshold = self.dual_predictor.warning_model.decision_threshold
        confirmation_threshold = self.dual_predictor.confirmation_model.decision_threshold
        
        # Decision Logic
        if confirmation_score >= confirmation_threshold:
            risk_level = "HIGH_RISK_CONFIRMED"
            risk_label = "🔴 HIGH RISK - CONFIRMED"
            
        elif warning_score >= warning_threshold:
            if confirmation_score >= MODERATE_THRESHOLD:
                risk_level = "MODERATE_RISK"
                risk_label = "🟡 MODERATE RISK"
            else:
                risk_level = "LOW_RISK_WARNING"
                risk_label = "🟢 LOW RISK - WARNING"
                
        else:
            risk_level = "LOW_RISK"
            risk_label = "LOW RISK"
        
        return risk_level, risk_label
    
    def _get_clinical_note(self, risk_level: str, 
                          warning_score: float, 
                          confirmation_score: float,
                          warning_threshold: float,
                          confirmation_threshold: float) -> str:
        """Generate clinical note based on predictions"""
        
        notes = {
            "HIGH_RISK_CONFIRMED": (
                f"HIGH RISK - CONFIRMED\n"
                f"• Warning Model Score: {warning_score:.1%} (threshold: {warning_threshold:.1%})\n"
                f"• Confirmation Model Score: {confirmation_score:.1%} (threshold: {confirmation_threshold:.1%})\n"
                f"• Clinical Action: IMMEDIATE INTERVENTION REQUIRED\n"
                f"• Recommendation: Start sepsis protocol, consider antibiotics, monitor closely"
            ),
            "MODERATE_RISK": (
                f"MODERATE RISK\n"
                f"• Warning Model Score: {warning_score:.1%} (threshold: {warning_threshold:.1%})\n"
                f"• Confirmation Model Score: {confirmation_score:.1%} (threshold: {confirmation_threshold:.1%})\n"
                f"• Clinical Action: CLOSE MONITORING\n"
                f"• Recommendation: Increase monitoring frequency, prepare interventions"
            ),
            "LOW_RISK_WARNING": (
                f"🔔 LOW RISK - WARNING\n"
                f"• Warning Model Score: {warning_score:.1%} (threshold: {warning_threshold:.1%})\n"
                f"• Confirmation Model Score: {confirmation_score:.1%} (threshold: {confirmation_threshold:.1%})\n"
                f"• Clinical Action: ROUTINE MONITORING\n"
                f"• Recommendation: Continue current care plan, monitor for changes"
            ),
            "LOW_RISK": (
                f"LOW RISK\n"
                f"• Warning Model Score: {warning_score:.1%} (threshold: {warning_threshold:.1%})\n"
                f"• Confirmation Model Score: {confirmation_score:.1%} (threshold: {confirmation_threshold:.1%})\n"
                f"• Clinical Action: ROUTINE CARE\n"
                f"• Recommendation: Standard monitoring protocol"
            )
        }
        
        return notes.get(risk_level, "Assessment pending")
    
    def predict(self, input_df: pd.DataFrame) -> Dict:
        """Make prediction using both models"""
        try:
            logging.info("Starting dual-model sepsis prediction")
            
            # Clip inputs to safe ranges
            input_df = self._clip_inputs(input_df)
            
            # Preprocess input
            processed_df = self.dual_predictor._preprocess_input(input_df)
            
            # Get predictions from both models
            warning_prob = self.dual_predictor.warning_model.trained_model_object.predict_proba(processed_df)[:, 1][0]
            confirmation_prob = self.dual_predictor.confirmation_model.trained_model_object.predict_proba(processed_df)[:, 1][0]
            
            # Get thresholds
            warning_threshold = self.dual_predictor.warning_model.decision_threshold
            confirmation_threshold = self.dual_predictor.confirmation_model.decision_threshold
            
            # Determine risk level
            risk_level, risk_label = self._get_risk_level(warning_prob, confirmation_prob)
            
            # Generate clinical note
            clinical_note = self._get_clinical_note(
                risk_level, warning_prob, confirmation_prob,
                warning_threshold, confirmation_threshold
            )
            
            # Prepare detailed breakdown
            model_breakdown = {
                "warning_model": {
                    "score": float(warning_prob),
                    "threshold": float(warning_threshold),
                    "prediction": bool(warning_prob >= warning_threshold),
                    "model_type": "high_recall_screening"
                },
                "confirmation_model": {
                    "score": float(confirmation_prob),
                    "threshold": float(confirmation_threshold),
                    "prediction": bool(confirmation_prob >= confirmation_threshold),
                    "model_type": "high_precision_confirmation"
                }
            }
            
            # Calculate consensus score (weighted average)
            consensus_score = (warning_prob * 0.4 + confirmation_prob * 0.6)
            
            return {
                "sepsis_risk_score": float(consensus_score),
                "risk_label": risk_label,
                "risk_level": risk_level,
                "warning_model_score": float(warning_prob),
                "confirmation_model_score": float(confirmation_prob),
                "warning_threshold": float(warning_threshold),
                "confirmation_threshold": float(confirmation_threshold),
                "clinical_note": clinical_note,
                "model_breakdown": model_breakdown,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            raise MyException(e, sys)


# For backward compatibility
def get_predictor() -> SepsisRiskPredictor:
    """Factory function to get predictor instance"""
    return SepsisRiskPredictor()