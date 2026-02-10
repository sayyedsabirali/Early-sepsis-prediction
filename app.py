import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import json
import os

from src.logger import logging
from src.pipeline.prediction_pipeline import SepsisPatientData, SepsisRiskPredictor

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Dual Model Sepsis Risk Prediction API",
    description="Predicts Sepsis risk using dual models: Warning Model (High Recall) + Confirmation Model (High Precision)",
    version="2.0.0", 
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# 🔥 LOAD DUAL MODEL PREDICTOR
try:
    predictor = SepsisRiskPredictor()  # Direct initialization
    logging.info("Dual Model Predictor loaded successfully!")
    
    # Get actual thresholds for logging
    warning_threshold = predictor.dual_predictor.warning_model.decision_threshold
    confirmation_threshold = predictor.dual_predictor.confirmation_model.decision_threshold
    
    logging.info(f"Warning Model threshold: {warning_threshold:.4f}")
    logging.info(f"Confirmation Model threshold: {confirmation_threshold:.4f}")
    
except Exception as e:
    logging.error(f"❌ Failed to load predictor: {e}")
    predictor = None

# ============================================================
# Request Schema
# ============================================================
class SepsisPatientRequest(BaseModel):
    hr: float = Field(..., ge=30, le=220, description="Heart rate (bpm)")
    map: float = Field(..., ge=30, le=130, description="Mean arterial pressure (mmHg)")
    rr: float = Field(..., ge=8, le=60, description="Respiratory rate (breaths/min)")
    temp_c: float = Field(..., ge=34, le=42, description="Temperature (°C)")
    spo2: float = Field(..., ge=50, le=100, description="Oxygen saturation (%)")
    creatinine: float = Field(..., ge=0.2, le=15, description="Creatinine (mg/dL)")
    bilirubin: float = Field(..., ge=0.1, le=20, description="Bilirubin (mg/dL)")
    platelets: float = Field(..., ge=5, le=1000, description="Platelets (K/μL)")
    lactate: float = Field(..., ge=0.5, le=20, description="Lactate (mmol/L)")
    urine_output_6h_ml: float = Field(..., ge=0, le=3000, description="Urine output (ml/6h)")
    anchor_age: int = Field(..., ge=18, le=100, description="Age (years)")
    gender: int = Field(..., ge=0, le=1, description="Gender (0: Female, 1: Male)")


# ============================================================
# Response Schema
# ============================================================
class SepsisPredictionResponse(BaseModel):
    sepsis_risk_score: float
    risk_label: str
    risk_level: str
    warning_model_score: float
    confirmation_model_score: float
    warning_threshold: float
    confirmation_threshold: float
    clinical_note: str
    model_breakdown: dict
    timestamp: str


# ============================================================
# Routes
# ============================================================
@app.get("/")
def root():
    """Root endpoint with API info"""
    try:
        if predictor is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Get actual thresholds
        warning_threshold = predictor.dual_predictor.warning_model.decision_threshold
        confirmation_threshold = predictor.dual_predictor.confirmation_model.decision_threshold
        
        return {
            "api": "Dual Model Sepsis Risk Prediction",
            "version": "2.0.0",
            "status": "operational",
            "thresholds": {
                "warning_model": round(warning_threshold, 4),
                "confirmation_model": round(confirmation_threshold, 4)
            },
            "models": [
                {
                    "name": "Warning Model (XGBoost)",
                    "purpose": "High Recall Screening",
                    "threshold": round(warning_threshold, 4)
                },
                {
                    "name": "Confirmation Model (ExtraTrees)", 
                    "purpose": "High Precision Diagnosis",
                    "threshold": round(confirmation_threshold, 4)
                }
            ],
            "endpoints": {
                "/": "API information",
                "/health": "Health check",
                "/model-info": "Model details",
                "/predict": "Make prediction (POST)"
            }
        }
    except Exception as e:
        return {
            "api": "Dual Model Sepsis Risk Prediction",
            "version": "2.0.0",
            "status": "error",
            "error": str(e)
        }


@app.get("/health")
def health():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Get actual thresholds
        warning_threshold = predictor.dual_predictor.warning_model.decision_threshold
        confirmation_threshold = predictor.dual_predictor.confirmation_model.decision_threshold
        
        return {
            "status": "healthy",
            "models_loaded": True,
            "model_types": ["warning", "confirmation"],
            "thresholds": {
                "warning_model": round(warning_threshold, 4),
                "confirmation_model": round(confirmation_threshold, 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.get("/model-info")
def model_info():
    """Get model information"""
    try:
        if predictor is None:
            raise HTTPException(status_code=500, detail="Predictor not initialized")
        
        # Get thresholds from predictor
        warning_threshold = predictor.dual_predictor.warning_model.decision_threshold
        confirmation_threshold = predictor.dual_predictor.confirmation_model.decision_threshold
        
        # Try to load actual performance metrics from saved file
        metrics_path = "artifacts/evaluation/dual_model_metrics.json"
        performance_data = {}
        
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    if "warning_model" in metrics and "confirmation_model" in metrics:
                        performance_data = metrics
            except:
                pass
        
        return {
            "warning_model": {
                "type": "XGBoost",
                "purpose": "High Recall Screening",
                "threshold": round(warning_threshold, 4),
                "characteristics": "Optimized for high recall to catch potential sepsis cases",
                "performance": performance_data.get("warning_model", {
                    "recall": "Not available",
                    "precision": "Not available",
                    "roc_auc": "Not available"
                })
            },
            "confirmation_model": {
                "type": "ExtraTrees",
                "purpose": "High Precision Confirmation", 
                "threshold": round(confirmation_threshold, 4),
                "characteristics": "Optimized for high precision to confirm sepsis cases",
                "performance": performance_data.get("confirmation_model", {
                    "precision": "Not available",
                    "recall": "Not available", 
                    "roc_auc": "Not available"
                })
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=SepsisPredictionResponse)
def predict_sepsis(patient: SepsisPatientRequest):
    """Make sepsis risk prediction using dual models"""
    try:
        if predictor is None:
            raise HTTPException(status_code=500, detail="Models not loaded. Please restart the API.")
        
        logging.info("Received dual-model sepsis prediction request")
        
        # Create patient data object
        patient_data = SepsisPatientData(**patient.dict())
        input_df = patient_data.get_input_dataframe()
        
        # Get prediction
        result = predictor.predict(input_df)
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/ui")
def serve_ui():
    """Serve HTML UI"""
    ui_path = BASE_DIR / "templates" / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "UI not available. Please create templates/index.html"}


# ============================================================
# Example endpoint
# ============================================================
@app.get("/example")
def get_example():
    """Get example patient data for testing"""
    return {
        "example_patient": {
            "hr": 112,
            "map": 68,
            "rr": 24,
            "temp_c": 38.4,
            "spo2": 92,
            "creatinine": 2.1,
            "bilirubin": 1.8,
            "platelets": 95,
            "lactate": 3.6,
            "urine_output_6h_ml": 210,
            "anchor_age": 64,
            "gender": 1
        },
        "note": "This is a high-risk sepsis patient example"
    }


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING DUAL MODEL SEPSIS PREDICTION API")
    print("="*60)
    
    if predictor is None:
        print("ERROR: Failed to load predictor. Check if models are trained.")
        sys.exit(1)
    
    # Get actual thresholds for display
    try:
        warning_threshold = predictor.dual_predictor.warning_model.decision_threshold
        confirmation_threshold = predictor.dual_predictor.confirmation_model.decision_threshold
        
        print(f"\nModels Loaded Successfully:")
        print(f"   1. Warning Model (XGBoost): Threshold = {warning_threshold:.4f}")
        print(f"   2. Confirmation Model (ExtraTrees): Threshold = {confirmation_threshold:.4f}")
        
    except Exception as e:
        print(f"\nModels Loaded Successfully (threshold info not available)")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )