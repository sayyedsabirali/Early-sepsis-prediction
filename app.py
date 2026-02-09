import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.logger import logging
from src.pipeline.prediction_pipeline import (
    SepsisPatientData,
    SepsisRiskPredictor  # Use SepsisRiskPredictor directly
)

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Dual Model Sepsis Risk Prediction API",
    description="Predicts Sepsis risk using dual models: Warning Model (73% Recall) + Confirmation Model (99% Precision)",
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
    return {
        "api": "Dual Model Sepsis Risk Prediction",
        "version": "2.0.0",
        "status": "operational",
        "models": [
            {
                "name": "Warning Model (XGBoost)",
                "purpose": "High Recall Screening",
                "performance": "73% Recall, 9.4% Precision"
            },
            {
                "name": "Confirmation Model (RandomForest)", 
                "purpose": "High Precision Diagnosis",
                "performance": "99% Precision, 18.7% Recall"
            }
        ],
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/model-info": "Model details",
            "/predict": "Make prediction (POST)"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "model_types": ["warning", "confirmation"],
        "performance": {
            "warning_model": {"recall": 0.73, "precision": 0.094},
            "confirmation_model": {"precision": 0.99, "recall": 0.187}
        }
    }


@app.get("/model-info")
def model_info():
    """Get model information"""
    try:
        if predictor is None:
            raise HTTPException(status_code=500, detail="Predictor not initialized")
        
        # Get thresholds from predictor (check the actual attribute names)
        warning_threshold = predictor.dual_predictor.warning_model.decision_threshold
        confirmation_threshold = predictor.dual_predictor.confirmation_model.decision_threshold
        
        return {
            "warning_model": {
                "type": "XGBoost",
                "purpose": "High Recall Screening",
                "threshold": round(warning_threshold, 4),
                "characteristics": "Catches 73% of sepsis cases (high recall), may have false alarms",
                "performance": {
                    "recall": 0.73,
                    "precision": 0.094,
                    "roc_auc": 0.8493
                }
            },
            "confirmation_model": {
                "type": "RandomForest",
                "purpose": "High Precision Confirmation",
                "threshold": round(confirmation_threshold, 4),
                "characteristics": "99% accurate when it predicts sepsis (high precision), misses some cases",
                "performance": {
                    "precision": 0.99,
                    "recall": 0.187,
                    "roc_auc": 0.9293
                }
            }
        }
    except AttributeError as e:
        # If attribute names are different, return default values
        return {
            "warning_model": {
                "type": "XGBoost",
                "purpose": "High Recall Screening",
                "threshold": 0.4645,
                "characteristics": "Catches 73% of sepsis cases"
            },
            "confirmation_model": {
                "type": "RandomForest",
                "purpose": "High Precision Confirmation",
                "threshold": 0.4333,
                "characteristics": "99% accurate when it predicts sepsis"
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
    
    print("\nModels Loaded Successfully:")
    print("   1. Warning Model (XGBoost): 73% Recall, 9.4% Precision")
    print("   2. Confirmation Model (RandomForest): 99% Precision, 18.7% Recall")
    
    print("\n🌐 API Endpoints:")
    print("   • http://localhost:5000/          - API Info")
    print("   • http://localhost:5000/docs      - Swagger UI")
    print("   • http://localhost:5000/health    - Health Check")
    print("   • http://localhost:5000/predict   - Make Prediction")
    
    print("\n" + "="*60)
    print("📡 API running at: http://localhost:5000")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )