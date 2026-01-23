import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from src.logger import logging
from src.pipeline.prediction_pipeline import (
    SepsisPatientData,
    SepsisRiskPredictor
)

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Sepsis Risk Prediction API",
    description="Predicts Sepsis risk 12 hours in advance (MIMIC-IV ICU)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# 🔥 LOAD MODEL ONCE
predictor = SepsisRiskPredictor()

# ============================================================
# Request Schema
# ============================================================
class SepsisPatientRequest(BaseModel):
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


# ============================================================
# Response Schema
# ============================================================
class SepsisPredictionResponse(BaseModel):
    sepsis_risk_score: float
    risk_label: str
    decision_threshold: float
    note: str


# ============================================================
# Routes
# ============================================================
@app.get("/")
def root():
    ui_path = BASE_DIR / "index.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    return {"message": "Sepsis Risk Prediction API running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=SepsisPredictionResponse)
def predict_sepsis(patient: SepsisPatientRequest):
    try:
        logging.info("Received sepsis prediction request")

        patient_data = SepsisPatientData(**patient.dict())
        input_df = patient_data.get_input_dataframe()

        result = predictor.predict(input_df)

        return {
            "sepsis_risk_score": result["sepsis_risk_score"],
            "risk_label": result["risk_label"],
            "decision_threshold": result["decision_threshold"],
            "note": result["clinical_note"]
        }

    except Exception as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
