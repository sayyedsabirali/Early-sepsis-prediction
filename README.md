# 🏥 Early Sepsis Prediction – 12 Hour Forecasting 

This repository contains a production-grade machine learning pipeline for the **early detection of sepsis** in intensive care units. This project operationalizes the **Sepsis-3 clinical definition** and focuses on a **1-12 hour predictive window** to provide actionable clinical lead time.

---

## 📌 Project Objective

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection.

> **Clinical Logic:** Suspected Infection + Acute Change in Sequential Organ Failure Assessment (SOFA) Score $\ge 2$.

**The Challenge:**
* **Predictive Lead Time:** Predicting sepsis at the moment of onset is often too late for optimal intervention. We target a **1–12 hour forecasting window**.
* **Data Leakage:** Strict temporal boundaries ensure no "future" information (from the moment of diagnosis) is used during training.
* **Class Imbalance:** Sepsis hours represent only ~1.66% of the dataset, making Precision and Recall more critical than Accuracy.

---

## 🗂 Data Source & Cohort Selection

### Database
* **Source:** MIMIC-IV (v3.1).
* **Infrastructure:** BigQuery (SQL) for extraction and Python for processing.
* **Unit of Analysis:** **ICU-Hour**. Every hour of a patient's stay is a unique data point.

### Cohort Filtering
1.  **Stay Duration:** ICU stays $\ge$ 48 hours (ensures sufficient history for 12h lookback).
2.  **Age:** Adult patients only ($\ge$ 18 years).
3.  **Scale:** * **46,337** unique ICU stays.
    * **6,937,673** total hourly observations.
    * **Median Length of Stay (LOS):** 76.8 hours.

---

## 📊 Feature Engineering & Clinical Preprocessing

### 1. Vital Signs (Worst-Value Aggregation)
Capturing physiological instability by taking the most extreme values per hour.
| Feature | Range Filtering | Description |
| :--- | :--- | :--- |
| **Heart Rate (HR)** | 20 – 300 bpm | Detecting tachycardia/bradycardia. |
| **MAP** | 30 – 200 mmHg | Mean Arterial Pressure (key for Cardio SOFA). |
| **Resp Rate (RR)** | 4 – 60 bpm | Detecting respiratory distress. |
| **Temp (°C)** | 30 – 45 °C | Standardized to Celsius. |
| **SpO2** | 50 – 100% | Oxygen saturation levels. |

### 2. Sepsis-3 Component Logic
* **SOFA Calculation:** Dynamically computed every hour for Renal (Creatinine), Liver (Bilirubin), Coagulation (Platelets), and Cardiovascular (MAP + Vasopressors) systems.
* **Baseline:** The **Running Minimum** SOFA score per stay is used as the patient's individual baseline.
* **Infection Suspicion:** Flagged via Culture Collection and Antibiotic Administration within a $\pm$24h window.

---

## 🧮 Labeling Strategy: The 12-Hour Forecast

1.  **Onset Hour ($t_{onset}$):** First hour where $\Delta \text{SOFA} \ge 2$ + Infection Suspicion.
2.  **Positive Window ($t_{onset} - 12$ to $t_{onset}$):** Labeled `1`.
3.  **Negative Window:** All hours prior to the warning window are labeled `0`.
4.  **Leakage Prevention:** All data following $t_{onset}$ is discarded to ensure a purely predictive task.

---

## 📊 Model Performance Comparison (Test Set)



| Model | Test Precision | Test Recall | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **ExtraTrees_100** | 0.9872 | 0.4565 | **0.9501** |
| **RandomForest_100** | **0.9961** | 0.1669 | 0.9159 |
| **XGBoost** | 0.0751 | 0.7518 | 0.8522 |
| **CatBoost** | 0.0715 | 0.7300 | 0.8371 |
| **HistGradientBoosting** | 0.2440 | 0.0096 | 0.8355 |
| **LightGBM** | 0.0666 | 0.7163 | 0.8187 |
| **AdaBoost** | 0.0000 | 0.0000 | 0.7858 |
| **NaiveBayes** | 0.0595 | 0.0648 | 0.6998 |
| **DecisionTree** | 0.4285 | 0.3799 | 0.6844 |
| **LogisticRegression_SGD** | 0.0220 | **0.9256** | 0.5086 |
| **LinearSVC** | 0.0336 | 0.6935 | - |

---

## 🏗 Two-Stage Clinical System Strategy

To manage the trade-off between "missing a case" and "triggering false alarms," we utilize a two-model approach:

### Stage 1: The Warning Model (High Recall)
* **Model:** XGBoost
* **Metric:** 75.18% Recall
* **Goal:** Act as a "Screening" layer to identify as many potential sepsis cases as possible 12 hours early.

### Stage 2: The Confirmation Model (High Precision)
* **Model:** ExtraTrees / RandomForest
* **Metric:** 98.7% - 99.6% Precision
* **Goal:** Act as a "Verification" layer. Only if this model also flags the patient does the system trigger a high-priority alert to the clinical team, effectively eliminating false alarm fatigue.

---

## 🏗 System Architecture & MLOps

### 1. Technology Stack
* **Storage:** BigQuery / Parquet.
* **Training:** Scikit-learn, XGBoost, CatBoost.
* **Tracking:** **MLflow** for hyperparameter logging and model versioning.
* **Deployment:** **FastAPI** for real-time inference.

### 2. CI/CD Pipeline

The project utilizes a self-hosted GitHub Actions runner on AWS:
1.  **Commit:** Code pushed to `main`.
2.  **Build:** Docker image created with pre-trained model weights.
3.  **Registry:** Image pushed to **AWS ECR**.
4.  **Deploy:** **EC2** pulls the latest image and restarts the FastAPI service.

---

## ⚠️ Limitations & Notes
* **Missing Data:** Lab values are forward-filled. If no lab exists, SOFA is assumed to be 0.
* **ICU Specific:** The model is trained on MIMIC-IV and may require fine-tuning for non-ICU (general ward) settings.
* **Recall vs. Precision:** In a clinical setting, a false positive costs time, but a false negative costs a life. The thresholding is adjustable based on hospital policy.

---

## 📌 Summary
This system provides a clinically robust, end-to-end solution for sepsis forecasting. By integrating Sepsis-3 logic with a high-performance MLOps pipeline, it moves beyond simple classification into the realm of real-time clinical decision support.
