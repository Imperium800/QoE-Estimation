# QoE Estimation in Mobile Radio Networks

This project presents a machine learning pipeline designed to predict **Quality of Experience (QoE)** for mobile users based on session-level data over **UMTS and LTE networks**. The objective is to classify user satisfaction during video streaming sessions (e.g., YouTube), enabling telecom operators to detect and respond to poor user experiences in near real-time.

---

## Project Summary

- **Problem**: Identify unsatisfied users in mobile networks based on technical indicators.
- **Approach**: Supervised classification using engineered features and model ensembles.
- **Tech Stack**: Python, scikit-learn, XGBoost, SHAP, scikit-optimize.
- **Deployment**: Modular pipeline with preprocessing, classification, thresholding, and drift monitoring.

---

## Objectives

- Predict binary **User Satisfaction** (0 = Satisfied, 1 = Unsatisfied).
- Handle class imbalance and high-dimensional skewed data.
- Maximize **AUC** while maintaining low **False Positive Rate (FPR)**.
- Deploy a reliable model capable of retraining on updated data.

---

## Dataset Overview

- **Source**: Anonymized 30-day mobile session logs.
- **Technologies**: UMTS and LTE.
- **Features**: Download time/volume, signal strength, service coverage, engineered ratios.
- **Label Distribution**:
  - Class 0 (Satisfied): 67.3%
  - Class 1 (Unsatisfied/Alarm): 32.7%

---

## Pipeline Architecture

1. **Data Preprocessing**
2. **Feature Engineering**
   - Ratio-based indicators
   - Service time shares
   - Log-transformed counters
3. **Model Training**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost (final model)
4. **Evaluation**
   - 5-fold stratified cross-validation
   - ROC-AUC, Precision, Recall, F1-score
5. **Deployment**
   - Thresholding (p = 0.608) to control FPR
   - Drift detection (KS-test)
   - Automated retraining

---

## Model Performance

| Model           | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|----------------|---------|--------|-----------|--------|----------|
| Logistic Reg.   | 0.651   |   —    |   0.61    |  0.26  |  0.41    |
| Random Forest   | 0.724   |   —    |   0.64    |  0.35  |  0.45    |
| **XGBoost**     | **0.727** | **0.56** | **0.61** | **0.46** | **0.52** |

---

## Explainability and Monitoring

- **SHAP Analysis**: Identifies impactful features such as:
  - `UMTS_LimSvc_Share`
  - `Cumulative_YoutubeSess_LTE_DL_Volume`
  - `Max_SNR`
- **Drift Detection**: Weekly data batches evaluated using Kolmogorov-Smirnov test. Retraining is triggered if 3 or more features drift significantly.
- **Retraining Module**: Automatically fits a new model and updates the production pipeline.
