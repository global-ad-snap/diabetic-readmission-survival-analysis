# Model Card — Diabetes Readmission Survival Analysis

## Model Name
Diabetes Readmission Risk Survival Models (CoxPH & RSF)

## Model Purpose
To estimate the risk of 30-day hospital readmission among diabetic patients using survival analysis techniques for research-oriented decision-support prototyping.

## Training Data
- Dataset: Diabetes 1999–2008 Hospital Encounters
- Data Type: Structured administrative and clinical variables
- Sample Size: 99,549 encounters (after cohort filtering)
- Target Event: Readmission within 30 days
- Limitations: Retrospective data, simulated survival time, class imbalance

## Model Architecture
- Statistical Model: Cox Proportional Hazards
- Machine Learning Model: Random Survival Forest
- Frameworks: lifelines, scikit-survival
- Feature Engineering: Utilization history, demographics, discharge variables

## Evaluation Metrics
- C-index (CoxPH): 0.6375
- C-index (RSF): 0.7055
- Brier Score (RSF, t=176 days): 0.0976
- Validation: Internal validation only

## Performance Summary
The Random Survival Forest model demonstrated improved discrimination compared to the CoxPH baseline. Risk stratification based on model outputs showed clear separation in Kaplan–Meier survival curves across risk groups.

## Limitations
- Survival time simulated for demonstration purposes
- No external or prospective validation
- Dataset-specific coding practices and population bias

## Ethical Considerations
- Risk of misinterpretation without clinical oversight
- Operational bias due to healthcare utilization patterns
- Not validated for individual patient decision-making

## Intended Use Statement
These models are intended for research and educational purposes, including portfolio demonstration, and are not approved for clinical diagnosis, treatment decisions, or operational deployment without further validation.