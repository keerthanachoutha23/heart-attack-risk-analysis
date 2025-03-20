# Heart Attack Risk Prediction - Final Report

## Introduction
Heart disease is a leading cause of mortality worldwide, and early prediction can help in risk mitigation and prevention. This project aims to build a **machine learning model** to predict **heart attack risk** based on health and lifestyle factors. The model is deployed using **Streamlit**, providing an interactive interface for real-time predictions.

## Data Description
The dataset consists of **9,651 patient records** with **27 features**, including:
- **Demographics**: Age, Gender
- **Lifestyle**: Smoking, Alcohol Consumption, Exercise Hours
- **Medical History**: Diabetes, Previous Heart Problems, Medication Use
- **Vital Signs**: Cholesterol, Heart Rate, Blood Pressure, BMI

The target variable:
- **Heart Attack Risk (Binary)** (`0` = Low Risk, `1` = High Risk)

### Data Preprocessing
- **Handling Missing Values**: Median imputation for numerical features, mode imputation for categorical features.
- **Feature Scaling**: Standardized numerical features using `StandardScaler()`.
- **Encoding**: Converted categorical features (e.g., Gender: Male = 1, Female = 0).

## Methodology
### Feature Selection
- Analyzed **feature correlations** using a heatmap.
- Identified key risk factors using **Random Forest feature importance**.

### Model Training
We trained and evaluated three models:
1. **Logistic Regression**
2. **Random Forest Classifier** ✅ Best Model
3. **XGBoost Classifier**

Each model was evaluated using:
- **Accuracy**
- **Precision & Recall**
- **AUC-ROC Score**

### Model Performance Comparison
| Model                | Accuracy | Precision | Recall | AUC-ROC |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression | 85.4%    | 0.82      | 0.79   | 0.87     |
| **Random Forest** ✅| **90.2%** | **0.89**  | **0.86** | **0.92** |
| XGBoost             | 88.7%    | 0.86      | 0.83   | 0.90     |

### Model Deployment
- The **best-performing Random Forest model** was saved as `heart_attack_model.pkl`.
- The **scaler** was saved as `scaler.pkl`.
- A **Streamlit app** (`app.py`) was created for real-time predictions.

## Results & Discussion
- **Random Forest outperformed all models** in terms of accuracy and AUC-ROC.
- **Key risk factors** include **Cholesterol, BMI, Systolic Blood Pressure, and Diabetes**.
- The **Streamlit deployment** allows easy accessibility for users.

## Conclusion & Future Work
### Key Takeaways
- Machine learning models can effectively predict heart attack risk.
- **Lifestyle & medical factors** play a crucial role in heart disease risk.
- A user-friendly **web app** enhances accessibility for healthcare professionals.

### Future Enhancements
- Integrate **real-time patient data** for continuous monitoring.
- Deploy the model using **Flask or FastAPI** for wider scalability.
- Improve model interpretability using **SHAP (Explainable AI Techniques)**.

## References
- WHO Heart Disease Reports, 2024
- Machine Learning for Healthcare, Stanford University

