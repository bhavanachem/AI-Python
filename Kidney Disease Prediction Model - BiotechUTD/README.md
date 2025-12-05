# Chronic Kidney Disease (CKD) Prediction Model

A hands-on machine learning project designed as a practical application exercise for students who completed the Protein Structure Classification workshop series at Biotech Club, UT Dallas.

## Project Overview

This project applies machine learning techniques to predict whether a patient has Chronic Kidney Disease (CKD) based on various clinical and laboratory test results. Students use the concepts learned from the protein classification workshop to work through a real-world healthcare prediction problem.

**Dataset**: 400 patient records with 24 clinical features  
**Task**: Binary classification (CKD vs. Not CKD)  
**Models**: Stochastic Gradient Descent (SGD), Decision Tree, Random Forest

## Repository Structure

This repository contains **two versions** of the project:

### 1. **Student Version** (Redacted - `Kidney_Disease_Prediction_Student.ipynb`)
A partially complete notebook with `# YOUR CODE HERE` sections where students apply what they learned from the protein structure workshop series. Includes helpful hints and references to guide independent problem-solving.

**Purpose**: Practice applying ML concepts independently  
**Target Audience**: Students completing the Biotech Club workshop series  
**Features**: 
- Clear section markers for student completion
- Hints pointing to protein classification notebook for reference
- Guided structure following the workshop progression

### 2. **Complete Version** (Full Solution - `Kidney_Disease_Prediction_Complete.ipynb`)
The fully implemented notebook with all code completed, serving as a reference solution and teaching tool.

**Purpose**: Reference solution and learning resource  
**Target Audience**: Instructors and students checking their work  
**Features**:
- Complete implementation of all analysis steps
- Detailed comments explaining each decision
- Production-ready prediction function

## Dataset Features

### Clinical Measurements (Continuous)
- Age
- Blood pressure
- Specific gravity
- Blood glucose (random)
- Blood urea
- Serum creatinine
- Sodium
- Potassium
- Haemoglobin
- Packed cell volume
- White blood cell count
- Red blood cell count

### Laboratory Tests (Categorical)
- Albumin levels
- Sugar levels
- Red blood cells (normal/abnormal)
- Pus cells (normal/abnormal)
- Pus cell clumps (present/not present)
- Bacteria (present/not present)

### Medical History (Binary)
- Hypertension
- Diabetes mellitus
- Coronary artery disease
- Appetite (good/poor)
- Pedal edema
- Anemia

## Concepts Applied from Workshop Series

Students apply these concepts from the protein structure workshop:

### From Week 1 (EDA)
- Loading and inspecting datasets with pandas
- Understanding data shape and structure
- Using `.value_counts()` to understand distributions
- Identifying missing values with `.isnull().sum()`
- Creating visualizations with matplotlib and seaborn

### From Week 2 (Preprocessing)
- Handling missing values (various imputation strategies)
- Encoding categorical variables (Label Encoding, One-Hot Encoding)
- Feature scaling with StandardScaler
- Train-test splitting with stratification

### From Week 3 (Model Training)
- Training multiple classifiers (SGD, Decision Tree, Random Forest)
- Understanding different algorithms and their trade-offs
- Evaluating models with multiple metrics
- Interpreting classification reports and confusion matrices

### From Week 4 (Model Comparison)
- Comparing model performance across metrics
- Understanding precision, recall, F1-score trade-offs
- Selecting the best model for deployment
- Considering model complexity vs. performance

### From Final Workshop (Prediction Interface)
- Building user-friendly prediction functions
- Handling input preprocessing for new data
- Feature engineering for inference
- Creating interpretable outputs with confidence scores


## Model Performance

The Random Forest classifier achieves:
- **Accuracy**: ~97-98%
- **Precision**: High for both classes
- **Recall**: Strong detection of CKD cases
- **F1-Score**: Balanced performance

Performance metrics demonstrate the model's reliability for this binary classification task.

## Making Predictions

After training, use the prediction function with clinical values:

```python
predict_kidney_disease(
    age=60,
    blood_pressure=100,
    specific_gravity=1.005,
    blood_glucose=200,
    blood_urea=80,
    serum_creatinine=5.0,
    haemoglobin=8.0
)
```

## Clinical Significance

Key indicators of CKD from the model:
- **Serum Creatinine**: Elevated levels indicate reduced kidney function
- **Blood Urea**: High levels suggest kidney dysfunction
- **Haemoglobin**: Lower levels often associated with CKD
- **Specific Gravity**: Abnormal values indicate filtering issues
- **Medical History**: Hypertension and diabetes are major risk factors

## Extension Ideas

Students can extend this project by:
- Implementing hyperparameter tuning (Grid Search or Random Search)
- Adding feature importance analysis
- Creating additional visualizations (ROC curves, feature distributions)
- Experimenting with other algorithms (XGBoost, SVM, Neural Networks)
- Building a web interface with Streamlit or Flask
- Analyzing feature correlations with target variable
