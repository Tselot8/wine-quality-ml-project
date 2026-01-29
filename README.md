# Predicting Wine Quality Using Machine Learning

## Course
Machine Learning Lab – Final Course Project

## Project Type
Supervised Learning – Binary Classification

## Group Members
- Betanya Addisalem  – Data Collection & Initial Analysis and Model Testing  
- Bethlehem Woldekidan – Exploratory Data Analysis  
- Tselot Million – Data Preprocessing & Feature Engineering  
- Tsega Ephrem – Model Training & Hyperparameter Tuning  
- Meron Sisay – Model Evaluation & Documentation  

---

## 1. Project Overview
This project predicts the quality of red wine using supervised machine learning based on physicochemical properties. The goal is to automate wine quality assessment, which is traditionally subjective, using a data-driven approach.

---

## 2. Problem Definition & Motivation
Given numerical features of red wine, the model classifies wine as:
- **Good quality**
- **Bad quality**

**ML Type:** Classification  
**Motivation:** Improve consistency in quality control for the wine industry and demonstrate a real-world ML application.

---

## 3. Dataset Description
- **Dataset:** Wine Quality – Red Wine  
- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/Wine+Quality  
- **Samples:** 1599  
- **Features:** 11 numerical features  
- **Target:** Binary wine quality label  
- **Limitations:** Subjective expert ratings and region-specific data.

---

## 4. Project Structure
```bash
wine-quality-ml-project/
│
├── data/
│ ├── winequality-red.csv
│ ├── winequality_processed.csv
│ └── best_model.pkl
│
├── notebooks/
│ ├── 01_data_loading_and_inspection.ipynb
│ ├── 02_eda_and_visualization.ipynb
│ ├── 03_preprocessing_and_feature_engineering.ipynb
│ ├── 04_modeling_and_training.ipynb
│ ├── 05_evaluation_and_analysis.ipynb
│ └── 06_model_testing.ipynb
│
├── requirements.txt
├── README.md
└── report.pdf

```

---

## 5. Machine Learning Pipeline
1. Data loading and inspection  
2. Exploratory Data Analysis (EDA)  
3. Data preprocessing and feature engineering  
4. Model training and hyperparameter tuning  
5. Model evaluation and analysis  

---

## 6. Model & Evaluation
- **Model Used:** Random Forest Classifier  
- **Justification:** Handles non-linear relationships and performs well on numerical data  

**Evaluation Metrics:**
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC

---

## 7. Results Summary
The final model achieved:
- **Test Accuracy:** ~82%
- Balanced precision and recall for both classes
- Strong ROC-AUC performance, indicating good class separation

---

## 8. How to Use the Model

### Clone the Repository
```bash
git clone https://github.com/Tselot8/wine-quality-ml-project.git
cd wine-quality-ml-project

pip install -r requirements.txt
```
### Run the model

```bash
import joblib
import pandas as pd

model = joblib.load("data/best_model.pkl")
data = pd.read_csv("data/winequality_processed.csv")

predictions = model.predict(data.drop("quality_label", axis=1))
```
---

## 9. Deployment (Bonus – Local Streamlit Application)

As an optional bonus component, the trained model was deployed as a **local Streamlit web application**.  
The purpose of this deployment is to demonstrate model loading, inference, and basic user interaction through a simple web interface.

### Deployment Features
- Loads the trained Random Forest model (`best_model.pkl`)
- Allows users to input physicochemical properties of wine
- Predicts wine quality as **Good** or **Bad**
- Runs locally in a web browser

This satisfies the deployment requirement specified in the course guidelines using a **local interface**, which is sufficient for the bonus component.

### How to Run the Streamlit App

1. Navigate to the project root directory:
```bash
cd wine-quality-ml-project
```
2. Run the Streamlit application:
```bash 
streamlit run app/streamlit_app.py
```
## 10. Reproducibility & Code Quality

- Modular Jupyter notebooks
- Fixed random seeds
- All dependencies listed in requirements.txt
- GitHub used for version control and collaboration

 
