# 🫀AI-Powered Heart Disease Prediction System

## 📝 Project Overview
The **AI-Powered Heart Disease Prediction System** is a machine learning solution designed to assess the likelihood of heart disease in patients based on their medical history and physiological markers. By analyzing factors such as chest pain type, cholesterol levels, heart rate, and age, the system provides an early risk assessment to support healthcare decisions.

This project implements a **complete ML pipeline**: data ingestion, random sample imputation, variable transformation, class balancing (SMOTE), feature scaling, model training, evaluation, and deployment through a **Flask web application**.

---

## 🎯 Main Goal
- Assist medical professionals with a **rapid, data-driven screening tool**.  
- Identify high-risk patients for **early intervention** and better resource allocation.  

---

## 📁 Dataset Description
The dataset contains physiological parameters and medical test results:

| Feature | Description |
| :--- | :--- |
| **age** | Age in years |
| **sex** | Gender (1 = Male, 0 = Female) |
| **cp** | Chest pain type (0-3 scale) |
| **trestbps** | Resting blood pressure (mm Hg) |
| **chol** | Serum cholesterol (mg/dl) |
| **fbs** | Fasting blood sugar > 120 mg/dl (1 = True; 0 = False) |
| **restecg** | Resting ECG results (0,1,2) |
| **thalach** | Max heart rate achieved |
| **exang** | Exercise-induced angina (1 = Yes; 0 = No) |
| **oldpeak** | ST depression induced by exercise |
| **slope** | Slope of peak exercise ST segment |
| **ca** | Number of major vessels colored by fluoroscopy (0-3) |
| **thal** | Thalassemia (0-3) |
| **target** | Heart disease diagnosis (1 = Yes; 0 = No) |

### 🏷️ Dataset Categories
- **Target Variable:** `target` (Binary Classification)  
- **Continuous Features:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`  
- **Categorical/Discrete Features:** `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`  

---

## 🗂️ Project Structure

```text
Heart_Disease_Prediction/
│
├─ data/
│   └─ heart_disease.csv          # Source dataset
├─ app.py                         # Flask web app
├─ main.py                        # ML pipeline execution
├─ handling_missing.py            # Missing value imputation
├─ variable_transformation.py     # Outlier capping & log transform
├─ balance_data.py                # SMOTE for class imbalance
├─ Scaling.py                     # StandardScaler for features
├─ model_training.py              # Train & evaluate ML models
├─ log_code.py                    # Logging configuration
│
├─ models/
│   ├─ Heart_Disease_Prediction.pkl # Best trained model
│   ├─ scaler.pkl                   # Saved StandardScaler
│
├─ templates/
│   └─ index.html                 # Frontend HTML
├─ plot_path/                     # KDE plots & Boxplots
├─ requirements.txt               # Python dependencies
└─ README.md                      # Project documentation
```

## 🔄 ML Pipeline

### 📊 Data Visualization
- **Library:** Matplotlib & Seaborn  
- **Techniques:** KDE Plots, Boxplots  
- **Purpose:** Visualize distributions, detect skewness & outliers  

---

### 🛠️ Feature Engineering

#### 1️⃣ Handling Missing Values
- **Script:** `handling_missing.py`  
- **Technique:** Random Sample Imputation  
- **Method:** Fill missing values by randomly sampling existing valid data  
- **Reason:** Preserves statistical distribution better than mean/median  

#### 2️⃣ Variable Transformation
- **Script:** `variable_transformation.py`  
- **Techniques:**  
  - Outlier Capping: Chol, Thalach, Trestbps, Oldpeak → capped at 1.5×IQR  
  - Log Transformation: Chol, Oldpeak → `np.log1p` to reduce skewness  
- **Reason:** Normalizes distributions for better ML performance  

#### 3️⃣ Data Balancing
- **Script:** `balance_data.py`  
- **Technique:** SMOTE (Synthetic Minority Oversampling Technique)  
- **Method:** Generate synthetic samples for minority class  
- **Reason:** Prevents bias towards majority class  

#### 4️⃣ Feature Scaling
- **Script:** `Scaling.py`  
- **Technique:** StandardScaler  
- **Target Columns:** `age`, `trestbps`, `chol`, `thalach`, `oldpeak`  
- **Reason:** Ensures large-magnitude features do not dominate model learning  

---

### 🧠 Model Training & Selection
- **Script:** `model_training.py`  
- **Models Evaluated:** KNN, Naive Bayes, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost  
- **Evaluation Metric:** ROC-AUC  
- **Final Model:** Logistic Regression → saved as `Heart_Disease_Prediction.pkl`  

---

### 🌐 Deployment (Flask Web App)
- **Script:** `app.py`  
- **Frontend:** HTML form accepting inputs like Age, Cholesterol, BP, Heart Rate, etc.  
- **Backend:**  
  - Loads `Heart_Disease_Prediction.pkl` and `scaler.pkl`  
  - Scales numerical inputs  
  - Returns probability of heart disease  

---

### 📦 Install Required Packages
```bash
pip install -r requirements.txt
```

---

## **🚀 Run the Project**
> ```
> python main.py
> python app.py
> ```
---

## **👤 Author**
 ```
 Varadhana Varshini Kolipakula
 Machine Learning & Data Science Enthusiast
 ```

---
