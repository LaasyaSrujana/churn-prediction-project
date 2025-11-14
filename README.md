#  Customer Churn Prediction – End-to-End Machine Learning Project

This repository contains an **end-to-end machine learning project** to predict **customer churn** for a telecom company.  
It includes:

- **Data preprocessing**
- **Model training (XGBoost + LightGBM Ensemble)**
- **FastAPI backend for real-time predictions**
- **Streamlit UI for user-friendly interaction**
- **Model artifacts saved for production use**

This project is designed to look professional and industry-oriented, making it perfect for resumes, GitHub portfolios, and interviews.

---

##  Project Overview

Customer churn refers to customers leaving a service.  
Telecom companies lose significant revenue when customers switch to other service providers.

This ML system predicts whether a customer will **Churn** or **Not Churn** using features like:

- Contract type  
- Monthly charges  
- Internet service  
- Tenure  
- Payment method  
- Support services  
- Customer demographics  

---

##  Tech Stack

| Component | Technology |
|----------|------------|
| Programming | Python |
| Frameworks | FastAPI, Streamlit |
| ML Models | XGBoost, LightGBM, Voting Ensemble |
| Preprocessing | LabelEncoder, StandardScaler |
| Serving | Uvicorn |
| Storage | Joblib |

---

##  Project Structure

churn-prediction-project
│
├── data/
│ └── raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│ ├── churn_model.joblib
│ ├── scaler.joblib
│ ├── label_encoders.joblib
│ └── feature_order.joblib
│
├── src/
│ ├── train.py → Train the ML model
│ ├── app.py → FastAPI backend
│ └── ui.py → Streamlit user interface
│
└── README.md


---

##  Model Performance

The Ensemble model achieved:

- **Accuracy:** 0.88  
- Strong recall and precision for both churn and non-churn classes  

          precision    recall  f1-score   support
       0       0.93      0.82      0.87      1044
       1       0.84      0.93      0.88      1026


---

##  How to Run the Project

### 1️. Create Virtual Environment
python -m venv .venv

### 2️. Activate it
Windows
.venv\Scripts\activate

### 3️. Install dependencies
pip install -r requirements.txt

### 4️.  Train the Machine Learning Model
python src/train.py

### 5️.  Start FastAPI Server
python src/app.py
Access Swagger API Docs
http://127.0.0.1:8000/docs
 
 ### 6️. Start Streamlit UI
streamlit run src/ui.py

Prediction Output
 Sample Prediction Result
<img width="1919" height="930" alt="Image" src="https://github.com/user-attachments/assets/bfcebf03-bd15-4377-92eb-6e4f61598506" />
<img width="1912" height="886" alt="Image" src="https://github.com/user-attachments/assets/95c5770d-6b43-4de8-af4f-6d16f6551b50" />
Example:

Prediction: Churn
Probability of Churn: 0.55


