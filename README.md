# 🏦 Loan Default Prediction using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

A complete **end-to-end Machine Learning project** that predicts whether a loan applicant is likely to default on a loan.

The project covers the entire ML lifecycle—from **data preprocessing and exploratory data analysis (EDA)** to **feature engineering, model training, hyperparameter tuning, ensemble learning, model selection, and deployment using FastAPI** with a modern responsive web interface.

---

# 🚀 Live Demo




### 🌐 Web Application


**🔗 Live Link:**

```
👉 https://loan-default-prediction-6sqs.onrender.com
```

*(Replace the above URL after deploying on Render, Railway, AWS, etc.)*

---

# 📸 Project Preview

> Add screenshots here after deployment.

```
images/
│── homepage.png
│── prediction.png
│── swagger.png
```

Example:

```md
![Home](images/homepage.png)

![Prediction](images/prediction.png)

![Swagger](images/swagger.png)
```

---

# ✨ Features

✅ End-to-End Machine Learning Pipeline

✅ Data Cleaning & Preprocessing

✅ Exploratory Data Analysis (EDA)

✅ Feature Engineering

✅ Automated Preprocessing Pipeline

✅ Multiple Machine Learning Models

✅ Hyperparameter Tuning

✅ Ensemble Learning

✅ Automatic Best Model Selection

✅ Model Persistence using Joblib

✅ REST API using FastAPI

✅ Interactive Swagger Documentation

✅ Responsive Frontend

✅ Risk Level Classification

✅ Probability Prediction

---

# 📂 Project Structure

```text
Loan-Default-Prediction/
│
├── archive (4)/
│   └── credit_risk_dataset.csv
│
├── artifacts/
│   ├── cleaned_data.csv
│   ├── engineered_data.csv
│   ├── preprocessor.joblib
│   ├── baseline_pipeline.joblib
│   ├── best_pipeline.joblib
│   ├── best_model_name.txt
│   ├── model_comparison_results.csv
│   ├── X_train.joblib
│   ├── X_test.joblib
│   ├── y_train.joblib
│   └── y_test.joblib
│
├── webapp/
│   ├── app.py
│   ├── requirements.txt
│   │
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── script.js
│
├── 01_data_loading_cleaning.py
├── 02_eda.py
├── 03_feature_engineering.py
├── 04_preprocessing_pipeline.py
├── 05_model_training_evaluation.py
├── 06_model_comparison_ensembles.py
├── 07_save_and_predict.py
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost

### Backend

- FastAPI
- Uvicorn

### Frontend

- HTML
- CSS
- JavaScript

### Model Storage

- Joblib

---

# 📊 Dataset

### Dataset

**Credit Risk Dataset**

### Target Variable

| Value | Meaning |
|--------|----------|
| 0 | No Default |
| 1 | Default |

---

# 🔄 Machine Learning Workflow

```text
                 Raw Dataset
                      │
                      ▼
           Data Cleaning
                      │
                      ▼
     Exploratory Data Analysis
                      │
                      ▼
        Feature Engineering
                      │
                      ▼
      Preprocessing Pipeline
                      │
                      ▼
          Model Training
                      │
                      ▼
      Hyperparameter Tuning
                      │
                      ▼
         Model Comparison
                      │
                      ▼
         Ensemble Learning
                      │
                      ▼
       Best Model Selection
                      │
                      ▼
       FastAPI Deployment
```

---

# 🤖 Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- AdaBoost
- Bagging Classifier
- Voting Classifier
- Stacking Classifier
- XGBoost
- LightGBM
- CatBoost

---

# 🏆 Best Model

The project automatically compares all implemented models and saves the best-performing pipeline.

```text
artifacts/
└── best_pipeline.joblib
```

The selected model is also stored in:

```text
artifacts/
└── best_model_name.txt
```

---

# 📈 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross Validation

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Loan-Default-Prediction.git
```

Move into the project

```bash
cd Loan-Default-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Machine Learning Pipeline

### Step 1

```bash
python 01_data_loading_cleaning.py
```

### Step 2

```bash
python 02_eda.py
```

### Step 3

```bash
python 03_feature_engineering.py
```

### Step 4

```bash
python 04_preprocessing_pipeline.py
```

### Step 5

```bash
python 05_model_training_evaluation.py
```

### Step 6

```bash
python 06_model_comparison_ensembles.py
```

### Step 7

```bash
python 07_save_and_predict.py
```

---

# 🚀 Run FastAPI

Move to the web application

```bash
cd webapp
```

Run the server

```bash
python -m uvicorn app:app --reload --port 8000
```

Server will start at

```
http://127.0.0.1:8000
```

---

# 🌐 API Endpoints

| Method | Endpoint | Description |
|----------|------------|----------------------------|
| GET | / | Home Page |
| GET | /api/health | Health Check |
| GET | /api/model-info | Model Information |
| POST | /api/predict | Predict Loan Default |

---

# 📥 Sample Request

```json
{
  "person_age": 30,
  "person_income": 90000,
  "person_home_ownership": "OWN",
  "person_emp_length": 8,
  "loan_intent": "HOMEIMPROVEMENT",
  "loan_grade": "A",
  "loan_amnt": 10000,
  "loan_int_rate": 7.5,
  "loan_percent_income": 0.11,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 10
}
```

---

# 📤 Sample Response

```json
{
  "prediction": 0,
  "verdict": "APPROVED",
  "default_probability": 0.0824,
  "risk_level": "Low",
  "model_used": "Stacking Classifier"
}
```

---

# 📚 API Documentation

FastAPI automatically provides interactive API documentation.

Open your browser and visit:

```
http://127.0.0.1:8000/docs
```

Swagger UI allows you to test every API endpoint directly from the browser.

---

# 📌 Future Improvements

- Docker Containerization
- Render Deployment
- AWS Deployment
- Azure Deployment
- Google Cloud Deployment
- Explainable AI (SHAP)
- Model Monitoring
- CI/CD Pipeline
- User Authentication
- Database Integration
- MLOps Pipeline

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you would like to improve this project, feel free to fork the repository and submit a Pull Request.

---

# 👨‍💻 Author

## Sohel Ali

**B.Tech – Artificial Intelligence & Machine Learning**

📧 Email: sayyedsohelali448@mail.com

💼 LinkedIn:

```
[https://linkedin.com/in/YOUR-LINKEDIN](https://www.linkedin.com/in/sohel-ali-6435253a8/
```

🐙 GitHub:

```
[https://github.com/YOUR_USERNAME](https://github.com/Sohel123-png
```

---

# ⭐ Support

If you found this project helpful,

please consider giving it a ⭐ on GitHub.

It motivates me to build more Machine Learning and AI projects.

---

<p align="center">

⭐ Thank You for Visiting ⭐

</p>
