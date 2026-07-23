"""
STEP 7: SAVE FINAL MODEL + PREDICT ON NEW DATA
==================================================
SABSE BADI LOGICAL BUG (interview me zaroor pucha jayega):
------------------------------------------------------------
Original notebook me itne saare models train/compare karne, Voting aur
Stacking ensembles banane ke BAAD BHI, save karte waqt yeh code tha:

    joblib.dump(pipeline, "loan_default_pipeline.joblib")

Yahan 'pipeline' woh VERY FIRST basic Logistic Regression pipeline tha
(jo sabse shuru me bina kisi comparison ke train kiya gaya tha) — na ki
best-performing model jo comparison table se pata chala, na Voting, na
Stacking. Matlab: itni mehnat se best model dhoondne ke baad bhi galti se
sabse pehla/weakest pipeline hi save/deploy ho raha tha!

FIX: Ab hum step 06 me save kiya gaya 'best_pipeline.joblib' load karke
usko hi final deployment ke liye use kar rahe hain — jo actual comparison
ke basis par best ROC-AUC deta hai.
"""

import joblib
import os
import numpy as np
import pandas as pd

ARTIFACT_DIR = "artifacts"

# ---- Best model load karo (FIX: baseline ki jagah asli best model) ----
best_pipeline = joblib.load(os.path.join(ARTIFACT_DIR, "best_pipeline.joblib"))
with open(os.path.join(ARTIFACT_DIR, "best_model_name.txt")) as f:
    best_model_name = f.read().strip()
print(f"Loaded best model for deployment: {best_model_name}")

# Final deployment ke liye clean naam se dubara save (production artifact)
joblib.dump(best_pipeline, os.path.join(ARTIFACT_DIR, "loan_default_pipeline.joblib"))

# ---- New sample data par prediction ----
new_data = pd.DataFrame({
    "person_age": [28],
    "person_home_ownership": ["RENT"],
    "person_emp_length": [5],
    "loan_intent": ["EDUCATION"],
    "loan_grade": ["B"],
    "loan_amnt": [12000],
    "loan_int_rate": [10.5],
    "loan_percent_income": [0.17],
    "cb_person_default_on_file": ["N"],
    "cb_person_cred_hist_length": [4],
    "person_income": [70000],   # temp column, log ke liye chahiye, phir drop hoga
})

# FIX (consistency with 03_feature_engineering.py):
# Training time par hamne 'person_income' ko drop kar diya tha aur sirf
# 'person_income_log' rakha tha. Isliye naye data par bhi WAHI transformation
# apply karna zaroori hai, warna pipeline "column mismatch" error dega.
new_data["person_income_log"] = np.log1p(new_data["person_income"])
new_data = new_data.drop(columns=["person_income"])

prediction = best_pipeline.predict(new_data)
probability = best_pipeline.predict_proba(new_data)

print("Prediction  :", "Default" if prediction[0] == 1 else "No Default")
print("Probability :", probability)
