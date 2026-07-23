"""
STEP 3: FEATURE ENGINEERING
=============================
Yahan hum 'person_income' ka log-transform bana rahe hain kyunki income
column bahut zyada right-skewed hai (kuch log ke baad model behtar sikhta hai,
especially Logistic Regression jaise linear models ke liye).

BUG (SABSE BADI GALTI ORIGINAL NOTEBOOK ME):
------------------------------------------------
Original code:
    df["person_income_log"] = np.log1p(df["person_income"])
    ...
    X = df.drop("loan_status", axis=1)   # <-- yahan person_income
                                          #     AUR person_income_log
                                          #     DONO columns X me chale gaye!

Problem: 'person_income_log' sirf 'person_income' ka mathematical function hai
(log(1+x)). Dono columns ek hi information carry karte hain -> yeh
MULTICOLLINEARITY create karta hai.

Multicollinearity ka nuksaan:
- Logistic Regression jaise linear model ke coefficients unstable/misleading
  ho jaate hain (kaunsa feature actually important hai, samajhna mushkil).
- Model ko duplicate/redundant signal milta hai, jo overfitting/noise
  add kar sakta hai.
- Interview me yeh ek classic "feature engineering mistake" hai jo
  bataya jaata hai: "raw feature aur uska transformed version dono
  ek saath model me mat daalo."

FIX: log-transform banane ke baad original 'person_income' column ko
DROP kar diya gaya hai, sirf 'person_income_log' feature ke roop me
model ko diya jayega.
"""

import pandas as pd
import numpy as np
import os

ARTIFACT_DIR = "artifacts"
df = pd.read_csv(os.path.join(ARTIFACT_DIR, "cleaned_data.csv"))

# Log transform (right-skewed income ko normal-ish banane ke liye)
df["person_income_log"] = np.log1p(df["person_income"])

# FIX: original column drop kar rahe hain taaki duplicate information na ho
df = df.drop(columns=["person_income"])

out_path = os.path.join(ARTIFACT_DIR, "engineered_data.csv")
df.to_csv(out_path, index=False)
print(f"Feature engineered data saved to: {out_path}")
print("Columns going into modeling:\n", df.columns.tolist())
