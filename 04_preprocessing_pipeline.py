"""
STEP 4: TRAIN-TEST SPLIT + PREPROCESSING PIPELINE
=====================================================
Yahan hum:
1. Features (X) aur Target (y) alag karte hain
2. Train-test split karte hain (stratified, kyunki target thoda imbalanced hai)
3. Numerical aur categorical columns ke liye alag preprocessing pipeline banate hain
4. ColumnTransformer se dono ko combine karte hain

Yeh step original notebook me sahi tha (koi major bug nahi), bas ab yeh
clean 'engineered_data.csv' (jisme multicollinearity fix ho chuki hai) use
karta hai.

Interview point:
- Preprocessing pipeline SIRF training data par 'fit' hota hai (fit_transform),
  test data par sirf 'transform' hota hai. Isse DATA LEAKAGE nahi hota
  (test set ki information training statistics -- jaise median/mean -- me
  leak nahi hoti). sklearn ka Pipeline yeh automatically handle karta hai.
"""

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

ARTIFACT_DIR = "artifacts"
df = pd.read_csv(os.path.join(ARTIFACT_DIR, "engineered_data.csv"))

# ---- Step 1: Features aur Target ----
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# ---- Step 2: Train-Test Split (stratified) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Step 3: Numerical aur Categorical columns identify karo ----
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include="object").columns
print("Numerical features:", list(num_features))
print("Categorical features:", list(cat_features))

# ---- Step 4: Numerical Pipeline (median impute + scale) ----
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# ---- Step 5: Categorical Pipeline (mode impute + one-hot encode) ----
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# ---- Step 6: Combine via ColumnTransformer ----
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
])

# Sab kuch save karo taaki agle steps me reuse ho sake
joblib.dump(X_train, os.path.join(ARTIFACT_DIR, "X_train.joblib"))
joblib.dump(X_test, os.path.join(ARTIFACT_DIR, "X_test.joblib"))
joblib.dump(y_train, os.path.join(ARTIFACT_DIR, "y_train.joblib"))
joblib.dump(y_test, os.path.join(ARTIFACT_DIR, "y_test.joblib"))
joblib.dump(preprocessor, os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

print("Train-test split & preprocessor saved to 'artifacts/' folder.")
