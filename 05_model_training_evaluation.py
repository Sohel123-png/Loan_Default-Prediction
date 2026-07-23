"""
STEP 5: BASELINE MODEL TRAINING + EVALUATION + CROSS VALIDATION
====================================================================
Yahan Logistic Regression ko baseline model ki tarah train karte hain,
evaluate karte hain, aur phir cross-validation se stability check karte hain.

MINOR BUG FIXED:
    ConfusionMatrixDisplay(cm).plot()
    -> plt.show() missing tha. .ipynb me auto-display ho jaata hai isliye
       chalta tha, lekin agar isse .py script me run karo to plot save/show
       nahi hoga. FIX: plt.show() add kiya gaya hai (yeh script/production
       me chalane ke liye zaroori hai).
"""

import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

ARTIFACT_DIR = "artifacts"
X_train = joblib.load(os.path.join(ARTIFACT_DIR, "X_train.joblib"))
X_test = joblib.load(os.path.join(ARTIFACT_DIR, "X_test.joblib"))
y_train = joblib.load(os.path.join(ARTIFACT_DIR, "y_train.joblib"))
y_test = joblib.load(os.path.join(ARTIFACT_DIR, "y_test.joblib"))
preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

# ---- Step 7: Complete baseline pipeline ----
baseline_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

# ---- Step 8: Training ----
baseline_pipeline.fit(X_train, y_train)

# ---- Step 9: Prediction ----
y_pred = baseline_pipeline.predict(X_test)
y_prob = baseline_pipeline.predict_proba(X_test)[:, 1]

# ---- Step 10: Evaluation ----
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1       :", f1_score(y_test, y_pred))
print("ROC AUC  :", roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix - Baseline Logistic Regression")
plt.show()  # FIX: yeh line missing thi

print(classification_report(y_test, y_pred))

# ---- Cross Validation (5-Fold Stratified) ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    baseline_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
)
print("ROC AUC Scores :", cv_scores)
print("Mean ROC AUC   :", cv_scores.mean())
print("Std            :", cv_scores.std())
# Lower std = zyada stable model across different folds

joblib.dump(baseline_pipeline, os.path.join(ARTIFACT_DIR, "baseline_pipeline.joblib"))
print("Baseline pipeline saved.")
