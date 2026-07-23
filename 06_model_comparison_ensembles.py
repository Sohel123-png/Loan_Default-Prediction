"""
STEP 6: MULTIPLE MODEL COMPARISON + ENSEMBLE MODELS (Voting, Bagging,
        Gradient Boosting, AdaBoost, Stacking)
=========================================================================
BUGS FIXED IN THIS FILE:

BUG A (Dead code): Original notebook me BaggingClassifier, GradientBoostingClassifier,
   AdaBoostClassifier sirf DEFINE kiye gaye the:
       bagging = BaggingClassifier(...)
       gb = GradientBoostingClassifier(...)
       ada = AdaBoostClassifier(...)
   ...lekin inhe kabhi train (fit) ya evaluate nahi kiya gaya. Yeh "dead code"
   hai -> likha gaya lekin use hi nahi hua. Interview me isse "incomplete
   experimentation" bola jaata hai. FIX: in teeno models ko bhi baaki models
   ke saath cross-validation comparison me shaamil kiya gaya hai.

BUG B (Stacking evaluate nahi kiya gaya): Original notebook me:
       stack_pipe.fit(X_train, y_train)
   ke baad koi predict/score/metric print nahi hua — pata hi nahi chalta
   Stacking model achha hai ya bura. FIX: stacking model ka bhi ROC-AUC,
   accuracy, precision, recall, F1 print kiya gaya hai jaise baaki models ka.

BUG C (Best model select karne ka koi logic nahi tha): Notebook me itne
   saare models compare karne ke baad bhi, aage jaake save/deploy sirf
   sabse pehla basic Logistic Regression pipeline hi kiya gaya — model
   comparison ka result istemal hi nahi hua! FIX: is file ke end me
   sabse best ROC-AUC wala model dhoond kar 'artifacts/best_model_name.txt'
   me likha jaata hai, jo agle step (07) me use hota hai.
"""

import joblib
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, VotingClassifier,
    BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier   # pip install catboost (agar installed nahi hai)

ARTIFACT_DIR = "artifacts"
X_train = joblib.load(os.path.join(ARTIFACT_DIR, "X_train.joblib"))
X_test = joblib.load(os.path.join(ARTIFACT_DIR, "X_test.joblib"))
y_train = joblib.load(os.path.join(ARTIFACT_DIR, "y_train.joblib"))
y_test = joblib.load(os.path.join(ARTIFACT_DIR, "y_test.joblib"))
preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.joblib"))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---- Sab base models (FIX A: bagging, gb, ada bhi shaamil) ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

results = []
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    results.append({"Model": name, "Mean ROC AUC": score.mean(), "Std": score.std()})
    print(f"{name:20s} -> Mean ROC AUC: {score.mean():.4f}  Std: {score.std():.4f}")

# ---- Voting Classifier ----
voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("xgb", XGBClassifier(random_state=42)),
    ],
    voting="soft",
)
voting_pipe = Pipeline([("preprocessor", preprocessor), ("model", voting)])
voting_pipe.fit(X_train, y_train)
voting_prob = voting_pipe.predict_proba(X_test)[:, 1]
voting_auc = roc_auc_score(y_test, voting_prob)
results.append({"Model": "Voting Classifier", "Mean ROC AUC": voting_auc, "Std": 0.0})
print(f"{'Voting Classifier':20s} -> Test ROC AUC: {voting_auc:.4f}")

# ---- Stacking Classifier (FIX B: ab evaluate bhi ho raha hai) ----
estimators = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("xgb", XGBClassifier(random_state=42)),
    ("cat", CatBoostClassifier(verbose=0, random_state=42)),
]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_pipe = Pipeline([("preprocessor", preprocessor), ("model", stack)])
stack_pipe.fit(X_train, y_train)

stack_pred = stack_pipe.predict(X_test)
stack_prob = stack_pipe.predict_proba(X_test)[:, 1]
print("Stacking Classifier Results:")
print("  Accuracy :", accuracy_score(y_test, stack_pred))
print("  Precision:", precision_score(y_test, stack_pred))
print("  Recall   :", recall_score(y_test, stack_pred))
print("  F1       :", f1_score(y_test, stack_pred))
print("  ROC AUC  :", roc_auc_score(y_test, stack_prob))
results.append({"Model": "Stacking Classifier", "Mean ROC AUC": roc_auc_score(y_test, stack_prob), "Std": 0.0})

# ---- Final comparison table ----
results_df = pd.DataFrame(results).sort_values(by="Mean ROC AUC", ascending=False)
print("\nFinal Model Comparison:\n", results_df)

# ---- FIX C: best model dhoondo aur save karo ----
best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest Model: {best_model_name}")

# Best model ka fitted pipeline save karo taaki step 07 use kar sake
if best_model_name == "Voting Classifier":
    best_pipeline = voting_pipe
elif best_model_name == "Stacking Classifier":
    best_pipeline = stack_pipe
else:
    best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", models[best_model_name])])
    best_pipeline.fit(X_train, y_train)

joblib.dump(best_pipeline, os.path.join(ARTIFACT_DIR, "best_pipeline.joblib"))
with open(os.path.join(ARTIFACT_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_model_name)

results_df.to_csv(os.path.join(ARTIFACT_DIR, "model_comparison_results.csv"), index=False)
print("Best model pipeline saved to artifacts/best_pipeline.joblib")
