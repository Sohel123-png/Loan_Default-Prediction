# Loan Default Prediction — Code Review, Bugs & Fixes
### (Viva / Interview ready explanation — Hinglish)

Original notebook (`Untitled38__1_.ipynb`) ko maine 7 alag Python files me
split kar diya hai (production/industry style: data ingestion → cleaning →
EDA → feature engineering → preprocessing → training → ensembling → deployment).
Har file `artifacts/` folder me apna output save karti hai, jise next step
use karta hai. Isse notebook "modular ML pipeline" ban jaata hai — jo
interviewers ko bahut pasand aata hai.

| File | Kaam |
|---|---|
| `01_data_loading_cleaning.py` | CSV load, missing values, duplicates, outliers |
| `02_eda.py` | Univariate / Bivariate / Multivariate analysis, plots |
| `03_feature_engineering.py` | Log transform, multicollinearity fix |
| `04_preprocessing_pipeline.py` | Train-test split, ColumnTransformer pipeline |
| `05_model_training_evaluation.py` | Baseline Logistic Regression + metrics + CV |
| `06_model_comparison_ensembles.py` | 10+ models compare + Voting/Bagging/GB/AdaBoost/Stacking |
| `07_save_and_predict.py` | Sahi (best) model save + naye data par prediction |

---

## Mile hue Bugs (severity ke hisaab se, sabse important pehle)

### 🔴 Bug 1 — Sabse critical: Galat model save ho raha tha
**Kahan:** Notebook ke last me `joblib.dump(pipeline, "loan_default_pipeline.joblib")`

**Problem:** Poore notebook me 10+ models compare kiye gaye, Voting aur
Stacking ensembles banaye gaye — lekin save karte waqt sabse **pehla / simplest**
Logistic Regression pipeline hi save ho raha tha (`pipeline` variable jo bahut
shuru me define hua tha). Model comparison ka result kabhi use hi nahi hua.

**Kyun important hai (interview angle):** Yeh ek "silent bug" hai — code error
nahi deta, chalta hai bilkul theek se, lekin production me **galat / weakest
model deploy ho jaata**. Real projects me aisi galtiyan production tak pahunch
jaati hain kyunki koi error nahi aata.

**Fix:** `06_model_comparison_ensembles.py` me comparison table (`results_df`)
se best ROC-AUC wala model automatically select hota hai aur save hota hai.
`07_save_and_predict.py` usi best model ko load karta hai.

---

### 🔴 Bug 2 — Multicollinearity: log aur original feature dono model me
**Kahan:** `person_income_log` banaya gaya lekin `person_income` column drop
nahi hua — dono X (features) me chale gaye.

**Problem:** `person_income_log = log(1 + person_income)` — yeh dusre column
ka hi mathematical function hai, matlab same information do baar di ja rahi
thi. Isse:
- Linear models (Logistic Regression) ke coefficients unstable/misleading
  ho sakte hain.
- Feature importance samajhna mushkil ho jaata hai.

**Fix (`03_feature_engineering.py`):** Log transform ke baad original
`person_income` ko drop kar diya.

---

### 🟠 Bug 3 — Outlier identify kiya, lekin treat nahi kiya
**Kahan:** EDA me observation likha tha "Age 144 unrealistic hai" — lekin
kahin bhi is value ko cap/remove nahi kiya gaya.

**Problem:** "Identify karna" aur "fix karna" alag steps hain. Sirf note kar
dena kaafi nahi — agar model training tak yeh value gayi to model galat
patterns seekh sakta hai.

**Fix (`01_data_loading_cleaning.py`):** `person_age <= 90` cap laga diya,
aur ek extra sanity check bhi add kiya (`emp_length` kabhi `age` se zyada
nahi honi chahiye).

---

### 🟠 Bug 4 — `fillna(..., inplace=True)` chained assignment
**Kahan:**
```python
df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
```

**Problem:** Yeh "chained assignment" hai (`df["col"].method(inplace=True)`).
Pandas ke naye versions me is se `ChainedAssignmentError` / `FutureWarning`
aata hai kyunki pandas guarantee nahi karta ki original DataFrame actually
update hoga ya sirf ek temporary copy update hogi.

**Fix:** `df["col"] = df["col"].fillna(value)` — yeh hamesha reliable tareeka hai.
(Notebook me interesting baat: dusri line `loan_int_rate` ke liye already
sahi tareeke se likhi thi — yeh dikhata hai ki code me inconsistency thi.)

---

### 🟠 Bug 5 — Skewness me target column bhi included tha
**Kahan:**
```python
num_cols = df.select_dtypes(include='number')
num_cols.skew()
```

**Problem:** Isme `loan_status` (0/1 target) bhi shaamil ho gaya. Target ek
binary/categorical variable hai, uski "skewness" nikalna meaningless hai
(uske liye class-imbalance dekhte hain, skewness nahi).

**Fix (`02_eda.py`):** Target column ko explicitly exclude kiya.

---

### 🟡 Bug 6 — Bagging, Gradient Boosting, AdaBoost define kiye but kabhi use nahi kiye ("dead code")
**Kahan:** Cells me `bagging = BaggingClassifier(...)`, `gb = GradientBoostingClassifier(...)`,
`ada = AdaBoostClassifier(...)` define hue, lekin kabhi `.fit()` ya
`cross_val_score` nahi hua.

**Problem:** Incomplete experimentation — code likha gaya but result kabhi
nikala hi nahi. Model comparison table incomplete rah gayi.

**Fix (`06_model_comparison_ensembles.py`):** In teeno models ko bhi comparison
loop me shaamil kiya, taaki fair comparison ho.

---

### 🟡 Bug 7 — Stacking Classifier fit hua lekin evaluate nahi hua
**Kahan:** `stack_pipe.fit(X_train, y_train)` ke baad koi predict/metric print
nahi hua.

**Problem:** Pata hi nahi chal raha tha Stacking model achha hai ya bura —
sabse zyada complex/expensive model banaya but result hi nahi dekha.

**Fix:** Accuracy, Precision, Recall, F1, ROC-AUC — sab print kiya jaata hai,
aur comparison table me bhi add kiya jaata hai.

---

### 🟢 Bug 8 — Hardcoded Colab path
**Kahan:** `pd.read_csv('/content/credit_risk_dataset.csv')`

**Problem:** `/content/` sirf Google Colab me exist karta hai. Local Jupyter,
VS Code, ya kisi bhi non-Colab environment me `FileNotFoundError` aayega.

**Fix:** Path ko ek `CONFIG` variable (`RAW_DATA_PATH`) me nikal diya, top of
file par, aur clear error-message diya agar file na mile.

---

### 🟢 Bug 9 — `ConfusionMatrixDisplay(cm).plot()` ke baad `plt.show()` missing
**Kahan:** Evaluation section.

**Problem:** Notebook me chal jaata hai (auto-display), lekin agar isi code
ko `.py` script me chalao (jaisa production me hota hai) to plot window/figure
show/save nahi hoga.

**Fix:** `plt.show()` add kiya.

---

## Interview me kaise explain karo (summary in 30 seconds)

> "Maine is notebook ko review karke ek modular pipeline me convert kiya —
> data cleaning, EDA, feature engineering, preprocessing, training, aur
> ensembling ko alag files me separate kiya. Sabse important bug jo mila
> woh yeh tha ki model-comparison ke baad bhi galti se weakest baseline model
> hi save/deploy ho raha tha; maine best-ROC-AUC wala model automatically
> select aur save hone ki logic add ki. Iske alawa multicollinearity
> (log aur raw income column dono), untreated outliers, chained-assignment
> pandas warning, aur kuch incomplete/unevaluated models (Bagging, GB, AdaBoost,
> Stacking) bhi fix kiye."

## Aage kya improve kiya ja sakta hai (bonus points agar poocha jaye)
- Hyperparameter tuning (GridSearchCV / Optuna) best model par
- SHAP / feature importance se model explainability
- Class imbalance ke liye SMOTE ya `class_weight='balanced'`
- Threshold tuning (0.5 ke bajaye business-cost ke hisaab se threshold choose karna)
- Config file (YAML) se paths aur hyperparameters manage karna
