"""
STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
========================================
Is file me sirf ANALYSIS hai — koi data change nahi ho raha (yeh cleaned_data.csv
se read karta hai jo step 1 me generate hui thi).

BUG JO FIX KIYA:
Original notebook me skewness nikalte waqt:
    num_cols = df.select_dtypes(include='number')
    num_cols.skew()
Isme TARGET column 'loan_status' (0/1) bhi shaamil ho gaya tha, jabki skewness
sirf FEATURES ki nikalni chahiye thi. Target ek binary column hai, uski
"skewness" ka koi meaningful matlab nahi (woh class-imbalance ka topic hai,
skewness ka nahi). FIX: target ko explicitly exclude kiya gaya hai.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

ARTIFACT_DIR = "artifacts"
df = pd.read_csv(os.path.join(ARTIFACT_DIR, "cleaned_data.csv"))

# -------------------- Basic Info --------------------
print(df.shape)
print(df.describe())

# -------------------- Univariate Analysis --------------------
sns.histplot(df["person_age"])
plt.title("Person Age Distribution")
plt.show()

sns.histplot(df["person_income"])
plt.title("Person Income Distribution")
plt.show()

sns.countplot(data=df, x="person_home_ownership")
plt.title("Home Ownership Counts")
plt.show()

plt.figure(figsize=(15, 7))
sns.countplot(data=df, x="loan_intent")
plt.title("Loan Intent Counts")
plt.show()

sns.histplot(df["loan_amnt"])
plt.title("Loan Amount Distribution")
plt.show()

sns.countplot(data=df, x="loan_status")
plt.title("Loan Status (Target) Distribution")
plt.show()

# -------------------- Skewness (FIXED: target excluded) --------------------
num_cols = df.drop(columns=["loan_status"]).select_dtypes(include="number")
print("Skewness of numeric features (target excluded):")
print(num_cols.skew())
# near 0 -> centralized | negative -> left skewed | positive -> right skewed

# -------------------- Outlier count via IQR (for reference/reporting) --------------------
Q1 = num_cols.quantile(0.25)
Q3 = num_cols.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outlier_count = ((num_cols < lower) | (num_cols > upper)).sum()
print("Outlier counts per column:\n", outlier_count.sort_values(ascending=False))

# -------------------- Bivariate Analysis --------------------
for feature, title in [
    ("loan_amnt", "Loan Amount vs Loan Status"),
    ("loan_int_rate", "Interest Rate vs Loan Status"),
    ("loan_percent_income", "Loan Percent Income vs Loan Status"),
    ("person_age", "Person Age vs Loan Status"),
]:
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="loan_status", y=feature)
    plt.title(title)
    plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="loan_grade", hue="loan_status")
plt.title("Loan Grade vs Loan Status")
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="loan_intent", hue="loan_status")
plt.title("Loan Intent vs Loan Status")
plt.xticks(rotation=20)
plt.show()

# -------------------- Multivariate Analysis --------------------
plt.figure(figsize=(10, 7))
corr = df.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

default_rate_grade = (
    df.groupby("loan_grade")["loan_status"].mean().sort_values(ascending=False)
)
default_rate_grade.plot(kind="bar", figsize=(8, 5), title="Default Rate by Loan Grade")
plt.show()

default_rate_intent = (
    df.groupby("loan_intent")["loan_status"].mean().sort_values(ascending=False)
)
default_rate_intent.plot(kind="bar", figsize=(10, 5), title="Default Rate by Loan Purpose")
plt.xticks(rotation=20)
plt.show()
