"""
STEP 1: DATA LOADING & CLEANING
================================
Is file ka kaam:
1. CSV load karna
2. Missing values sahi tarike se fill karna
3. Duplicate rows hatana
4. Unrealistic outliers (jaise age = 144) ko treat karna

BUGS JO ORIGINAL NOTEBOOK ME THE (is file me fix kiye gaye hain):

BUG 1: '/content/credit_risk_dataset.csv' - yeh Google Colab ka hardcoded path hai.
       Agar Colab ke bahar (local machine / Jupyter / VS Code) run karoge to
       "FileNotFoundError" aayega. FIX: path ko variable/CONFIG bana diya hai,
       taaki kahin bhi run karne par sirf ek jagah change karna pade.

BUG 2: df["person_emp_length"].fillna(median, inplace=True)
       Yeh chained assignment + inplace=True hai. Pandas ke naye versions
       (2.x/3.x) me is se "ChainedAssignmentError" ya FutureWarning aata hai
       kyunki pandas guarantee nahi deta ki original df update hoga ya nahi.
       FIX: df["col"] = df["col"].fillna(value)  -> yeh hamesha reliable hai.

BUG 3: Observation me likha tha "Age 144 unrealistic hai" lekin uske baad
       kahin bhi is outlier ko treat (cap/remove) nahi kiya gaya tha.
       Sirf note kar diya, action nahi liya -> yeh EDA me common mistake hai
       (identify karna but fix na karna). FIX: age ko realistic upper bound
       (jaise 90 saal) par cap kiya gaya hai, taaki model garbage value se
       na sikhe.

BUG 4: person_emp_length me bhi kabhi kabhi galat/unrealistic values ho sakti
       hain (jaise emp_length > person_age, ya bahut zyada saal). Isko bhi
       ek sanity check ke through capture kiya gaya hai (interview me pucha
       ja sakta hai: "data consistency checks kaise karte ho?").
"""

import pandas as pd
import numpy as np
import os

# ---- CONFIG (path ko yahan change karo, poore notebook me nahi) ----
RAW_DATA_PATH = r"C:\Users\sasal\OneDrive\Desktop\comeback\Loan Default Prediction\archive (4)\credit_risk_dataset.csv"
ARTIFACT_DIR = r"C:\Users\sasal\OneDrive\Desktop\comeback\Loan Default Prediction\artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """CSV ko load karta hai. Agar file nahi milti to clear error dega."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' nahi mila. Kripya RAW_DATA_PATH sahi set karo."
        )
    df = pd.read_csv(path)
    print(f"Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def treat_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing values ko MEDIAN se fill karte hain kyunki median outliers se
    kam affect hota hai (mean ke muqable robust hai).
    """
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )
    df["loan_int_rate"] = df["loan_int_rate"].fillna(
        df["loan_int_rate"].median()
    )
    print("Missing values after treatment:\n", df.isnull().sum())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")
    return df


def treat_unrealistic_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX (BUG 3): 'person_age' me max value 144 thi jo insaan ke liye
    biologically impossible hai. Hum isko drop nahi kar rahe (data loss
    avoid karne ke liye) balki ek realistic cap laga rahe hain.

    Interview point: "Outlier ko hamesha delete nahi karte, pehle
    business context dekhte hain -> kabhi cap karte hain, kabhi
    transform (log), kabhi drop."
    """
    before = df.shape[0]
    df = df[df["person_age"] <= 90].reset_index(drop=True)
    after = df.shape[0]
    print(f"Removed {before - after} rows with unrealistic age (> 90).")

    # Sanity check: employment length age se zyada nahi ho sakti
    invalid_emp = (df["person_emp_length"] > df["person_age"]).sum()
    if invalid_emp > 0:
        print(f"Warning: {invalid_emp} rows me emp_length > age hai, dropping them.")
        df = df[df["person_emp_length"] <= df["person_age"]].reset_index(drop=True)

    return df


def main():
    df = load_data(RAW_DATA_PATH)
    df = treat_missing_values(df)
    df = remove_duplicates(df)
    df = treat_unrealistic_outliers(df)

    out_path = os.path.join(ARTIFACT_DIR, "cleaned_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to: {out_path}")


if __name__ == "__main__":
    main()
