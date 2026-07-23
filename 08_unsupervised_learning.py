"""
STEP 8: UNSUPERVISED LEARNING (Customer Segmentation + Anomaly Detection)
=============================================================================
Original notebook me sirf SUPERVISED learning (Logistic Regression, Random
Forest, XGBoost, LightGBM, CatBoost, Voting, Stacking) tha — target column
'loan_status' hamesha use ho raha tha.

Yeh file UNSUPERVISED learning add karti hai, jisme target column (y) ka
istemal training ke waqt bilkul NAHI hota — model sirf features (X) ke
patterns se khud groups/anomalies dhoondta hai.

Is file me 3 cheezein hain:
1. K-Means Clustering        -> borrowers ko "similar risk profile" groups me baantna
2. PCA (2D visualization)    -> high-dimensional data ko 2D me plot karna
3. Isolation Forest          -> unusual / anomalous loan applications dhoondna

Interview point:
-----------------
"Supervised learning me hum label (loan_status) use karte hain model ko
sikhane ke liye. Unsupervised learning me hum label use hi nahi karte —
model khud data ke structure/patterns se groups banata hai. Iska use case
hota hai: naye customer segments discover karna, ya fraud/anomaly detect
karna jab humare paas labeled fraud data na ho."
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

ARTIFACT_DIR = "artifacts"
df = pd.read_csv(os.path.join(ARTIFACT_DIR, "engineered_data.csv"))

# Unsupervised learning me target column ka use NAHI karte — sirf features
X = df.drop(columns=["loan_status"])
y_true = df["loan_status"]  # sirf END me comparison/validation ke liye rakha hai, training me nahi

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include="object").columns

# ---------------------------------------------------------------------------
# Preprocessing (supervised wale jaisa hi — scale + one-hot encode)
# Clustering aur distance-based algorithms (KMeans) ke liye scaling
# ZAROORI hai, warna bade-range wale columns (jaise loan_amnt) chhote-range
# wale columns (jaise loan_percent_income) ko dominate kar dete hain.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_features),
])

X_processed = preprocessor.fit_transform(X)
# Sparse matrix (OneHotEncoder se) ko dense array me convert karo, KMeans/PCA ko dense chahiye
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

print(f"Processed feature matrix shape: {X_processed.shape}")


# ---------------------------------------------------------------------------
# PART 1: K-MEANS CLUSTERING — Customer Segmentation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 1: K-Means Clustering")
print("=" * 60)

# ---- Elbow Method: sahi 'k' (number of clusters) dhoondne ke liye ----
# Hum k=2 se k=8 tak WCSS (Within-Cluster Sum of Squares / inertia) plot
# karte hain. Jahan curve "elbow" ki tarah mudhta hai, wahi optimal k hai.
inertias = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_processed)
    inertias.append(km.inertia_)
    # Silhouette score: -1 (bad) se +1 (best) tak. Batata hai clusters
    # kitne "well-separated" hain.
    silhouette_scores.append(silhouette_score(X_processed, labels, sample_size=5000, random_state=42))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_range), inertias, marker="o")
axes[0].set_title("Elbow Method (WCSS vs k)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(list(k_range), silhouette_scores, marker="o", color="green")
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# Best k = jahan silhouette score sabse zyada hai
best_k = list(k_range)[int(np.argmax(silhouette_scores))]
print(f"Best k (by silhouette score): {best_k}")

# ---- Final KMeans model with best_k ----
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_processed)
df["cluster"] = cluster_labels

print(f"\nCluster sizes:\n{df['cluster'].value_counts().sort_index()}")

# ---- Cluster Profiling: har cluster ka average profile dekhna ----
profile_cols = ["person_age", "person_income_log", "loan_amnt",
                 "loan_int_rate", "loan_percent_income"]
cluster_profile = df.groupby("cluster")[profile_cols].mean()
print("\nCluster Profiles (mean values):\n", cluster_profile)

# ---- Validation (informational only): clusters actual default rate se compare ----
# NOTE: Yeh sirf VALIDATION/INTERPRETATION ke liye hai — training me target
# use nahi hua tha. Yeh dekhne ke liye hai ki jo clusters khud-b-khud bane,
# kya unme koi cluster "high risk" ki tarah dikhta hai.
default_rate_by_cluster = df.groupby("cluster")["loan_status"].mean().sort_values(ascending=False)
print("\nActual default rate per cluster (for interpretation only):\n", default_rate_by_cluster)

plt.figure(figsize=(7, 5))
default_rate_by_cluster.plot(kind="bar", color="steelblue")
plt.title("Default Rate by Unsupervised Cluster")
plt.ylabel("Default Rate")
plt.xlabel("Cluster")
plt.show()


# ---------------------------------------------------------------------------
# PART 2: PCA — 2D Visualization of Clusters
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: PCA (Dimensionality Reduction for Visualization)")
print("=" * 60)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_processed)

print(f"Explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")
print(f"Total variance captured by 2 components: {pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=cluster_labels, cmap="viridis", alpha=0.5, s=10
)
plt.title("Customer Segments (PCA-reduced, colored by K-Means cluster)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.show()


# ---------------------------------------------------------------------------
# PART 3: ISOLATION FOREST — Anomaly / Unusual Application Detection
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3: Isolation Forest (Anomaly Detection)")
print("=" * 60)
# Isolation Forest bhi UNSUPERVISED hai — yeh label use nahi karta.
# Yeh un applications ko flag karta hai jo baaki data se "structurally
# different" hain (jaise: bahut high loan_amnt + bahut low income + bahut
# high interest rate ka unusual combination) — chahe woh actually default
# karein ya nahi. Real underwriting me isse "manual review ke liye flag
# karo" jaisa use hota hai.

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.03,   # expect ~3% data ko anomaly maanna
    random_state=42,
)
anomaly_pred = iso_forest.fit_predict(X_processed)   # -1 = anomaly, 1 = normal
df["is_anomaly"] = np.where(anomaly_pred == -1, 1, 0)

print(f"Number of anomalous applications flagged: {df['is_anomaly'].sum()} "
      f"out of {len(df)} ({df['is_anomaly'].mean():.2%})")

anomaly_default_rate = df.groupby("is_anomaly")["loan_status"].mean()
print("\nDefault rate: anomalous vs normal applications:\n", anomaly_default_rate)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[df["is_anomaly"] == 0, 0], X_pca[df["is_anomaly"] == 0, 1],
    alpha=0.4, s=10, label="Normal", color="steelblue"
)
plt.scatter(
    X_pca[df["is_anomaly"] == 1, 0], X_pca[df["is_anomaly"] == 1, 1],
    alpha=0.8, s=20, label="Anomaly", color="red"
)
plt.title("Anomalous Applications (Isolation Forest) on PCA plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()


# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
df.to_csv(os.path.join(ARTIFACT_DIR, "data_with_clusters_and_anomalies.csv"), index=False)
joblib.dump(kmeans, os.path.join(ARTIFACT_DIR, "kmeans_model.joblib"))
joblib.dump(iso_forest, os.path.join(ARTIFACT_DIR, "isolation_forest_model.joblib"))
joblib.dump(pca, os.path.join(ARTIFACT_DIR, "pca_model.joblib"))

print("\nSaved: data_with_clusters_and_anomalies.csv, kmeans_model.joblib, "
      "isolation_forest_model.joblib, pca_model.joblib")
