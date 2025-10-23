import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier  # Changed from RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1 – Define column names
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Step 2 – Load and clean Hospital 1 (Cleveland)
h1 = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    names=columns,
    na_values="?"
)
h1 = h1.dropna().reset_index(drop=True)
h1 = h1.apply(pd.to_numeric)

# Convert target to binary: 0 = no disease, 1 = has disease
h1["target"] = h1["target"].apply(lambda x: 1 if x > 0 else 0)
h1.to_csv("cleveland_raw.csv", index=False)

# Step 3 – Load and clean Hospital 2 (Hungarian)
h2 = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data",
    sep=r"\s+",
    header=None,
    names=columns,
    na_values="?"
)
h2 = h2.apply(pd.to_numeric, errors='coerce')
h2.fillna(h2.mean(), inplace=True)
h2 = h2.dropna().reset_index(drop=True)

# Convert target to binary
h2["target"] = h2["target"].apply(lambda x: 1 if x > 0 else 0)
h2.to_csv("hungarian_raw.csv", index=False)

# Step 4 – Train Teacher Model for Hospital 1
X1 = h1.drop("target", axis=1)
y1 = h1["target"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

teacher1 = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Changed to GradientBoosting
teacher1.fit(X1_train, y1_train)

# Step 5 – Train Teacher Model for Hospital 2
X2 = h2.drop("target", axis=1)
y2 = h2["target"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

teacher2 = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Changed to GradientBoosting
teacher2.fit(X2_train, y2_train)

# Step 6 – Generate Soft Labels
public_data = pd.concat([X1_test, X2_test]).reset_index(drop=True)

soft_pred1 = teacher1.predict_proba(public_data)
soft_pred2 = teacher2.predict_proba(public_data)

pd.DataFrame(soft_pred1, columns=["prob_0_h1", "prob_1_h1"]).to_csv("cleveland_softlabels.csv", index=False)
pd.DataFrame(soft_pred2, columns=["prob_0_h2", "prob_1_h2"]).to_csv("hungarian_softlabels.csv", index=False)

# Step 7 – Combine Soft Labels (Federated Knowledge Distillation)
avg_soft = (soft_pred1 + soft_pred2) / 2.0
combined_dataset = public_data.copy()
combined_dataset["soft_label_0"] = avg_soft[:, 0]
combined_dataset["soft_label_1"] = avg_soft[:, 1]

combined_dataset.to_csv("combined_phase_fkd.csv", index=False)

# Step 8 – Output File Paths
output_paths = {
    "Hospital_1_phase_raw.csv": "cleveland_raw.csv",
    "Hospital_2_phase_raw.csv": "hungarian_raw.csv",
    "Hospital_1_phase_softlabels.csv": "cleveland_softlabels.csv",
    "Hospital_2_phase_softlabels.csv": "hungarian_softlabels.csv",
    "Hospital_combined_phase_fkd.csv": "combined_phase_fkd.csv"
}

# Optional: Print file paths
print("Generated Files:")
for label, path in output_paths.items():
    print(f"{label}: {path}")
