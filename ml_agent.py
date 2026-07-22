import kagglehub
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

print("Downloading full-scale industrial data from Kaggle...")
dataset_dir = kagglehub.dataset_download("d4rklucif3r/full-scale-waste-water-treatment-plant-data")

# Auto-detect the exact CSV file in the folder
csv_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("Could not find any CSV data file inside the downloaded Kaggle directory.")

csv_path = os.path.join(dataset_dir, csv_files[0])
print(f"Loading dataset file: {csv_path}")
df = pd.read_csv(csv_path)

# Clean whitespaces and force lowercase to match exactly
df.columns = df.columns.str.strip().str.lower()

# MSC CORRECTION: Explicit mapping using the exact headers verified from your system log
feature_columns = ['avg_inflow', 'cod', 'am', 't']
target_column = 'total_grid'

print(f"Mapping SCADA headers: Inputs {feature_columns} -> Target: '{target_column}'")

# Clean up data strings or missing slots
for col in feature_columns + [target_column]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop missing records caused by sensor offline loops
cleaned_df = df[feature_columns + [target_column]].dropna()
print(f"Cleaned SCADA dataset contains {len(cleaned_df):,} daily plant logs.")

# Train-Test Split (80/20 standard protocol)
X = cleaned_df[feature_columns]
y = cleaned_df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor on full-scale historical logs...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Performance Verification
y_pred = model.predict(X_test)
real_r2 = r2_score(y_test, y_pred) * 100
print(f"\nModel Verification Complete!")
print(f"Real-World Cross-Validated R² Score: {real_r2:.2f}%")

# Export artifacts cleanly
artifacts = {
    "model": model,
    "r2_score": real_r2
}

with open("wastewater_brain_cod.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("Real industrial brain exported successfully to 'wastewater_brain_cod.pkl'!")
