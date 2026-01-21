"""
Offline training script.
Run locally only.
DO NOT use on production servers.
"""

import glob
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

# -------- CONFIG --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
N_COMPONENTS = 50
# ------------------------

print("🔄 Loading CSV files...")

data = {}
for file in glob.glob(os.path.join(DATA_PATH, "*.csv")):
    name = os.path.basename(file).replace(".csv", "")
    data[name] = pd.read_csv(file)
    print("✔ Loaded:", name)

# Validate studentVle files
for i in range(8):
    if f"studentVle_{i}" not in data:
        raise RuntimeError(f"Missing studentVle_{i}.csv")

# Combine studentVle
studentVle = pd.concat(
    [data[f"studentVle_{i}"] for i in range(8)],
    ignore_index=True
)

if "Unnamed: 0" in studentVle.columns:
    studentVle.drop(columns=["Unnamed: 0"], inplace=True)

# Merge with VLE metadata
vle = data["vle"]
studentVle_full = studentVle.merge(vle, on="id_site", how="left")

# Interaction matrix
print("🔄 Building interaction matrix...")
interaction_matrix = studentVle_full.pivot_table(
    index="id_student",
    columns="id_site",
    values="sum_click",
    aggfunc="sum",
    fill_value=0
)

print("✔ Interaction matrix shape:", interaction_matrix.shape)

# Train NMF
print("🔄 Training NMF model...")
nmf = NMF(
    n_components=N_COMPONENTS,
    init="random",
    random_state=42,
    max_iter=300
)

W = nmf.fit_transform(interaction_matrix.values)
H = nmf.components_

# Save model
np.save(os.path.join(DATA_PATH, "W.npy"), W)
np.save(os.path.join(DATA_PATH, "H.npy"), H)

# Save mapping files
interaction_matrix.index.to_series().to_csv(os.path.join(DATA_PATH, "students.csv"), index=False)
interaction_matrix.columns.to_series().to_csv(os.path.join(DATA_PATH, "resources.csv"), index=False)

print("🎉 Model training complete!")
print("✔ Saved W.npy, H.npy")
print("🚀 Ready for deployment")
