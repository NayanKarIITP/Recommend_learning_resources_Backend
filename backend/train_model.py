import os
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")

print("🔄 Reading studentVle files...")

# Load and combine studentVle files
frames = []
for i in range(8):
    file_path = os.path.join(DATA_PATH, f"studentVle_{i}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Missing: {file_path}")
    
    df = pd.read_csv(file_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    frames.append(df)

studentVle = pd.concat(frames, ignore_index=True)

print("✔ studentVle loaded. Rows:", studentVle.shape[0])

# Build interaction matrix
print("🔄 Building interaction matrix...")

interaction_matrix = studentVle.pivot_table(
    index="id_student",
    columns="id_site",
    values="sum_click",
    aggfunc="sum",
    fill_value=0,
)

print("✔ Interaction matrix shape:", interaction_matrix.shape)

# Extract IDs
students = interaction_matrix.index.tolist()
resources = interaction_matrix.columns.tolist()

# Save ID lists
pd.DataFrame(students).to_csv(os.path.join(DATA_PATH, "students.csv"), index=False)
pd.DataFrame(resources).to_csv(os.path.join(DATA_PATH, "resources.csv"), index=False)

print("✔ Saved students.csv and resources.csv")

# Train NMF
print("🔄 Training NMF model... (may take a minute)")

nmf = NMF(n_components=50, init="random", random_state=42, max_iter=300)
W = nmf.fit_transform(interaction_matrix.values)
H = nmf.components_

# Save W, H
np.save(os.path.join(DATA_PATH, "W.npy"), W)
np.save(os.path.join(DATA_PATH, "H.npy"), H)

print("🎉 Training finished!")
print("✔ Saved W.npy and H.npy")
print("🚀 Model ready for Render deployment")
