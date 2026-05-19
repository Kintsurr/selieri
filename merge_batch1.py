"""
merge_batch1.py
Merges features 1.xlsx with batch1.xlsx into one combined dataset.
Run from any Python environment with pandas + openpyxl installed:
    pip install pandas openpyxl
    python merge_batch1.py
Output: batch1_features_labels.xlsx  (same folder)
"""
import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

print("Loading features 1.xlsx ...")
features = pd.read_excel(os.path.join(BASE, "features 1.xlsx"))
print(f"  Features shape: {features.shape}")

print("Loading batch1.xlsx ...")
batch1 = pd.read_excel(os.path.join(BASE, "sim_games", "batch1.xlsx"))
print(f"  Batch1 shape:   {batch1.shape}")

# Align game IDs: features uses 1-based index, batch uses 0-based
features["game_id"] = features["GameIndex"] - 1

merged = features.merge(batch1, on="game_id", how="inner")
print(f"  Merged shape:   {merged.shape}")

out_path = os.path.join(BASE, "batch1_features_labels.xlsx")
merged.to_excel(out_path, index=False)
print(f"\nSaved: {out_path}")
print("Done.")
