"""
merge_batch1_batch2.py
Merges batch1_features_labels.xlsx and batch2_features_labels.xlsx into one
combined dataset for AI training. Offsets batch2 game_id and GameIndex to
avoid collisions, and adds a 'batch' column for provenance.

Output: combined_features_labels.xlsx  (same folder)
"""
import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

print("Loading batch1_features_labels.xlsx ...")
b1 = pd.read_excel(os.path.join(BASE, "batch1_features_labels.xlsx"))
b1["batch"] = 1
print(f"  Shape: {b1.shape}")

print("Loading batch2_features_labels.xlsx ...")
b2 = pd.read_excel(os.path.join(BASE, "batch2_features_labels.xlsx"))
b2["batch"] = 2
# Offset so game_ids are globally unique across both batches
offset = b1["game_id"].max() + 1
b2["game_id"] = b2["game_id"] + offset
b2["GameIndex"] = b2["GameIndex"] + offset
print(f"  Shape: {b2.shape}")
print(f"  Offset applied: +{offset} to game_id and GameIndex")

combined = pd.concat([b1, b2], ignore_index=True)
print(f"\nCombined shape: {combined.shape}")
print(f"game_id range:  {combined['game_id'].min()} - {combined['game_id'].max()}")
print(f"Batch counts:   {combined['batch'].value_counts().to_dict()}")

out_path = os.path.join(BASE, "combined_features_labels.xlsx")
print(f"\nSaving to {out_path} ...")
combined.to_excel(out_path, index=False)
print("Done.")
