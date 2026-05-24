"""
data_merger.py

Merges any number of batches into a single combined dataset for training.

Two modes:
  1. Feature-label merge   — joins a features Excel with a sim_games batch Excel on game_id
  2. Batch combine         — stacks multiple batchN_features_labels.xlsx files, offsets
                             game_id / GameIndex to keep them globally unique, adds a
                             'batch' provenance column

Usage
-----
# Step 1: merge features + labels for each batch (repeat for each new batch)
python data_merger.py merge --features "features 1.xlsx" --batch sim_games/batch1.xlsx --out batch1_features_labels.xlsx
python data_merger.py merge --features "features 2.xlsx" --batch sim_games/batch2.xlsx --out batch2_features_labels.xlsx
# ... add as many batches as needed

# Step 2: combine all batchN_features_labels.xlsx into one training file
python data_merger.py combine --inputs batch1_features_labels.xlsx batch2_features_labels.xlsx ... --out combined_features_labels.xlsx

# Auto-discover all batchN_features_labels.xlsx files in the current folder:
python data_merger.py combine --auto --out combined_features_labels.xlsx
"""

import argparse
import glob
import os
import sys
import pandas as pd


BASE = os.path.dirname(os.path.abspath(__file__))


def load(path: str) -> pd.DataFrame:
    path = path if os.path.isabs(path) else os.path.join(BASE, path)
    print(f"  Loading {os.path.basename(path)} ...", end=" ", flush=True)
    df = pd.read_excel(path)
    print(f"{df.shape}")
    return df


def cmd_merge(args):
    features = load(args.features)
    batch = load(args.batch)

    # features uses 1-based GameIndex; batch uses 0-based game_id
    features["game_id"] = features["GameIndex"] - 1

    merged = features.merge(batch, on="game_id", how="inner")
    print(f"  Merged shape: {merged.shape}")

    out = args.out if os.path.isabs(args.out) else os.path.join(BASE, args.out)
    merged.to_excel(out, index=False)
    print(f"  Saved: {out}")


def cmd_combine(args):
    if args.auto:
        pattern = os.path.join(BASE, "batch*_features_labels.xlsx")
        paths = sorted(glob.glob(pattern))
        if not paths:
            sys.exit("No batch*_features_labels.xlsx files found in the project folder.")
        print(f"Auto-discovered {len(paths)} batch file(s).")
    else:
        paths = [p if os.path.isabs(p) else os.path.join(BASE, p) for p in args.inputs]

    frames = []
    global_offset = 0

    for i, path in enumerate(paths, start=1):
        df = load(path)
        df["batch"] = i

        if global_offset > 0:
            df["game_id"] = df["game_id"] + global_offset
            df["GameIndex"] = df["GameIndex"] + global_offset
            print(f"    game_id / GameIndex offset: +{global_offset}")

        global_offset = df["game_id"].max() + 1
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape : {combined.shape}")
    print(f"game_id range  : {combined['game_id'].min()} - {combined['game_id'].max()}")
    print(f"Batch counts   : {combined['batch'].value_counts().sort_index().to_dict()}")

    out = args.out if os.path.isabs(args.out) else os.path.join(BASE, args.out)
    print(f"Saving to {out} ...")
    combined.to_excel(out, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Selieri data merger")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- merge subcommand ---
    p_merge = sub.add_parser("merge", help="Join a features file with a sim_games batch file")
    p_merge.add_argument("--features", required=True, help="Path to features Excel (e.g. 'features 1.xlsx')")
    p_merge.add_argument("--batch",    required=True, help="Path to sim_games batch Excel (e.g. sim_games/batch1.xlsx)")
    p_merge.add_argument("--out",      required=True, help="Output path (e.g. batch1_features_labels.xlsx)")

    # --- combine subcommand ---
    p_combine = sub.add_parser("combine", help="Stack multiple batchN_features_labels files")
    group = p_combine.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs", nargs="+", help="Explicit list of batch Excel files to combine")
    group.add_argument("--auto",   action="store_true", help="Auto-discover batch*_features_labels.xlsx in project folder")
    p_combine.add_argument("--out", required=True, help="Output path (e.g. combined_features_labels.xlsx)")

    args = parser.parse_args()
    {"merge": cmd_merge, "combine": cmd_combine}[args.cmd](args)


if __name__ == "__main__":
    main()
