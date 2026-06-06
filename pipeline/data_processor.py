"""
process_all_batches.py

Continuous orchestrator for the Selieri chess cheating detection pipeline.

Workflow per loop iteration:
  1. Discover batch PGN + label pairs in sim_games/
  2. For each batch: extract features (irwin hybrid.py), merge with labels
  3. Build unified_features_labels.xlsx from all finished batches
  4. Sleep 30 minutes, then repeat to pick up new batches
"""

import sys
import os
import re
import time
import subprocess
import logging
import datetime
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT      = Path(__file__).parent.resolve()          # c:\...\Selieri
SIM_DIR   = ROOT / "sim_games"
IRWIN     = ROOT / "feature_extractor.py"            # per-move SF+LC0 feature extractor
LOG_FILE  = ROOT / "processing_log.txt"
UNIFIED   = ROOT / "unified_features_labels.xlsx"

SLEEP_SECONDS = 30 * 60   # 30 minutes between re-scans

# ─────────────────────────────────────────────
#  LOGGING  (console + file)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
log = logging.getLogger("orchestrator")


def banner(msg: str) -> None:
    line = "=" * 70
    log.info(line)
    log.info(msg)
    log.info(line)


# ─────────────────────────────────────────────
#  BATCH DISCOVERY
# ─────────────────────────────────────────────
def discover_batches() -> list[dict]:
    """
    Returns a sorted list of dicts:
        { 'n': int, 'pgn': Path, 'labels': Path }

    Scans sim_games/ for batch*.pgn paired with batch*.xlsx.
    Also handles simulated_games_with_time.pgn as batch 0 when
    sim_games/batch0.xlsx exists.
    """
    if not SIM_DIR.exists():
        log.warning(f"sim_games directory not found: {SIM_DIR}")
        return []

    batches: dict[int, dict] = {}

    # Standard batchN.pgn / batchN.xlsx pairs
    for pgn_path in sorted(SIM_DIR.glob("batch*.pgn")):
        m = re.fullmatch(r"batch(\d+)\.pgn", pgn_path.name, re.IGNORECASE)
        if not m:
            continue
        n = int(m.group(1))
        labels_path = SIM_DIR / f"batch{n}.xlsx"
        if labels_path.exists():
            batches[n] = {"n": n, "pgn": pgn_path, "labels": labels_path}
        else:
            log.debug(f"batch{n}.pgn found but batch{n}.xlsx missing — skipping")

    # Optional batch 0: simulated_games_with_time.pgn
    legacy_pgn = SIM_DIR / "simulated_games_with_time.pgn"
    if legacy_pgn.exists() and 0 not in batches:
        labels_path = SIM_DIR / "batch0.xlsx"
        if labels_path.exists():
            batches[0] = {"n": 0, "pgn": legacy_pgn, "labels": labels_path}
            log.info("Found legacy simulated_games_with_time.pgn as batch0")
        else:
            log.info(
                "simulated_games_with_time.pgn present but no batch0.xlsx "
                "— feature extraction only (no merge)"
            )

    return [batches[k] for k in sorted(batches)]


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
def features_path(n: int) -> Path:
    return ROOT / f"batch{n}_features.xlsx"


def merged_path(n: int) -> Path:
    return ROOT / f"batch{n}_features_labels.xlsx"


def checkpoint_path(n: int) -> Path:
    return ROOT / f"batch{n}_features_checkpoint.jsonl"


def run_irwin(pgn: Path, out: Path, batch_n: int) -> bool:
    """
    Run irwin hybrid.py as a subprocess.
    Returns True if the subprocess exited 0, False otherwise.
    The checkpoint file means partial progress is always preserved.
    """
    log.info(f"  Launching irwin hybrid.py for batch{batch_n} ...")
    log.info(f"    --pgn {pgn}")
    log.info(f"    --out {out}")

    result = subprocess.run(
        [sys.executable, str(IRWIN), "--pgn", str(pgn), "--out", str(out)],
        cwd=str(ROOT),
        check=False,          # do NOT raise on failure; checkpoint preserves progress
    )

    if result.returncode != 0:
        log.error(
            f"  irwin hybrid.py exited with code {result.returncode} for batch{batch_n}. "
            f"Checkpoint preserved — will resume next time."
        )
        return False

    log.info(f"  irwin hybrid.py finished OK for batch{batch_n}")
    return True


# ─────────────────────────────────────────────
#  MERGE  (features + labels)
# ─────────────────────────────────────────────
def merge_features_labels(features_xlsx: Path, labels_xlsx: Path, out_xlsx: Path) -> bool:
    """
    Merge features (GameIndex 1-based) with labels (game_id 0-based).
    Uses inner join on game_id = GameIndex - 1.
    Saves to out_xlsx.  Returns True on success.
    """
    log.info(f"  Merging features + labels -> {out_xlsx.name}")
    try:
        features_df = pd.read_excel(str(features_xlsx))
        labels_df   = pd.read_excel(str(labels_xlsx))

        # Align indices: features are 1-based, labels are 0-based
        features_df["game_id"] = features_df["GameIndex"] - 1

        merged = pd.merge(features_df, labels_df, on="game_id", how="inner")
        log.info(f"    Features rows: {len(features_df)}  Labels rows: {len(labels_df)}  Merged rows: {len(merged)}")

        merged.to_excel(str(out_xlsx), index=False)
        log.info(f"    Saved merged dataset to {out_xlsx}")
        return True

    except Exception as exc:
        log.error(f"  Merge failed for {out_xlsx.name}: {exc}")
        return False


# ─────────────────────────────────────────────
#  UNIFIED DATASET
# ─────────────────────────────────────────────
def build_unified(batch_numbers: list[int]) -> None:
    """
    Concatenate all batch{N}_features_labels.xlsx files into
    unified_features_labels.xlsx.
    """
    frames = []
    for n in sorted(batch_numbers):
        p = merged_path(n)
        if p.exists():
            try:
                df = pd.read_excel(str(p))
                df["source_batch"] = n
                frames.append(df)
                log.info(f"  Loaded {p.name}: {len(df)} rows")
            except Exception as exc:
                log.error(f"  Could not read {p.name}: {exc}")

    if not frames:
        log.warning("  No merged batch files available — unified dataset not written.")
        return

    unified = pd.concat(frames, ignore_index=True)
    unified.to_excel(str(UNIFIED), index=False)
    log.info(f"  Unified dataset saved: {UNIFIED}  ({len(unified)} total rows)")


# ─────────────────────────────────────────────
#  PER-BATCH PIPELINE
# ─────────────────────────────────────────────
def process_batch(batch: dict) -> bool:
    """
    Full pipeline for one batch.  Returns True if merged output now exists.
    """
    n       = batch["n"]
    pgn     = batch["pgn"]
    labels  = batch["labels"]
    feat    = features_path(n)
    merged  = merged_path(n)

    banner(f"Processing batch{n}")

    # ── Already fully done?
    if merged.exists():
        log.info(f"  batch{n}: merged output already exists, skipping.")
        return True

    # ── Features extraction needed?
    if not feat.exists():
        log.info(f"  batch{n}: features file missing — running irwin hybrid.py")
        ok = run_irwin(pgn, feat, n)
        if not ok or not feat.exists():
            log.error(f"  batch{n}: feature extraction did not produce {feat.name} — will retry later")
            return False
    else:
        log.info(f"  batch{n}: features file already exists ({feat.name}), skipping extraction")

    # ── Merge
    ok = merge_features_labels(feat, labels, merged)
    return ok


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main() -> None:
    banner("Selieri batch orchestrator starting")
    log.info(f"ROOT       : {ROOT}")
    log.info(f"SIM_DIR    : {SIM_DIR}")
    log.info(f"irwin script: {IRWIN}")
    log.info(f"Log file   : {LOG_FILE}")

    iteration = 0
    known_batches: set[int] = set()

    while True:
        iteration += 1
        banner(f"Scan iteration {iteration}  ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

        batches = discover_batches()

        if not batches:
            log.warning("No batches with both PGN and labels found in sim_games/")
        else:
            log.info(f"Discovered {len(batches)} batch(es): {[b['n'] for b in batches]}")

        new_batches = [b for b in batches if b["n"] not in known_batches]
        if new_batches:
            log.info(f"New batch(es) this scan: {[b['n'] for b in new_batches]}")

        # Process all batches (new or previously failed)
        finished: list[int] = []
        for batch in batches:
            ok = process_batch(batch)
            if ok:
                finished.append(batch["n"])
                known_batches.add(batch["n"])

        # Build / rebuild unified dataset whenever at least one batch is done
        if finished:
            banner("Building unified dataset")
            build_unified(finished)
        else:
            log.info("No batches fully finished yet — unified dataset not updated.")

        # Summary
        log.info(
            f"Iteration {iteration} complete. "
            f"Done batches: {sorted(known_batches)}. "
            f"Sleeping {SLEEP_SECONDS // 60} minutes before next scan ..."
        )

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user. Exiting.")
