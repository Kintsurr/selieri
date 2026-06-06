# feature_extractor.py
# One row per game (single sheet). Preserves all parameters:
#   Per-move: Ply, MoveNo, Side, UCI, SAN, FEN_before, FEN_after, EMT_ms, ClkAfter_s
#   For each engine (SF/LC0) and each depth (10/15/20):
#       Rank, CPL, AdvWP, BestWP, WCL, Ambiguity05, difNextBest, difNextWorst, Sharpness
# Only changes:
#   - Lc0 speed: depth -> short time mapping
#   - Condense per-move sequences into JSON lists per game row
# Everything else is preserved.

import sys, re, math, os, time, datetime as dt, json
import chess, chess.engine, chess.pgn
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
#        CONFIG
# =========================
PGN_FILE  = r"sim_games\batch3.pgn"
OUT_XLSX  = r"batch3.xlsx"

# Engines
ENABLE_STOCKFISH = True
ENABLE_LC0       = True
STOCKFISH_PATH   = r"stockfish\stockfish.exe"
LC0_PATH         = r"Lc0\lc0.exe"

# Depth tiers to collect
DEPTHS   = [10, 15, 20]
MULTIPV  = 5  # PVs to consider (affects Rank/Ambiguity/diffs)

# UCI options
SF_THREADS  = 4
SF_HASH_MB  = 256

LC0_THREADS = 4
LC0_NN_CACHE= 8192
# DO NOT set Ponder for Lc0

# Lc0 "fast mode": map depth->short think time
LC0_TIME_PER_DEPTH = {10: 0.08, 15: 0.15, 20: 0.30}
LC0_PLAYED_MULT    = 1.0  # fallback eval for played move
# Stockfish main depth & tiny fallback for played move
SF_RANK_DEPTH   = 15
SF_PLAYED_NODES = 60000

# Debug & limits
PRINT_PROGRESS = True
VERBOSE_DEBUG  = True
MAX_GAMES = None
MAX_MOVES = None

# =========================
#   LOG HELPERS
# =========================
def ts(): return dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
def dbg(*a):
    if PRINT_PROGRESS: print(f"[{ts()}]", *a, flush=True)
def vdbg(*a):
    if VERBOSE_DEBUG:  print(f"[{ts()}]   ", *a, flush=True)

# =========================
#   CLOCK/TIMING PARSERS
# =========================
def parse_clock_seconds(text):
    m = re.search(r"\[%clk\s+([0-9:.\']+)\]", text or "")
    if not m: return None
    s = m.group(1).strip()
    if "'" in s:
        parts = [int(x) for x in s.split("'")]
        if len(parts) == 2: h,mn,sc = 0, parts[0], parts[1]
        elif len(parts) == 3: h,mn,sc = parts
        else: return None
        return h*3600 + mn*60 + sc
    if s.count(":") == 2:
        h,mn,sec = s.split(":"); return int(h)*3600 + int(mn)*60 + float(sec)
    if s.count(":") == 1:
        mn,sec = s.split(":"); return int(mn)*60 + float(sec)
    try: return float(s)
    except: return None

def parse_emt_ms(text):
    m = re.search(r"\[%emt\s+(\d+)\]", text or "")
    return int(m.group(1)) if m else None

# =========================
#   WIN CHANCES & WDL
# =========================
def logistic_wp_from_cp(cp):
    return 1.0 / (1.0 + math.exp(-0.004 * cp))

def wdl_counts_for_pov(wdl_obj, turn_is_white):
    if wdl_obj is None: return None
    if hasattr(wdl_obj, "pov"):
        try:
            w = wdl_obj.pov(chess.WHITE if turn_is_white else chess.BLACK)
            return (int(w.wins), int(w.draws), int(w.losses))
        except Exception:
            pass
    if all(hasattr(wdl_obj, a) for a in ("wins","draws","losses")):
        return (int(wdl_obj.wins), int(wdl_obj.draws), int(wdl_obj.losses))
    if isinstance(wdl_obj, (list, tuple)) and len(wdl_obj) == 3:
        return (int(wdl_obj[0]), int(wdl_obj[1]), int(wdl_obj[2]))
    return None

def wp_from_wdl_pov(wdl_obj, turn_is_white):
    counts = wdl_counts_for_pov(wdl_obj, turn_is_white)
    if not counts: return None
    W, D, L = counts
    tot = W + D + L
    if tot <= 0: return None
    return (W + 0.5 * D) / float(tot)

def sharpness_from_wdl_counts(wdl_counts):
    if not wdl_counts: return None
    W_raw, _, L_raw = wdl_counts
    W = min(max(float(W_raw)/1000.0, 0.0001), 0.9999)
    L = min(max(float(L_raw)/1000.0, 0.0001), 0.9999)
    a = np.log((1.0/W) - 1.0); b = np.log((1.0/L) - 1.0)
    denom = a + b
    if not np.isfinite(denom) or denom == 0.0: return 0.0
    inv = 2.0/denom
    return float(max(inv, 0.0)**2)

# =========================
#   ENGINE ADAPTER
# =========================
class EngineAdapter:
    def __init__(self, name, path, cfg, mode):
        self.name = name
        self.mode = mode  # "sf" or "lc0"
        self.path = path
        self.proc = None
        self.cfg = cfg

    def start(self):
        dbg(f"{self.name}: starting {self.path}")
        t0 = time.perf_counter()
        self.proc = chess.engine.SimpleEngine.popen_uci(self.path, stderr=sys.stderr)
        if self.cfg: self.proc.configure(self.cfg)
        dt = time.perf_counter() - t0
        dbg(f"{self.name}: ready in {dt:.2f}s; options={self.cfg}")

    def quit(self):
        if self.proc:
            try: self.proc.quit()
            except Exception as e: dbg(f"{self.name}: quit error: {e}")

    # Depth/time mapping
    def limit_rank(self, depth):
        if self.mode == "sf":
            return chess.engine.Limit(depth=depth)
        return chess.engine.Limit(time=LC0_TIME_PER_DEPTH.get(depth, 0.15))

    def limit_played(self, depth):
        if self.mode == "sf":
            return chess.engine.Limit(nodes=SF_PLAYED_NODES)
        return chess.engine.Limit(time=LC0_TIME_PER_DEPTH.get(depth, 0.15)*LC0_PLAYED_MULT)

    # One MultiPV call => list of (head_move, cp, info)
    def multipv_once(self, board, depth, multipv):
        lim = self.limit_rank(depth)
        infos = self.proc.analyse(board, lim, multipv=multipv)
        infos = [infos] if isinstance(infos, dict) else infos
        try:
            infos.sort(key=lambda d: d['score'].relative.score(mate_score=100000), reverse=True)
        except Exception:
            pass
        pairs = []
        for d in infos:
            if 'pv' in d and d['pv'] and 'score' in d:
                head = d['pv'][0]
                cp   = d['score'].relative.score(mate_score=100000)
                pairs.append((head, cp, d))
        if VERBOSE_DEBUG:
            for i,(mv, cp, d) in enumerate(pairs):
                wp = wp_from_wdl_pov(d.get("wdl"), board.turn)
                if wp is None:
                    s = d['score'].pov(board.turn)
                    cp2 = None if s.is_mate() else s.score(mate_score=100000)
                    wp = logistic_wp_from_cp(cp2 if cp2 is not None else 0.0)
                vdbg(f"{self.name}: D{depth} PV#{i+1} {board.san(mv)} {mv.uci()} cp={cp} wp~{wp:.3f} wdl={d.get('wdl')}")
        return pairs

    # Short, targeted eval for played move (no push/pop)
    def eval_played(self, board, depth, played_move):
        lim = self.limit_played(depth)
        info = self.proc.analyse(board, lim, multipv=1, root_moves=[played_move])
        info = info if isinstance(info, dict) else info[0]
        cp   = info['score'].relative.score(mate_score=100000)
        wp   = wp_from_wdl_pov(info.get("wdl"), board.turn) or logistic_wp_from_cp(cp)
        return cp, wp

# =========================
#   IRWIN-LIKE HELPERS
# =========================
def ambiguity_count(wps, best_wp, eps=0.05):
    return sum(1 for wp in wps if wp is not None and best_wp is not None and abs(wp - best_wp) < eps)

def dif_next_best(wps, idx):
    if idx is None or idx == 0: return 0.0
    b = wps[idx-1] if idx-1 >= 0 else None
    h = wps[idx]
    if b is None or h is None: return 0.0
    return b - h

def dif_next_worst(wps, idx):
    if idx is None: return 0.0
    if idx < len(wps)-1:
        w = wps[idx+1]; h = wps[idx]
        if w is None or h is None: return 0.0
        return w - h
    return 0.0

def wcl_val(best_wp, adv_wp):
    if best_wp is None or adv_wp is None: return None
    return max(0.0, best_wp - adv_wp)

# =========================
#   SINGLE-SHEET WRITER
# =========================
def save_games_seq_one_sheet(out_path, rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    tried = []
    for engine_name in ("openpyxl","xlsxwriter"):
        try:
            with pd.ExcelWriter(out_path, engine=engine_name) as w:
                df.to_excel(w, index=False, sheet_name="games_seq")
            dbg(f"Saved single-sheet to {out_path} ({engine_name})")
            return
        except Exception as e:
            tried.append(f"{engine_name}:{type(e).__name__}:{e}")
    base = out_path.rsplit(".",1)[0]
    df.to_csv(base+"_games_seq.csv", index=False)
    dbg("Excel failed; wrote CSV instead:\n" + "\n".join(tried))

# =========================
#        MAIN
# =========================
def analyze_all_games(pgn_path, out_xlsx):
    # Checkpointing: derive JSONL path from out_xlsx
    checkpoint_path = out_xlsx.replace('.xlsx', '_checkpoint.jsonl')

    processed_indices = set()
    all_rows = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    processed_indices.add(row['GameIndex'])
                    all_rows.append(row)
                except Exception:
                    pass
        dbg(f"Resuming: {len(processed_indices)} games already in checkpoint")

    # Start engines
    sf = lc = None
    if ENABLE_STOCKFISH and os.path.isfile(STOCKFISH_PATH):
        sf_cfg = {"Threads": SF_THREADS, "UCI_ShowWDL": True}
        if SF_HASH_MB: sf_cfg["Hash"] = SF_HASH_MB
        sf = EngineAdapter("Stockfish", STOCKFISH_PATH, sf_cfg, "sf"); sf.start()
    else:
        if ENABLE_STOCKFISH: dbg(f"WARNING: Stockfish not found at {STOCKFISH_PATH}")

    if ENABLE_LC0 and os.path.isfile(LC0_PATH):
        lc_cfg = {"Threads": LC0_THREADS, "UCI_ShowWDL": True, "NNCacheSize": LC0_NN_CACHE}
        lc = EngineAdapter("Lc0", LC0_PATH, lc_cfg, "lc0"); lc.start()
    else:
        if ENABLE_LC0: dbg(f"WARNING: Lc0 not found at {LC0_PATH}")

    if not sf and not lc:
        dbg("No engines available. Aborting."); return

    engines_present = []
    if sf: engines_present.append("SF")
    if lc: engines_present.append("LC0")

    # Column schema (one row per game; JSON lists for sequences)
    meta_cols = ["GameIndex","Date","Event","EventRounds","Round","White","Black","Result","WhiteElo","BlackElo"]
    seq_basic = ["Ply","MoveNo","Side","UCI","SAN","FEN_before","FEN_after","EMT_ms","ClkAfter_s"]
    seq_engine = []
    for eng in engines_present:
        for d in DEPTHS:
            for m in ("Rank","CPL","AdvWP","BestWP","WCL","Ambiguity05","difNextBest","difNextWorst","Sharpness"):
                seq_engine.append(f"{eng}_D{d}_{m}")
    all_cols = meta_cols + seq_basic + seq_engine

    # Count games
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        total = sum(1 for _ in chess.pgn.read_headers(f))
    dbg(f"Total games: {total}")

    # Analyze
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        gi = 0
        while True:
            if MAX_GAMES is not None and gi >= MAX_GAMES: break
            game = chess.pgn.read_game(f)
            if game is None: break
            gi += 1
            dbg(f"\n=== Game {gi}/{total} ===")
            if gi in processed_indices:
                dbg(f"  Already processed, skipping.")
                continue
            vdbg(f"Headers: {dict(game.headers)}")

            # Metadata
            ds = game.headers.get("Date","")
            try:
                date_val = datetime.strptime(ds, "%Y.%m.%d") if re.match(r"^\d{4}\.\d{2}\.\d{2}$", ds) else datetime.now()
            except: date_val = datetime.now()
            meta = dict(
                GameIndex=gi,
                Date=date_val,
                Event=game.headers.get("Event",""),
                EventRounds=game.headers.get("EventRounds",""),
                Round=game.headers.get("Round",""),
                White=game.headers.get("White",""),
                Black=game.headers.get("Black",""),
                Result=game.headers.get("Result",""),
                WhiteElo=game.headers.get("WhiteElo","?"),
                BlackElo=game.headers.get("BlackElo","?"),
            )

            # Per-move accumulators
            acc = {k: [] for k in (seq_basic + seq_engine)}

            board = game.board()
            node  = game
            ply   = 0
            last_clock = {True: None, False: None}

            while node.variations:
                if MAX_MOVES is not None and ply >= MAX_MOVES: break
                nxt  = node.variation(0)
                move = nxt.move
                turn_white = board.turn
                ply += 1

                san        = board.san(move)
                fen_before = board.fen()
                comment    = nxt.comment

                clk = parse_clock_seconds(comment)
                emt = parse_emt_ms(comment)
                emt_ms = emt if emt is not None else (None if last_clock[turn_white] is None or clk is None else int(1000*(last_clock[turn_white]-clk)))
                if clk is not None: last_clock[turn_white] = clk

                dbg(f"Move #{ply} ({'W' if turn_white else 'B'}): {san} [{move.uci()}]")
                vdbg(f"  FEN before: {fen_before}")
                vdbg(f"  EMT_ms={emt_ms}  ClkAfter_s={last_clock[turn_white]}  Comment={comment!r}")

                # Basics
                acc["Ply"].append(ply)
                acc["MoveNo"].append((ply+1)//2)
                acc["Side"].append("W" if turn_white else "B")
                acc["UCI"].append(move.uci())
                acc["SAN"].append(san)
                acc["FEN_before"].append(fen_before)
                acc["EMT_ms"].append(emt_ms)
                acc["ClkAfter_s"].append(last_clock[turn_white])

                # Engines x depths (preserve ALL parameters)
                def compute_for_engine(adapter, tag):
                    for d in DEPTHS:
                        # 1) One multipv call
                        try:
                            pairs = adapter.multipv_once(board, d, MULTIPV)
                        except Exception as e:
                            dbg(f"{adapter.name} D{d} MultiPV error: {e}")
                            pairs = []

                        # 2) Winning probs from those pairs
                        wps = []
                        for mv, cp, info in pairs:
                            wp = wp_from_wdl_pov(info.get("wdl"), turn_white) or logistic_wp_from_cp(cp)
                            wps.append(wp)

                        best_wp = wps[0] if wps else None
                        best_cp = pairs[0][1] if pairs else None

                        # 3) Rank + played move (fallback if not in PV heads)
                        rank = 4; played_idx = None
                        adv_cp = adv_wp = None
                        if pairs:
                            seq = [mv for mv,_,_ in pairs]
                            if move in seq:
                                played_idx = seq.index(move)
                                rank = played_idx + 1
                                adv_cp = pairs[played_idx][1]
                                adv_wp = wps[played_idx]
                        if adv_cp is None or adv_wp is None:
                            try:
                                adv_cp, adv_wp = adapter.eval_played(board, d, move)
                            except Exception as e:
                                dbg(f"{adapter.name} D{d} played-eval error: {e}")
                                adv_cp = adv_wp = None

                        # 4) Derived Irwin-style params (preserved)
                        cpl = abs(best_cp - adv_cp) if (best_cp is not None and adv_cp is not None) else None
                        wcl = wcl_val(best_wp, adv_wp)
                        amb = ambiguity_count(wps, best_wp, eps=0.05) if (wps and best_wp is not None) else None
                        dnb = dif_next_best(wps, played_idx)
                        dnw = dif_next_worst(wps, played_idx)
                        sharp = None
                        if pairs:
                            counts = wdl_counts_for_pov(pairs[0][2].get("wdl"), turn_white)
                            if counts: sharp = sharpness_from_wdl_counts(counts)

                        # 5) Store per-move for this enginexdepth
                        acc[f"{tag}_D{d}_Rank"].append(rank)
                        acc[f"{tag}_D{d}_CPL"].append(cpl)
                        acc[f"{tag}_D{d}_AdvWP"].append(adv_wp)
                        acc[f"{tag}_D{d}_BestWP"].append(best_wp)
                        acc[f"{tag}_D{d}_WCL"].append(wcl)
                        acc[f"{tag}_D{d}_Ambiguity05"].append(amb)
                        acc[f"{tag}_D{d}_difNextBest"].append(dnb)
                        acc[f"{tag}_D{d}_difNextWorst"].append(dnw)
                        acc[f"{tag}_D{d}_Sharpness"].append(sharp)

                        vdbg(f"  {adapter.name} D{d}: Rank={rank} CPL={cpl} AdvWP={adv_wp} BestWP={best_wp} WCL={wcl} Amb={amb} dNB={dnb} dNW={dnw} Sharp={sharp}")

                if sf:  compute_for_engine(sf,  "SF")
                if lc:  compute_for_engine(lc, "LC0")

                # Push & capture FEN_after
                board.push(move)
                acc["FEN_after"].append(board.fen())

                node = nxt
                vdbg("-"*60)

            # Finalize one row with JSON lists
            out_row = {**meta}
            jdump = lambda x: json.dumps(x, ensure_ascii=False)
            for key in seq_basic + seq_engine:
                if key not in acc: acc[key] = []
                out_row[key] = jdump(acc[key])

            all_rows.append(out_row)
            # Write checkpoint immediately so progress is never lost
            with open(checkpoint_path, 'a', encoding='utf-8') as cf:
                cf.write(json.dumps(out_row, default=str, ensure_ascii=False) + '\n')
            dbg(f"  Checkpoint saved ({len(all_rows)} games done so far)")

    # Save single sheet
    save_games_seq_one_sheet(out_xlsx, all_rows, all_cols)

    # Quit engines
    if sf: sf.quit()
    if lc: lc.quit()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", default=PGN_FILE)
    parser.add_argument("--out", default=OUT_XLSX)
    args = parser.parse_args()
    analyze_all_games(args.pgn, args.out)
