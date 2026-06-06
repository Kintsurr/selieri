#!/usr/bin/env python3
# requirements: pip install python-chess pandas tqdm openpyxl
import chess, chess.pgn, chess.engine
from chess import polyglot
import random, math, json, sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

LC0_BIN = r"Lc0\lc0.exe"
STOCKFISH_BIN = r"stockfish\stockfish.exe"

OUT_DIR = Path("sim_games"); OUT_DIR.mkdir(exist_ok=True)
PGN_PATH = OUT_DIR / "batch4.pgn"
LABELS_XLSX = OUT_DIR / "batch4.xlsx"

# Total games across both phases = 5000 + 5000 = 10000
NUM_GAMES_CHEAT = 500
NUM_GAMES_CLEAN = 500
MAX_MOVES = 100

BOOK_PLIES = 12
POLYGLOT_BOOK = None
BOOK_RANDOM_THINK = (0.5, 2.0)

BUILTIN_BOOK = {
    "start": [("e4", 0.32), ("d4", 0.30), ("Nf3", 0.20), ("c4", 0.15), ("b3", 0.03)],
    "e4":    [("c5", 0.37), ("e5", 0.35), ("e6", 0.10), ("c6", 0.10), ("d6", 0.05), ("g6", 0.03)],
    "d4":    [("Nf6", 0.40), ("d5", 0.32), ("g6", 0.12), ("e6", 0.10), ("c5", 0.06)],
    "Nf3":   [("d5", 0.30), ("Nf6", 0.35), ("c5", 0.12), ("g6", 0.10), ("e6", 0.08), ("d6", 0.05)],
    "c4":    [("e5", 0.18), ("Nf6", 0.30), ("c5", 0.22), ("g6", 0.15), ("e6", 0.10), ("c6", 0.05)],
    "after_e4_e5_white": [("Nf3", 0.60), ("Nc3", 0.15), ("Bc4", 0.15), ("d4", 0.10)],
    "after_e4_c5_white": [("Nf3", 0.55), ("Nc3", 0.18), ("c3", 0.12), ("d4", 0.15)],
    "after_d4_d5_white": [("c4", 0.60), ("Nf3", 0.22), ("e3", 0.10), ("Bf4", 0.08)],
    "after_d4_Nf6_white": [("c4", 0.50), ("Nf3", 0.30), ("g3", 0.12), ("Nc3", 0.08)],
}

# Cheating controls
FORCE_CHEAT_AFTER_PLY = 16
WORSE_BY_WDL = 0.10
SHARPNESS_WDL_L1 = 0.35   # tuned for L1; for LC0 sharpness you may re-tune threshold
CRITICAL_PROB = 0.5
ADVANTAGE_CP = 180

# Engine configs
MAIA_ANALYSE_TIME = 1.0
SF_MOVE_TIME = 0.40
MULTIPV = 6

# Policy / sampling
TOPK = 3
TAU_CP = 60.0
EPSILON = 0.08

PLAN_HORIZON_OWN = 3
PLAN_STICKINESS = 0.9
PLAN_TOLERANCE_CP = 50
TIME_PRESSURE_STICKINESS_DROP = 0.6

# Safety, anti-blunder
USE_SF_BLUNDERCHECK = True
BLUNDER_CHECK_TOPK = 4
BLUNDER_DROP_CP = 180
SANITY_TIME = 0.12
ONLY_MOVE_GREEDY = True

# Anti-shuffle penalties
SHUFFLE_BACKTRACK_CP = 120
SHUFFLE_RETURN_ORIGIN_CP = 80
SHUFFLE_SAME_PIECE_3X_CP = 60
FORCED_KICKBACK_RELIEF = True

# Time controls
START_SECONDS = 10*60
INCREMENT_SECONDS = 2.0
BASE_OPENING = 1.3
BASE_MIDDLEGAME = 3.0
BASE_ENDGAME = 2.1
MAX_EMT = 28.0
MIN_EMT = 0.25

RANDOM_SEED = None
if RANDOM_SEED is not None: random.seed(RANDOM_SEED)
DEBUG = True
def dbg(*args):
    if DEBUG: print(*args, flush=True)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def configure_lc0(eng):
    for opt in ("UCI_ShowWDL", "UCI_ShowWDLs", "UCI_AnalyseWDL"):
        try: eng.configure({opt: True})
        except: pass
    try: eng.configure({"Threads": 2, "MultiPV": max(3, MULTIPV)})
    except Exception as e: dbg(f"[WARN] LC0 configure(Threads/MultiPV) failed: {e}")
    try: eng.ping()
    except Exception as e: dbg(f"[WARN] LC0 ping failed: {e}")
    return eng

def configure_sf(eng):
    try: eng.configure({"Threads": 2, "Hash": 256, "UCI_ShowWDL": True})
    except Exception as e: dbg(f"[WARN] SF configure failed: {e}")
    try: eng.ping()
    except Exception as e: dbg(f"[WARN] SF ping failed: {e}")
    return eng

def pov_cp(info_or_score, pov_color):
    s = info_or_score.get("score") if isinstance(info_or_score, dict) else info_or_score
    if s is None:
        return 0
    try:
        s = s.pov(pov_color)
    except Exception:
        pass
    try:
        return s.score(mate_score=100000)
    except Exception:
        try:
            m = s.mate()
            if m is None:
                return 0
            return 100000 if m > 0 else -100000
        except Exception:
            return 0

def info_wdl_tuple(info, stm):
    try:
        wdl_obj = info.get("wdl")
        if wdl_obj is None: return None
        try: pov = wdl_obj.pov(stm)
        except Exception: pov = wdl_obj
        w, d, l = pov.wdl()
        tot = max(1.0, w + d + l)
        return (w/tot, d/tot, l/tot)
    except Exception:
        return None

def expected_score_from_wdl(wdl):
    if not wdl: return 0.5
    w,d,l = wdl
    return w + 0.5*d

def entropy_wdl(wdl):
    if not wdl: return 0.0
    w,d,l = wdl
    e = 0.0
    for p in (w,d,l):
        if p > 0: e -= p*math.log2(p)
    return e

def l1_dist(p,q): return sum(abs(a-b) for a,b in zip(p,q))

def analyse_multipv(engine, board, t, multipv):
    if board.is_game_over(): return []
    out, seen = [], set()
    try:
        infos = engine.analyse(board, chess.engine.Limit(time=t), multipv=multipv)
        if not isinstance(infos, list): infos = [infos]
        infos.sort(key=lambda x: x.get("multipv", 1))
        for info in infos:
            pv = info.get("pv") or []
            mv = pv[0] if pv else None
            cp = pov_cp(info, board.turn)
            wdl = info_wdl_tuple(info, board.turn)
            if mv and mv not in seen:
                out.append((mv, cp, wdl, pv)); seen.add(mv)
    except Exception as e:
        dbg(f"[WARN] analyse_multipv: engine.analyse failed: {e}")
        out = []
    if not out:
        try:
            r = engine.play(board, chess.engine.Limit(time=t))
            if r and r.move:
                pv = [r.move]
                try:
                    quick = engine.analyse(board, chess.engine.Limit(time=min(0.2, max(0.05, t*0.25))))
                    if isinstance(quick, dict) and quick.get("pv") and quick["pv"][0] == r.move:
                        pv = quick["pv"]
                except Exception: pass
                out = [(r.move, 0, None, pv)]
        except Exception as e:
            dbg(f"[WARN] analyse_multipv: play fallback failed: {e}")
    return out

def sample_weighted(options):
    tot = sum(w for _,w in options)
    if tot <= 0: return random.choice([m for m,_ in options])
    r = random.random()*tot; acc = 0.0
    for item,w in options:
        acc += w
        if r <= acc: return item
    return options[-1][0]

def book_move_polyglot(board, path):
    try:
        with polyglot.open_reader(path) as reader:
            entries = list(reader.find_all(board))
        if not entries: return None
        options = [(e.move(), e.weight) for e in entries]
        return sample_weighted(options)
    except Exception:
        return None

def book_move_builtin(board):
    if board.fullmove_number == 1 and board.turn == chess.WHITE:
        san = sample_weighted(BUILTIN_BOOK["start"])
        try: return board.parse_san(san)
        except: return None
    last_san = None
    try: last_san = board.san(board.peek())
    except Exception: pass
    if last_san in ("e4","d4","Nf3","c4") and board.turn == chess.BLACK:
        table = BUILTIN_BOOK.get(last_san)
        if table:
            mv_san = sample_weighted(table)
            try: return board.parse_san(mv_san)
            except: return None
    if board.fullmove_number == 2 and board.turn == chess.WHITE:
        try:
            w1 = board.move_stack[0]; b1 = board.move_stack[1]
            san_w1 = chess.Board().san(w1)
            b_after = chess.Board(); b_after.push(w1); san_b1 = b_after.san(b1)
        except Exception:
            san_w1 = san_b1 = None
        key = None
        if san_w1=="e4" and san_b1=="e5": key="after_e4_e5_white"
        elif san_w1=="e4" and san_b1=="c5": key="after_e4_c5_white"
        elif san_w1=="d4" and san_b1=="d5": key="after_d4_d5_white"
        elif san_w1=="d4" and san_b1=="Nf6": key="after_d4_Nf6_white"
        if key and BUILTIN_BOOK.get(key):
            mv_san = sample_weighted(BUILTIN_BOOK[key])
            try: return board.parse_san(mv_san)
            except: return None
    return None

def try_book_move(board):
    if BOOK_PLIES <= 0 or board.ply() >= BOOK_PLIES:
        return None
    if POLYGLOT_BOOK and Path(POLYGLOT_BOOK).exists():
        mv = book_move_polyglot(board, POLYGLOT_BOOK)
        if mv: return mv
    return book_move_builtin(board)

def is_forced_kickback(board, sq, color):
    if not FORCED_KICKBACK_RELIEF: return False
    opp = not color
    attackers = list(board.attackers(opp, sq))
    if not attackers: return False
    defenders = list(board.attackers(color, sq))
    return len(defenders) < len(attackers)

def shuffle_penalty(board, move, hist_state):
    p = board.piece_at(move.from_square)
    if not p: return 0
    color = p.color; ptype = p.piece_type
    penalty = 0
    last = hist_state.get(("last_pair", color))
    if last and last == (move.to_square, move.from_square):
        penalty += SHUFFLE_BACKTRACK_CP
    start = chess.Board()
    if ptype in (chess.KNIGHT, chess.BISHOP):
        to_piece = start.piece_at(move.to_square)
        if to_piece and to_piece.piece_type == ptype and to_piece.color == color:
            if not is_forced_kickback(board, move.from_square, color):
                penalty += SHUFFLE_RETURN_ORIGIN_CP
    same_cnt = hist_state.get(("same_piece_streak", color), 0)
    last_piece = hist_state.get(("last_piece", color))
    cur_key = (ptype, move.from_square)
    if last_piece == cur_key and same_cnt >= 2:
        penalty += SHUFFLE_SAME_PIECE_3X_CP
    return penalty

def update_shuffle_state(board_before, move, hist_state):
    p = board_before.piece_at(move.from_square)
    if not p: return
    color = p.color; ptype = p.piece_type
    hist_state[("last_pair", color)] = (move.from_square, move.to_square)
    last_piece = hist_state.get(("last_piece", color))
    cur_key = (ptype, move.from_square)
    if last_piece == cur_key:
        hist_state[("same_piece_streak", color)] = hist_state.get(("same_piece_streak", color), 1) + 1
    else:
        hist_state[("last_piece", color)] = (ptype, move.to_square)
        hist_state[("same_piece_streak", color)] = 1

def softmax_sample_cp(cands_cp, tau_cp, epsilon):
    if not cands_cp: return None, None
    sorted_cp = sorted(cands_cp, key=lambda x: x[1], reverse=True)
    best_mv, best_pv = sorted_cp[0][0], sorted_cp[0][3]
    if random.random() < (1.0 - epsilon): return best_mv, best_pv
    scores = [c[1] for c in cands_cp]
    m = max(scores); exps = [math.exp((s - m)/max(1e-9, tau_cp)) for s in scores]
    tot = sum(exps); r = random.random()*tot; acc = 0.0
    for (mv, _, _, pv), w in zip(cands_cp, exps):
        acc += w
        if r <= acc: return mv, pv
    return best_mv, best_pv

def choose_maia_move_with_intent_WDL(engine, board, intent_move, clk_remain_s, hist_state):
    cands = analyse_multipv(engine, board, MAIA_ANALYSE_TIME, MULTIPV)
    if not cands:
        ms = list(board.legal_moves)
        return (random.choice(list(ms)) if ms else None), [], [], 0
    cands_pen = []
    for mv, cp, wdl, pv in cands:
        pen = shuffle_penalty(board, mv, hist_state)
        cands_pen.append((mv, cp - pen, wdl, pv, pen))
    have_wdl = any(c[2] is not None for c in cands_pen)
    if have_wdl:
        cands_sorted = sorted(
            cands_pen,
            key=lambda x: (expected_score_from_wdl(x[2]) if x[2] else 0.5, x[1]),
            reverse=True
        )
    else:
        cands_sorted = sorted(cands_pen, key=lambda x: x[1], reverse=True)
    if DEBUG:
        preview = []
        for mv, cp_adj, wdl, pv, pen in cands_sorted[:min(4, len(cands_sorted))]:
            e = expected_score_from_wdl(wdl) if wdl else None
            preview.append(f"{mv.uci()}: cp_adj={cp_adj:+d}" + (f",E={e:.3f}" if e is not None else "") + (f",pen={pen}" if pen else ""))
        dbg("[CANDS]", "; ".join(preview))
    if ONLY_MOVE_GREEDY:
        legal = len(list(board.legal_moves))
        if board.is_check() or legal <= 2:
            chosen_mv, chosen_pv, chosen_pen = cands_sorted[0][0], cands_sorted[0][3], cands_sorted[0][4]
        else:
            chosen_mv = chosen_pv = None; chosen_pen = 0
    else:
        chosen_mv = chosen_pv = None; chosen_pen = 0
    if chosen_mv is None:
        intent_ok = intent_move in board.legal_moves if intent_move else False
        best_cp_adj = cands_sorted[0][1]
        if intent_ok:
            row = next(((mv, cp_adj, wdl, pv, pen) for (mv, cp_adj, wdl, pv, pen) in cands_sorted if mv == intent_move), None)
            if row:
                intent_cp_adj, intent_pen = row[1], row[4]
            else:
                try:
                    b2 = board.copy(); b2.push(intent_move)
                    info = engine.analyse(b2, chess.engine.Limit(time=max(0.15, MAIA_ANALYSE_TIME*0.4)))
                    intent_cp_adj = pov_cp(info, chess.WHITE) if info else -9999
                except: intent_cp_adj = -9999
                intent_pen = 0
            stick = PLAN_STICKINESS if clk_remain_s > 30 else PLAN_STICKINESS*TIME_PRESSURE_STICKINESS_DROP
            if intent_pen > 0: stick *= 0.5
            if (best_cp_adj - intent_cp_adj) <= PLAN_TOLERANCE_CP and random.random() < stick:
                chosen_mv, chosen_pv, chosen_pen = intent_move, (row[3] if row else [intent_move]), intent_pen
            else:
                topk = cands_sorted[:max(1, TOPK)]
                topk_plain = [(mv, cp_adj, wdl, pv) for (mv, cp_adj, wdl, pv, pen) in topk]
                chosen_mv, chosen_pv = softmax_sample_cp(topk_plain, TAU_CP, EPSILON)
                chosen_pen = next((pen for (mv, cp_adj, wdl, pv, pen) in topk if mv == chosen_mv), 0)
        else:
            topk = cands_sorted[:max(1, TOPK)]
            topk_plain = [(mv, cp_adj, wdl, pv) for (mv, cp_adj, wdl, pv, pen) in topk]
            chosen_mv, chosen_pv = softmax_sample_cp(topk_plain, TAU_CP, EPSILON)
            chosen_pen = next((pen for (mv, cp_adj, wdl, pv, pen) in topk if mv == chosen_mv), 0)
    future_intent = []
    if chosen_pv:
        for i, mv in enumerate(chosen_pv):
            if i % 2 == 0: future_intent.append(mv)
            if len(future_intent) >= PLAN_HORIZON_OWN: break
        if future_intent: future_intent = future_intent[1:]
    return chosen_mv, future_intent, [(mv,cp,wdl,pv) for (mv,cp,wdl,pv,pen) in cands_sorted], chosen_pen

def two_ply_eval_sf(sf, board, move, t=SANITY_TIME):
    if move not in board.legal_moves: return -99999
    b = board.copy(); side = b.turn
    try:
        b.push(move)
        reply = sf.play(b, chess.engine.Limit(time=t)).move
        if reply: b.push(reply)
        info = sf.analyse(b, chess.engine.Limit(time=t))
        cp_w = pov_cp(info, chess.WHITE)
        return cp_w if side == chess.WHITE else -cp_w
    except Exception:
        return -99999

def blunder_filter(sf, board, chosen_mv, cands_sorted):
    if not USE_SF_BLUNDERCHECK or board.is_game_over(): return chosen_mv
    mvset = []
    for mv,cp,wdl,pv in cands_sorted[:BLUNDER_CHECK_TOPK]:
        if mv not in mvset: mvset.append(mv)
    if chosen_mv not in mvset: mvset.append(chosen_mv)
    results = [(mv, two_ply_eval_sf(sf, board, mv)) for mv in mvset]
    results.sort(key=lambda x: x[1], reverse=True)
    best_mv, best_val = results[0]
    chosen_val = next((v for m,v in results if m == chosen_mv), None)
    if chosen_val is None or (best_val - chosen_val) > BLUNDER_DROP_CP:
        if DEBUG:
            msg = "eval fail" if chosen_val is None else f"delta={best_val - chosen_val:+}"
            dbg(f"[BLUNDER] Swap {chosen_mv.uci()} -> {best_mv.uci()} ({msg})")
        return best_mv
    return chosen_mv

def game_phase(board):
    pawns = sum(1 for p in board.piece_map().values() if p.piece_type == chess.PAWN)
    queens = any(p.piece_type == chess.QUEEN for p in board.piece_map().values())
    minors_majors = sum(1 for p in board.piece_map().values()
                        if p.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
    if minors_majors <= 4 and (not queens) and pawns <= 10: return "end"
    if minors_majors >= 8: return "mid"
    if board.fullmove_number <= 12: return "open"
    return "mid"

def sigmoid_from_cp(cp):
    x = cp / 120.0
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    else:
        ex = math.exp(max(x, -60.0))
        return ex / (1.0 + ex)

def human_think_time(board, sharp_l1, wdl_top1, clk_remaining_s, book=False):
    if book:
        lo, hi = BOOK_RANDOM_THINK
        t = random.uniform(lo, hi)
        if clk_remaining_s - t < 1.0: t = max(MIN_EMT, clk_remaining_s - 1.0)
        return max(MIN_EMT, min(t, MAX_EMT))
    base = {"open": BASE_OPENING, "mid": BASE_MIDDLEGAME, "end": BASE_ENDGAME}[game_phase(board)]
    ent = entropy_wdl(wdl_top1) if wdl_top1 else 0.8
    sharp_bonus = min(1.0 + (sharp_l1 / 0.5), 2.2)
    uncertainty = 1.0 + max(0.0, (ent - 0.8) / 0.8)
    time_pressure = 0.7 + 0.3 * min(1.0, clk_remaining_s / (START_SECONDS*0.5))
    mu = base * sharp_bonus * uncertainty * time_pressure
    t = random.lognormvariate(math.log(max(1e-6, mu)), 0.55)
    t = max(MIN_EMT, min(t, MAX_EMT))
    if clk_remaining_s - t < 1.0: t = max(MIN_EMT, clk_remaining_s - 1.0)
    return max(MIN_EMT, t)

def fmt_clock(sec):
    if sec < 0: sec = 0
    h = int(sec // 3600); m = int((sec % 3600)//60); s = int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}" if h>0 else f"{m}:{s:02d}"

def append_clock_comment(node, emt_s, clk_after_s):
    node.comment = (node.comment + " " if node.comment else "") + f"[%emt 0:{int(emt_s)//60:02d}:{int(emt_s)%60:02d}] [%clk {fmt_clock(clk_after_s)}]"

# ---- LC0 sharpness implementation (from your first script) ----
def sharpness_lc0_from_wdl(wdl):
    """LC0 sharpness from a single WDL tuple (w,d,l) with w+d+l=1 (white POV)."""
    if not wdl:
        return 0.0
    w, _, l = wdl
    W = min(max(w, 1e-4), 0.9999)
    L = min(max(l, 1e-4), 0.9999)
    denom = math.log((1.0 / W) - 1.0) + math.log((1.0 / L) - 1.0)
    val = 0.0 if denom == 0 else 2.0 / denom
    return max(val, 0.0) ** 2
# ---------------------------------------------------------------

def measure_sharpness_and_eval_WDL(engine, board):
    """
    Uses LC0 sharpness on the top candidate's WDL (if available).
    Falls back to CP-gap sharpness when WDL is missing.
    Returns: (sharpness, expected_score_top1, wdl_top1_or_None)
    """
    cands = analyse_multipv(engine, board, MAIA_ANALYSE_TIME, max(MULTIPV, 2))
    if not cands: return 0.0, 0.5, (1.0, 0.0, 0.0)

    have_wdl = any(c[2] is not None for c in cands)
    if have_wdl:
        c_sorted = sorted(cands, key=lambda x: (expected_score_from_wdl(x[2]) if x[2] else 0.5, x[1]), reverse=True)
    else:
        c_sorted = sorted(cands, key=lambda x: x[1], reverse=True)

    mv1, cp1, wdl1, _ = c_sorted[0]
    if have_wdl and wdl1:
        sharp = sharpness_lc0_from_wdl(wdl1)
        e_top1 = expected_score_from_wdl(wdl1)
        if DEBUG:
            dbg(f"[ANALYZE] Ply {board.fullmove_number*2 - (0 if board.turn==chess.WHITE else 1)} "
                f"{'W' if board.turn==chess.WHITE else 'B'} | E1={e_top1:.3f} WDL1={tuple(round(x,3) for x in wdl1)} "
                f"| sharpLC0={sharp:.3f}")
        return sharp, e_top1, wdl1
    else:
        if len(c_sorted) > 1:
            _, cp2, _, _ = c_sorted[1]
        else:
            cp2 = cp1
        sharp_cp = abs(cp1 - cp2) / 100.0
        e = sigmoid_from_cp(cp1)
        if DEBUG:
            dbg(f"[ANALYZE] (CP fallback) cp1={cp1:+d}, cp2={cp2:+d}, sharp~{sharp_cp:.3f}, e~{e:.3f}")
        return min(2.0, sharp_cp), e, None

def stockfish_move(sf, board):
    try:
        r = sf.play(board, chess.engine.Limit(time=SF_MOVE_TIME))
        return r.move if r else None
    except Exception as e:
        dbg(f"[WARN] stockfish_move: sf.play failed ({e}); choosing random legal move")
        ms = list(board.legal_moves)
        return random.choice(ms) if ms else None

def simulate_game(game_index, cheat_mode=True):
    with chess.engine.SimpleEngine.popen_uci(LC0_BIN) as lc0, \
         chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN) as sf:

        configure_lc0(lc0)
        configure_sf(sf)

        board = chess.Board()
        game = chess.pgn.Game(); node = game

        clk = {chess.WHITE: START_SECONDS, chess.BLACK: START_SECONDS}
        wl, bl = [], []
        emt_records = []
        hist_state = {}

        cheater = random.choice([chess.WHITE, chess.BLACK]) if cheat_mode else None
        dbg(f"\n=== GAME #{game_index} === mode={'cheat' if cheat_mode else 'clean'} "
            f"cheater={'-' if cheater is None else ('W' if cheater==chess.WHITE else 'B')}")

        cheating = {chess.WHITE: False, chess.BLACK: False}
        cheat_moves_count = {chess.WHITE: 0, chess.BLACK: 0}
        intent = {chess.WHITE: [], chess.BLACK: []}

        moves_played = 0

        while not board.is_game_over() and moves_played < MAX_MOVES:
            turn = board.turn

            mv_book = try_book_move(board)
            if mv_book:
                emt = human_think_time(board, 0.0, (1.0,0.0,0.0), clk[turn], book=True)
                clk[turn] = max(0.0, clk[turn] - emt) + INCREMENT_SECONDS
                board.push(mv_book); node = node.add_variation(mv_book)
                append_clock_comment(node, emt, clk[turn])
                (wl.append(0) if turn==chess.WHITE else bl.append(0))
                moves_played += 1
                dbg(f"[BOOK] {'W' if turn==chess.WHITE else 'B'} {mv_book.uci()} | EMT~{emt:.2f}s | CLK->{int(clk[turn])}s")
                intent[turn] = []
                update_shuffle_state(board.copy(stack=False), mv_book, hist_state)
                continue

            sharp_l1, e_top1, wdl_top1 = measure_sharpness_and_eval_WDL(lc0, board)
            worse = (e_top1 <= (0.5 - WORSE_BY_WDL))
            sharp = (sharp_l1 >= SHARPNESS_WDL_L1)  # consider retuning for LC0 sharpness
            critical = (entropy_wdl(wdl_top1) >= 1.0 if wdl_top1 else False) and (random.random() < CRITICAL_PROB)
            trigger = worse or sharp or critical

            force_now = False
            if cheater is not None and board.turn == cheater and cheat_moves_count[cheater] == 0 and board.ply() >= FORCE_CHEAT_AFTER_PLY:
                force_now = True

            dbg(f"[STATE] Move {board.fullmove_number}{'...' if turn==chess.BLACK else '.'} "
                f"{'W' if turn==chess.WHITE else 'B'} | trig={trigger} | force={force_now} | cheat_act={cheating[turn]} | clk={int(clk[turn])}s")

            if cheating[turn]:
                s_info = sf.analyse(board, chess.engine.Limit(time=MAIA_ANALYSE_TIME))
                sf_eval_w = pov_cp(s_info, chess.WHITE)
                side_adv = sf_eval_w if turn == chess.WHITE else -sf_eval_w
                if side_adv >= ADVANTAGE_CP:
                    cheating[turn] = False
                    intent[turn] = []
                else:
                    mv = stockfish_move(sf, board)
                    if mv is None: break
                    emt = human_think_time(board, sharp_l1, wdl_top1, clk[turn])
                    clk[turn] = max(0.0, clk[turn] - emt) + INCREMENT_SECONDS
                    board.push(mv); node = node.add_variation(mv)
                    append_clock_comment(node, emt, clk[turn])
                    (wl.append(1) if turn == chess.WHITE else bl.append(1))
                    cheat_moves_count[turn] += 1
                    moves_played += 1
                    emt_records.append({"game_id": game_index, "ply": moves_played, "mover": "W" if turn==chess.WHITE else "B",
                                        "emt_s": round(emt,2), "clk_after_s": round(clk[turn],2)})
                    dbg(f"[PLAY] SF {'W' if turn==chess.WHITE else 'B'} {mv.uci()} | EMT~{emt:.2f}s | CLK->{int(clk[turn])}s")
                    update_shuffle_state(board.copy(stack=False), mv, hist_state)
                    continue

            if cheater is not None and turn == cheater and (trigger or force_now):
                cheating[turn] = True
                mv = stockfish_move(sf, board)
                if mv is None: break
                emt = human_think_time(board, sharp_l1, wdl_top1, clk[turn])
                clk[turn] = max(0.0, clk[turn] - emt) + INCREMENT_SECONDS
                board.push(mv); node = node.add_variation(mv)
                append_clock_comment(node, emt, clk[turn])
                (wl.append(1) if turn == chess.WHITE else bl.append(1))
                cheat_moves_count[turn] += 1
                moves_played += 1
                emt_records.append({"game_id": game_index, "ply": moves_played, "mover": "W" if turn==chess.WHITE else "B",
                                    "emt_s": round(emt,2), "clk_after_s": round(clk[turn],2)})
                dbg(f"[PLAY] SF* {'W' if turn==chess.WHITE else 'B'} {mv.uci()} | EMT~{emt:.2f}s | CLK->{int(clk[turn])}s")
                intent[turn] = []
                update_shuffle_state(board.copy(stack=False), mv, hist_state)
                continue

            intended_move = intent[turn][0] if intent[turn] else None
            mv, future_intent, cands_sorted, chosen_pen = choose_maia_move_with_intent_WDL(
                lc0, board, intended_move, clk[turn], hist_state
            )
            if mv is None: break

            if USE_SF_BLUNDERCHECK and cands_sorted:
                mv_safe = blunder_filter(sf, board, mv, cands_sorted)
                if mv_safe != mv:
                    future_intent = []
                    mv = mv_safe
                    chosen_pen = 0

            if intended_move and mv == intended_move:
                intent[turn].pop(0)
            else:
                intent[turn] = future_intent

            emt = human_think_time(board, sharp_l1, wdl_top1, clk[turn])
            clk[turn] = max(0.0, clk[turn] - emt) + INCREMENT_SECONDS
            board.push(mv); node = node.add_variation(mv)
            append_clock_comment(node, emt, clk[turn])
            (wl.append(0) if turn==chess.WHITE else bl.append(0))
            moves_played += 1
            emt_records.append({"game_id": game_index, "ply": moves_played, "mover": "W" if turn==chess.WHITE else "B",
                                "emt_s": round(emt,2), "clk_after_s": round(clk[turn],2)})
            dbg(f"[PLAY] Maia {'W' if turn==chess.WHITE else 'B'} {mv.uci()} (intent_len={len(intent[turn])}, pen={chosen_pen}) "
                f"| EMT~{emt:.2f}s | CLK->{int(clk[turn])}s")
            update_shuffle_state(board.copy(stack=False), mv, hist_state)

        game.headers["Event"] = ("Maia-vs-Maia (cheat injected)"
                                 if cheat_mode else
                                 "Maia-vs-Maia (clean)")
        game.headers["White"] = "lc0_maia_A"; game.headers["Black"] = "lc0_maia_B"
        game.headers["TimeControl"] = f"{int(START_SECONDS)}+{int(INCREMENT_SECONDS)}"
        game.headers["Result"] = board.result()

        b = chess.Board(); aw, ab = [], []; wi = bi = 0
        for mv in game.mainline_moves():
            if b.turn == chess.WHITE:
                aw.append(wl[wi] if wi < len(wl) else 0); ab.append(0); wi += 1
            else:
                ab.append(bl[bi] if bi < len(bl) else 0); aw.append(0); bi += 1
            b.push(mv)

        cheater_side_str = "none" if cheater is None else ("white" if cheater == chess.WHITE else "black")
        return game, aw, ab, cheater_side_str, board.result(), emt_records, sum(aw), sum(ab)

if __name__ == "__main__":
    if PGN_PATH.exists(): PGN_PATH.unlink()
    game_rows, timing_rows = [], []

    phases = [
        ("cheat", True, NUM_GAMES_CHEAT),
        ("clean", False, NUM_GAMES_CLEAN),
    ]

    pbar = tqdm(total=sum(n for _,_,n in phases), desc="Sim games")
    with open(PGN_PATH, "a", encoding="utf8") as pgn_file:
        game_id = 0
        for phase_name, cheat_mode, count in phases:
            for _ in range(count):
                try:
                    game, white_labels, black_labels, cheater_side, result, emts, cheat_w, cheat_b = simulate_game(game_id, cheat_mode=cheat_mode)
                    game.headers["Site"] = f"Sim #{game_id} ({phase_name})"
                    game.accept(chess.pgn.FileExporter(pgn_file)); pgn_file.write("\n")

                    game_rows.append({
                        "game_id": game_id,
                        "phase": phase_name,                  # tag phase
                        "cheater_side": cheater_side,
                        "result": result,
                        "moves_count": len(list(game.mainline_moves())),
                        "white_labels": json.dumps(white_labels),
                        "black_labels": json.dumps(black_labels),
                        "cheat_moves_white": cheat_w,
                        "cheat_moves_black": cheat_b
                    })
                    for r in emts:
                        r["phase"] = phase_name              # tag timing rows too
                    timing_rows.extend(emts)
                    pbar.set_postfix_str(f"{phase_name}: last={result}, cheater={cheater_side}, cw={cheat_w}, cb={cheat_b}")
                except Exception as e:
                    print(f"[ERROR] Game {game_id} failed: {e}", flush=True)
                    game_rows.append({
                        "game_id": game_id, "phase": phase_name, "cheater_side": "error", "result": str(e),
                        "moves_count": None, "white_labels": "[]", "black_labels": "[]",
                        "cheat_moves_white": 0, "cheat_moves_black": 0
                    })
                game_id += 1
                pbar.update(1)
    pbar.close()

    try:
        with pd.ExcelWriter(LABELS_XLSX, engine="openpyxl") as writer:
            pd.DataFrame(game_rows).to_excel(writer, index=False, sheet_name="labels")
            pd.DataFrame(timing_rows).to_excel(writer, index=False, sheet_name="timing")
        print(f"PGN: {PGN_PATH}\nExcel: {LABELS_XLSX}")
    except ModuleNotFoundError:
        labels_csv = OUT_DIR / "labels.csv"
        timing_csv = OUT_DIR / "timing.csv"
        pd.DataFrame(game_rows).to_csv(labels_csv, index=False)
        pd.DataFrame(timing_rows).to_csv(timing_csv, index=False)
        print(f"PGN: {PGN_PATH}\n(openpyxl missing) Wrote CSVs instead:\n- {labels_csv}\n- {timing_csv}")
