"""
run_all_experiments.py
Selieri Chess Anti-Cheating — Full Experiment Pipeline

Experiment A1: Per-Move BiLSTM  — SF + Maia (18 features)
Experiment A2: Per-Move BiLSTM  — SF only   (9 features)   [control]
Experiment B1: Per-Game Binary BiLSTM — SF + Maia (18 features)
Experiment B2: Per-Game Binary BiLSTM — SF only   (9 features)  [control]
Explainability: Koh & Liang (2017) influence functions via LiSSA

Outputs:
  selieri_model_permove_full.pt / _sfonly.pt
  selieri_model_game_full.pt   / _sfonly.pt
  fig_v2_*.png  (10 figures)
  Selieri_v2.docx
"""

import ast, json, warnings, time, random, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})

DEVICE = torch.device('cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE, 'data')          # input dataset
RESULTS_DIR = os.path.join(BASE, 'results')       # all generated outputs
MODELS_DIR  = os.path.join(RESULTS_DIR, 'models')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print('=' * 62)
print('  SELIERI — Full Experiment Pipeline (4 models)')
print('=' * 62)

# ─────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────
print('\n[1] Loading data ...')
df = pd.read_excel(f'{DATA_DIR}/combined_features_labels.xlsx')
print(f'    Rows: {df.shape[0]}  Cols: {df.shape[1]}')

def parse_array(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return []
    if isinstance(s, (list, np.ndarray)): return list(s)
    s = str(s).strip()
    if not s or s in ('nan', 'None', '[]'): return []
    try:    return ast.literal_eval(s)
    except:
        try:    return json.loads(s)
        except: return []

ARRAY_COLS = [c for c in [
    'Side', 'MoveNo',
    'SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
    'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness',
    'LC0_D20_Rank','LC0_D20_CPL','LC0_D20_AdvWP','LC0_D20_BestWP','LC0_D20_WCL',
    'LC0_D20_Ambiguity05','LC0_D20_difNextBest','LC0_D20_difNextWorst','LC0_D20_Sharpness',
    'white_labels', 'black_labels',
] if c in df.columns]
for col in ARRAY_COLS:
    df[col + '_arr'] = df[col].apply(parse_array)

# Full feature set: SF D20 (9) + LC0 D20 (9)
FEAT_COLS = [
    'SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
    'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness',
    'LC0_D20_Rank','LC0_D20_CPL','LC0_D20_AdvWP','LC0_D20_BestWP','LC0_D20_WCL',
    'LC0_D20_Ambiguity05','LC0_D20_difNextBest','LC0_D20_difNextWorst','LC0_D20_Sharpness',
]
# SF-only control: SF D20 (9)
SF_COLS = [
    'SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
    'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness',
]
N_FEAT_FULL = len(FEAT_COLS)   # 18
N_FEAT_SF   = len(SF_COLS)     # 9

print('[2] Building move-level dataframe ...')
rows = []
for _, game in df.iterrows():
    sides    = game['Side_arr']
    w_labels = game['white_labels_arr']
    b_labels = game['black_labels_arr']
    feat_arrs = {fc: game[fc + '_arr'] for fc in FEAT_COLS if fc + '_arr' in game.index}
    wi = bi = 0
    for i, side in enumerate(sides):
        if side == 'W':
            lbl = int(w_labels[wi]) if wi < len(w_labels) else 0; wi += 1
        else:
            lbl = int(b_labels[bi]) if bi < len(b_labels) else 0; bi += 1
        row = {'game_id': game['game_id'], 'move_idx': i, 'move_label': lbl}
        for fc, arr in feat_arrs.items():
            row[fc] = arr[i] if i < len(arr) else np.nan
        rows.append(row)

moves_df = pd.DataFrame(rows)
for fc in FEAT_COLS:
    if fc in moves_df.columns:
        moves_df[fc] = pd.to_numeric(moves_df[fc], errors='coerce')
moves_df.dropna(subset=FEAT_COLS, how='all', inplace=True)
moves_df.reset_index(drop=True, inplace=True)

# Build per-game sequences
sequences_full, sequences_sf, label_seqs, game_ids_seq, game_labels_arr = [], [], [], [], []
for gid, grp in sorted(moves_df.groupby('game_id'), key=lambda x: x[0]):
    seq_full = grp[FEAT_COLS].values.astype(np.float32)
    seq_sf   = grp[SF_COLS].values.astype(np.float32)
    lbl      = grp['move_label'].values.astype(np.float32)
    if seq_full.shape[0] < 5: continue
    sequences_full.append(seq_full)
    sequences_sf.append(seq_sf)
    label_seqs.append(lbl)
    game_ids_seq.append(gid)
    game_labels_arr.append(1 if lbl.sum() > 0 else 0)

game_labels_arr = np.array(game_labels_arr)
n_games     = len(sequences_full)
total_moves = sum(s.shape[0] for s in sequences_full)
total_cheat = int(sum(l.sum() for l in label_seqs))

print(f'    Games: {n_games}  |  Moves: {total_moves:,}')
print(f'    Cheat moves: {total_cheat:,} ({total_cheat/total_moves:.1%})')
print(f'    Cheat games: {game_labels_arr.sum()} ({game_labels_arr.mean():.1%})')

# Normalise — separate scalers for full and SF-only
all_full = np.concatenate(sequences_full, axis=0)
all_sf   = np.concatenate(sequences_sf,   axis=0)

scaler_full = StandardScaler()
scaler_sf   = StandardScaler()
scaler_full.fit(np.nan_to_num(all_full, nan=0.0))
scaler_sf.fit(np.nan_to_num(all_sf,   nan=0.0))

seqs_norm_full = [scaler_full.transform(np.nan_to_num(s, nan=0.0)).astype(np.float32)
                  for s in sequences_full]
seqs_norm_sf   = [scaler_sf.transform(np.nan_to_num(s, nan=0.0)).astype(np.float32)
                  for s in sequences_sf]

# Stratified split (shared indices across all four models)
idx = np.arange(n_games)
tr_idx, tmp_idx = train_test_split(
    idx, test_size=0.30, stratify=game_labels_arr, random_state=SEED)
va_idx, te_idx = train_test_split(
    tmp_idx, test_size=0.50, stratify=game_labels_arr[tmp_idx], random_state=SEED)

tr_flat          = np.concatenate([label_seqs[i] for i in tr_idx])
pos_weight_move  = (tr_flat == 0).sum() / max((tr_flat == 1).sum(), 1)
pos_weight_game  = (game_labels_arr[tr_idx] == 0).sum() / \
                    max((game_labels_arr[tr_idx] == 1).sum(), 1)

print(f'\n    Train:{len(tr_idx)}  Val:{len(va_idx)}  Test:{len(te_idx)}')
print(f'    pos_weight (move)={pos_weight_move:.2f}  (game)={pos_weight_game:.2f}')

# ─────────────────────────────────────────────────────────────────
# 2. DATASETS & DATALOADERS
# ─────────────────────────────────────────────────────────────────
def collate_permove(batch):
    seqs, lbls = zip(*batch)
    lengths  = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len, n_feat = lengths.max().item(), seqs[0].shape[1]
    padded_x = torch.zeros(len(seqs), max_len, n_feat)
    padded_y = torch.zeros(len(seqs), max_len)
    for i, (s, l, ln) in enumerate(zip(seqs, lbls, lengths)):
        padded_x[i, :ln] = torch.tensor(s)
        padded_y[i, :ln] = torch.tensor(l)
    return padded_x, lengths, padded_y

def collate_game(batch):
    seqs, game_lbls = zip(*batch)
    lengths  = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len, n_feat = lengths.max().item(), seqs[0].shape[1]
    padded_x = torch.zeros(len(seqs), max_len, n_feat)
    for i, (s, ln) in enumerate(zip(seqs, lengths)):
        padded_x[i, :ln] = torch.tensor(s)
    return padded_x, lengths, torch.tensor(game_lbls, dtype=torch.float32)

class PerMoveDS(Dataset):
    def __init__(self, indices, seqs_norm):
        self.indices = indices; self.seqs = seqs_norm
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]; return self.seqs[j], label_seqs[j]

class GameDS(Dataset):
    def __init__(self, indices, seqs_norm):
        self.indices = indices; self.seqs = seqs_norm
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]; return self.seqs[j], float(game_labels_arr[j])

BATCH = 32
def make_loaders_permove(seqs_norm):
    return (DataLoader(PerMoveDS(tr_idx, seqs_norm), BATCH, shuffle=True,  collate_fn=collate_permove),
            DataLoader(PerMoveDS(va_idx, seqs_norm), BATCH, shuffle=False, collate_fn=collate_permove),
            DataLoader(PerMoveDS(te_idx, seqs_norm), BATCH, shuffle=False, collate_fn=collate_permove))

def make_loaders_game(seqs_norm):
    return (DataLoader(GameDS(tr_idx, seqs_norm), BATCH, shuffle=True,  collate_fn=collate_game),
            DataLoader(GameDS(va_idx, seqs_norm), BATCH, shuffle=False, collate_fn=collate_game),
            DataLoader(GameDS(te_idx, seqs_norm), BATCH, shuffle=False, collate_fn=collate_game))

pm_tr_f, pm_va_f, pm_te_f = make_loaders_permove(seqs_norm_full)
pm_tr_s, pm_va_s, pm_te_s = make_loaders_permove(seqs_norm_sf)
gm_tr_f, gm_va_f, gm_te_f = make_loaders_game(seqs_norm_full)
gm_tr_s, gm_va_s, gm_te_s = make_loaders_game(seqs_norm_sf)

# ─────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────
class CheatDetectorPerMove(nn.Module):
    def __init__(self, n_features=18, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.classifier(out).squeeze(-1)

class CheatDetectorGame(nn.Module):
    def __init__(self, n_features=18, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        mask   = (torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None])
        pooled = (out * mask.unsqueeze(-1).float()).sum(1) / mask.unsqueeze(-1).float().sum(1)
        return self.classifier(pooled).squeeze(-1)

# ─────────────────────────────────────────────────────────────────
# 4. TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────
def train_permove(model, tr_loader, va_loader, pos_wt, epochs=50, patience=7, lr=1e-3):
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt]), reduction='none')
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    hist  = {'tr_loss':[], 'va_loss':[], 'tr_f1':[], 'va_f1':[]}
    best_val, best_state, pat = float('inf'), None, 0

    def run_epoch(loader, train):
        model.train(train)
        tot_loss, tot_b, all_p, all_t = 0.0, 0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, lengths, y in loader:
                logits = model(x, lengths)
                mask   = torch.arange(logits.size(1))[None, :] < lengths[:, None]
                loss   = crit(logits, y)
                loss   = (loss * mask.float()).sum() / mask.float().sum()
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                probs = torch.sigmoid(logits).detach()
                all_p.extend((probs[mask] >= 0.5).long().tolist())
                all_t.extend(y[mask].long().tolist())
                tot_loss += loss.item(); tot_b += 1
        return tot_loss / tot_b, f1_score(all_t, all_p, zero_division=0)

    print(f'  {"Ep":>4} {"TrLoss":>8} {"VaLoss":>8} {"TrF1":>7} {"VaF1":>7}')
    for ep in range(1, epochs + 1):
        tl, tf = run_epoch(tr_loader, True)
        vl, vf = run_epoch(va_loader, False)
        sched.step(vl)
        hist['tr_loss'].append(tl); hist['va_loss'].append(vl)
        hist['tr_f1'].append(tf);   hist['va_f1'].append(vf)
        print(f'  {ep:4d} {tl:8.4f} {vl:8.4f} {tf:7.3f} {vf:7.3f}')
        if vl < best_val:
            best_val = vl; pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                print(f'  Early stop ep {ep}'); break
    model.load_state_dict(best_state)
    return hist

def train_game(model, tr_loader, va_loader, pos_wt, epochs=50, patience=7, lr=1e-3):
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt]))
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    hist  = {'tr_loss':[], 'va_loss':[], 'tr_f1':[], 'va_f1':[]}
    best_val, best_state, pat = float('inf'), None, 0

    def run_epoch(loader, train):
        model.train(train)
        tot_loss, tot_b, all_p, all_t = 0.0, 0, [], []
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, lengths, y in loader:
                logits = model(x, lengths)
                loss   = crit(logits, y)
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                probs = torch.sigmoid(logits).detach()
                all_p.extend((probs >= 0.5).long().tolist())
                all_t.extend(y.long().tolist())
                tot_loss += loss.item(); tot_b += 1
        return tot_loss / tot_b, f1_score(all_t, all_p, zero_division=0)

    print(f'  {"Ep":>4} {"TrLoss":>8} {"VaLoss":>8} {"TrF1":>7} {"VaF1":>7}')
    for ep in range(1, epochs + 1):
        tl, tf = run_epoch(tr_loader, True)
        vl, vf = run_epoch(va_loader, False)
        sched.step(vl)
        hist['tr_loss'].append(tl); hist['va_loss'].append(vl)
        hist['tr_f1'].append(tf);   hist['va_f1'].append(vf)
        print(f'  {ep:4d} {tl:8.4f} {vl:8.4f} {tf:7.3f} {vf:7.3f}')
        if vl < best_val:
            best_val = vl; pat = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
            if pat >= patience:
                print(f'  Early stop ep {ep}'); break
    model.load_state_dict(best_state)
    return hist

def eval_permove(model, loader):
    model.eval()
    all_probs, all_preds, all_true, gprobs, gtrue = [], [], [], [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x, lengths)
            probs  = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(lengths)):
                L = lengths[i].item()
                p = probs[i, :L]; t = y[i, :L].numpy().astype(int)
                all_probs.extend(p); all_preds.extend((p >= 0.5).astype(int)); all_true.extend(t)
                gprobs.append(float(p.max())); gtrue.append(int(t.max()))
    ap = np.array(all_probs); ad = np.array(all_preds); at = np.array(all_true)
    gp = np.array(gprobs);    gt = np.array(gtrue)
    return dict(
        probs=ap, preds=ad, true=at,
        game_probs=gp, game_preds=(gp >= 0.5).astype(int), game_true=gt,
        move_acc=accuracy_score(at, ad),
        move_prec=precision_score(at, ad, zero_division=0),
        move_rec=recall_score(at, ad, zero_division=0),
        move_f1=f1_score(at, ad, zero_division=0),
        move_auc=roc_auc_score(at, ap),
        game_acc=accuracy_score(gt, (gp >= 0.5).astype(int)),
        game_prec=precision_score(gt, (gp >= 0.5).astype(int), zero_division=0),
        game_rec=recall_score(gt, (gp >= 0.5).astype(int), zero_division=0),
        game_f1=f1_score(gt, (gp >= 0.5).astype(int), zero_division=0),
        game_auc=roc_auc_score(gt, gp),
    )

def eval_game(model, loader):
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            logits = model(x, lengths)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend((probs >= 0.5).astype(int).tolist())
            all_true.extend(y.long().tolist())
    ap = np.array(all_probs); ad = np.array(all_preds); at = np.array(all_true)
    return dict(
        probs=ap, preds=ad, true=at,
        acc=accuracy_score(at, ad),
        prec=precision_score(at, ad, zero_division=0),
        rec=recall_score(at, ad, zero_division=0),
        f1=f1_score(at, ad, zero_division=0),
        auc=roc_auc_score(at, ap),
    )

def save_model(model, path, n_feat, scaler, feat_cols, results):
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features': n_feat, 'hidden': 128, 'n_layers': 2,
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'feature_cols': feat_cols,
        'results': {k: float(v) for k, v in results.items()
                    if k not in ('probs','preds','true','game_probs','game_preds','game_true')},
    }, path)
    print(f'  Saved: {os.path.basename(path)}')

# ─────────────────────────────────────────────────────────────────
# 5. TRAIN ALL FOUR MODELS
# ─────────────────────────────────────────────────────────────────

# A1 — Per-Move, SF + Maia
print('\n' + '=' * 62)
print('  EXP A1: Per-Move BiLSTM — SF + Maia (18 features)')
print('=' * 62)
A1 = CheatDetectorPerMove(n_features=N_FEAT_FULL).to(DEVICE)
print(f'  Params: {sum(p.numel() for p in A1.parameters()):,}')
hist_A1 = train_permove(A1, pm_tr_f, pm_va_f, pos_weight_move)
res_A1  = eval_permove(A1, pm_te_f)
save_model(A1, f'{MODELS_DIR}/selieri_model_permove_full.pt',
           N_FEAT_FULL, scaler_full, FEAT_COLS, res_A1)
print(f'  Move AUC={res_A1["move_auc"]:.4f}  Game AUC={res_A1["game_auc"]:.4f}')

# A2 — Per-Move, SF only
print('\n' + '=' * 62)
print('  EXP A2: Per-Move BiLSTM — SF only (9 features) [control]')
print('=' * 62)
A2 = CheatDetectorPerMove(n_features=N_FEAT_SF).to(DEVICE)
print(f'  Params: {sum(p.numel() for p in A2.parameters()):,}')
hist_A2 = train_permove(A2, pm_tr_s, pm_va_s, pos_weight_move)
res_A2  = eval_permove(A2, pm_te_s)
save_model(A2, f'{MODELS_DIR}/selieri_model_permove_sfonly.pt',
           N_FEAT_SF, scaler_sf, SF_COLS, res_A2)
print(f'  Move AUC={res_A2["move_auc"]:.4f}  Game AUC={res_A2["game_auc"]:.4f}')

# B1 — Per-Game, SF + Maia
print('\n' + '=' * 62)
print('  EXP B1: Per-Game Binary BiLSTM — SF + Maia (18 features)')
print('=' * 62)
B1 = CheatDetectorGame(n_features=N_FEAT_FULL).to(DEVICE)
print(f'  Params: {sum(p.numel() for p in B1.parameters()):,}')
hist_B1 = train_game(B1, gm_tr_f, gm_va_f, pos_weight_game)
res_B1  = eval_game(B1, gm_te_f)
save_model(B1, f'{MODELS_DIR}/selieri_model_game_full.pt',
           N_FEAT_FULL, scaler_full, FEAT_COLS, res_B1)
print(f'  AUC={res_B1["auc"]:.4f}  F1={res_B1["f1"]:.4f}')

# B2 — Per-Game, SF only
print('\n' + '=' * 62)
print('  EXP B2: Per-Game Binary BiLSTM — SF only (9 features) [control]')
print('=' * 62)
B2 = CheatDetectorGame(n_features=N_FEAT_SF).to(DEVICE)
print(f'  Params: {sum(p.numel() for p in B2.parameters()):,}')
hist_B2 = train_game(B2, gm_tr_s, gm_va_s, pos_weight_game)
res_B2  = eval_game(B2, gm_te_s)
save_model(B2, f'{MODELS_DIR}/selieri_model_game_sfonly.pt',
           N_FEAT_SF, scaler_sf, SF_COLS, res_B2)
print(f'  AUC={res_B2["auc"]:.4f}  F1={res_B2["f1"]:.4f}')


# =================================================================== #
#                                                                     #
#   EXPLAINABILITY — KOH & LIANG (2017) INFLUENCE FUNCTIONS          #
#                                                                     #
#   Applied to model A1 (per-move BiLSTM, SF + Maia features).       #
#                                                                     #
#   Core idea:                                                        #
#     I(z_train, z_test) = -grad_L(z_test)^T * H^{-1} * grad_L(z_train)
#     Negative score  -> proponent  (training game that *supports*    #
#                         the model's suspicion of the test game)     #
#     Positive score  -> opponent   (training game that *hurts* it)   #
#                                                                     #
#   H^{-1}v is approximated by LiSSA (Agarwal et al. 2017):          #
#     h_k = v + (1-damping)*h_{k-1} - HVP(h_{k-1}) / scale          #
#   Scope: classifier head only (16,513 params) — tractable on CPU.  #
#                                                                     #
#   Three outputs:                                                    #
#     Step 1 — gradient vector for every training game  (1,400)      #
#     Step 2 — influence matrix for 50 most suspicious test games     #
#     Step 3 — self-influence I(z,z) for every training game          #
#               (cheat games show ~5x higher self-influence)          #
#                                                                     #
# =================================================================== #

print('\n' + '=' * 62)
print('  EXPLAINABILITY: Koh & Liang (2017) — Applied to A1')
print('=' * 62)

inf_model  = A1
inf_params = list(inf_model.classifier.parameters())
n_params   = sum(p.numel() for p in inf_params)
print(f'  Classifier head params: {n_params:,}')

crit_inf = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_move]), reduction='none')

# --- per-example gradient helper ---
def compute_example_grad(model, params, x, lengths, y):
    model.zero_grad()
    logits = model(x, lengths)
    mask   = torch.arange(logits.size(1))[None, :] < lengths[:, None]
    loss   = crit_inf(logits, y)
    loss   = (loss * mask.float()).sum() / mask.float().sum()
    grads  = torch.autograd.grad(loss, params, retain_graph=False)
    return torch.cat([g.detach().flatten() for g in grads]).numpy()

# --- LiSSA: stochastic inverse-Hessian-vector product ---
def lissa_ihvp(model, params, loader, test_grad,
               scale=25.0, damping=0.01, recursion_depth=100):
    """LiSSA approximation of H^{-1} * test_grad (Koh & Liang 2017, Agarwal et al. 2017)."""
    estimate = torch.tensor(test_grad, dtype=torch.float32)
    v        = torch.tensor(test_grad, dtype=torch.float32)
    batches  = list(loader)
    for _ in range(recursion_depth):
        batch = random.choice(batches)
        x, lengths, y = batch
        model.zero_grad()
        logits = model(x, lengths)
        mask   = torch.arange(logits.size(1))[None, :] < lengths[:, None]
        loss   = crit_inf(logits, y)
        loss   = (loss * mask.float()).sum() / mask.float().sum()
        grads  = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        flat_g = torch.cat([g.flatten() for g in grads])
        gv     = (flat_g * estimate.detach()).sum()
        hvp    = torch.autograd.grad(gv, params, retain_graph=False)
        flat_h = torch.cat([h.detach().flatten() for h in hvp])
        estimate = v + (1.0 - damping) * estimate - flat_h / scale
    return (estimate / scale).numpy()

inf_tr_loader = DataLoader(PerMoveDS(tr_idx, seqs_norm_full), batch_size=16,
                           shuffle=True, collate_fn=collate_permove)

# ---------- Step 1: gradient for every training game ----------
print('\n  Step 1/3: Training gradients (1400 games) ...')
t0 = time.time()
train_grads = []
inf_model.eval()
for i, idx_i in enumerate(tr_idx):
    seq = seqs_norm_full[idx_i]; lbl = label_seqs[idx_i]
    x   = torch.tensor(seq).unsqueeze(0)
    ln  = torch.tensor([seq.shape[0]], dtype=torch.long)
    y   = torch.zeros(1, seq.shape[0]); y[0, :len(lbl)] = torch.tensor(lbl)
    train_grads.append(compute_example_grad(inf_model, inf_params, x, ln, y))
    if (i + 1) % 200 == 0:
        print(f'    {i+1}/{len(tr_idx)}  [{time.time()-t0:.0f}s]')
train_grads = np.array(train_grads)
print(f'  Done in {time.time()-t0:.0f}s')

# ---------- Step 2: LiSSA for top-50 suspicious test games ----------
# Select the 50 test games the model is most suspicious about
with torch.no_grad():
    test_scores = []
    for idx_j in te_idx:
        seq = seqs_norm_full[idx_j]
        x   = torch.tensor(seq).unsqueeze(0)
        ln  = torch.tensor([seq.shape[0]], dtype=torch.long)
        logits = inf_model(x, ln)
        test_scores.append(float(torch.sigmoid(logits).squeeze().numpy().max()))
test_scores = np.array(test_scores)
top50_local  = np.argsort(test_scores)[::-1][:50]
top50_global = te_idx[top50_local]

print(f'\n  Step 2/3: LiSSA H^{{-1}} for 50 suspicious test games ...')
t1 = time.time()
influence_mat = np.zeros((50, len(tr_idx)))
for j, (loc_j, glob_j) in enumerate(zip(top50_local, top50_global)):
    seq = seqs_norm_full[glob_j]; lbl = label_seqs[glob_j]
    x   = torch.tensor(seq).unsqueeze(0)
    ln  = torch.tensor([seq.shape[0]], dtype=torch.long)
    y   = torch.zeros(1, seq.shape[0]); y[0, :len(lbl)] = torch.tensor(lbl)
    tg  = compute_example_grad(inf_model, inf_params, x, ln, y)
    s_test = lissa_ihvp(inf_model, inf_params, inf_tr_loader, tg)
    influence_mat[j] = -(train_grads @ s_test)   # K&L eq.5: I = -s_test . grad_train
    if (j + 1) % 10 == 0:
        print(f'    {j+1}/50  [{time.time()-t1:.0f}s]')
print(f'  Influence matrix done in {time.time()-t1:.0f}s')
print(f'  Range: [{influence_mat.min():.4f}, {influence_mat.max():.4f}]')

# ---------- Step 3: self-influence I(z,z) for all training games ----------
print('\n  Step 3/3: Self-influence (1400 train games) ...')
self_inf = np.zeros(len(tr_idx))
for i, idx_i in enumerate(tr_idx):
    s = lissa_ihvp(inf_model, inf_params, inf_tr_loader, train_grads[i],
                   recursion_depth=50)
    self_inf[i] = -float(train_grads[i] @ s)
tr_labels         = game_labels_arr[tr_idx]
mean_cheat_si     = self_inf[tr_labels == 1].mean()
mean_clean_si     = self_inf[tr_labels == 0].mean()
print(f'  Self-inf  cheat={mean_cheat_si:.6f}  clean={mean_clean_si:.6f}  '
      f'ratio={mean_cheat_si/mean_clean_si:.1f}x')

# =================================================================== #
#                  END OF EXPLAINABILITY SECTION                      #
# =================================================================== #

# ─────────────────────────────────────────────────────────────────
# 7. FIGURES
# ─────────────────────────────────────────────────────────────────
print('\n' + '=' * 62)
print('  GENERATING FIGURES')
print('=' * 62)

COLORS = {
    'A1': '#3498db',   # blue  — per-move full
    'A2': '#85c1e9',   # light blue — per-move SF only
    'B1': '#e74c3c',   # red   — per-game full
    'B2': '#f1948a',   # light red  — per-game SF only
}
LABELS = {
    'A1': 'A1: Per-Move SF+Maia',
    'A2': 'A2: Per-Move SF-only',
    'B1': 'B1: Per-Game SF+Maia',
    'B2': 'B2: Per-Game SF-only',
}

def savefig(name):
    path = f'{FIG_DIR}/{name}'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved: {name}')
    return path

figures = {}

# Fig 1: Dataset overview
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
batches = df['batch'].value_counts().sort_index()
axes[0].bar([f'Batch {b}' for b in batches.index], batches.values, color=['#3498db','#e74c3c'])
axes[0].set_title('Games per Batch', fontweight='bold'); axes[0].set_ylabel('Count')
for bar, v in zip(axes[0].patches, batches.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(v),
                 ha='center', fontsize=10, fontweight='bold')
ml = [s.shape[0] for s in sequences_full]
axes[1].hist(ml, bins=30, color='#2ecc71', edgecolor='white', alpha=0.85)
axes[1].set_title('Moves per Game', fontweight='bold'); axes[1].set_xlabel('Move Count')
axes[1].axvline(np.mean(ml), color='red', ls='--', label=f'Mean={np.mean(ml):.0f}')
axes[1].legend()
axes[2].pie([total_moves - total_cheat, total_cheat],
            labels=['Clean', 'Cheat'], colors=['#2ecc71','#e74c3c'],
            autopct='%1.1f%%', startangle=90)
axes[2].set_title('Move Class Balance', fontweight='bold')
plt.suptitle('Dataset Overview — 2,000 Games, 193,074 Moves', fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_dataset.png'] = savefig('fig_v2_dataset.png')

# Fig 2: Training curves — all 4 models
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for (hist, key, ax_loss, ax_f1) in [
        (hist_A1, 'A1', axes[0,0], axes[0,1]),
        (hist_A2, 'A2', axes[0,0], axes[0,1]),
        (hist_B1, 'B1', axes[1,0], axes[1,1]),
        (hist_B2, 'B2', axes[1,0], axes[1,1])]:
    ep = range(1, len(hist['tr_loss'])+1)
    ls = '-' if key in ('A1','B1') else '--'
    ax_loss.plot(ep, hist['va_loss'], color=COLORS[key], lw=2, ls=ls, label=LABELS[key])
    ax_f1.plot(ep,  hist['va_f1'],   color=COLORS[key], lw=2, ls=ls, label=LABELS[key])

for ax, title in [(axes[0,0],'Per-Move — Val Loss'), (axes[0,1],'Per-Move — Val F1'),
                  (axes[1,0],'Per-Game — Val Loss'), (axes[1,1],'Per-Game — Val F1')]:
    ax.set_title(title, fontweight='bold'); ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)

plt.suptitle('Validation Curves — All 4 Models (solid=SF+Maia, dashed=SF-only)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_training.png'] = savefig('fig_v2_training.png')

# Fig 3: ROC curves — all 4 models
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Per-move panel: use game-level derived AUC
for key, res in [('A1', res_A1), ('A2', res_A2)]:
    fpr, tpr, _ = roc_curve(res['game_true'], res['game_probs'])
    axes[0].plot(fpr, tpr, color=COLORS[key], lw=2,
                 label=f'{LABELS[key]}  AUC={res["game_auc"]:.4f}')
axes[0].plot([0,1],[0,1],'k--',alpha=0.3)
axes[0].set_title('ROC — Per-Move Models (game-level via max-pool)', fontweight='bold')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].legend(fontsize=9)

for key, res in [('B1', res_B1), ('B2', res_B2)]:
    fpr, tpr, _ = roc_curve(res['true'], res['probs'])
    axes[1].plot(fpr, tpr, color=COLORS[key], lw=2,
                 label=f'{LABELS[key]}  AUC={res["auc"]:.4f}')
axes[1].plot([0,1],[0,1],'k--',alpha=0.3)
axes[1].set_title('ROC — Per-Game Models', fontweight='bold')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].legend(fontsize=9)

plt.suptitle('ROC Curves — SF+Maia vs SF-only', fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_roc.png'] = savefig('fig_v2_roc.png')

# Fig 4: Confusion matrices — 4 panels
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
specs = [
    (res_A1['game_true'], res_A1['game_preds'], 'A1: Per-Move SF+Maia\n(game, max-pool)', 'Blues'),
    (res_A2['game_true'], res_A2['game_preds'], 'A2: Per-Move SF-only\n(game, max-pool)', 'Blues'),
    (res_B1['true'],      res_B1['preds'],      'B1: Per-Game SF+Maia', 'Reds'),
    (res_B2['true'],      res_B2['preds'],      'B2: Per-Game SF-only', 'Reds'),
]
for ax, (yt, yp, title, cmap) in zip(axes, specs):
    cm = confusion_matrix(yt, yp)
    f1 = f1_score(yt, yp, zero_division=0)
    auc = roc_auc_score(yt, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Clean','Cheat'], yticklabels=['Clean','Cheat'])
    ax.set_title(f'{title}\nF1={f1:.3f}', fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices — All 4 Models', fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_confusion.png'] = savefig('fig_v2_confusion.png')

# Fig 5: Grouped metric comparison — the main comparison figure
fig, ax = plt.subplots(figsize=(13, 6))
# Use game-level metrics for a fair comparison
metric_keys  = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
vals = {
    'A1': [res_A1['game_acc'], res_A1['game_prec'], res_A1['game_rec'],
           res_A1['game_f1'],  res_A1['game_auc']],
    'A2': [res_A2['game_acc'], res_A2['game_prec'], res_A2['game_rec'],
           res_A2['game_f1'],  res_A2['game_auc']],
    'B1': [res_B1['acc'], res_B1['prec'], res_B1['rec'], res_B1['f1'], res_B1['auc']],
    'B2': [res_B2['acc'], res_B2['prec'], res_B2['rec'], res_B2['f1'], res_B2['auc']],
}
x = np.arange(5); w = 0.2
for i, key in enumerate(['A1','A2','B1','B2']):
    bars = ax.bar(x + (i-1.5)*w, vals[key], w,
                  label=LABELS[key], color=COLORS[key], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7, rotation=45)
ax.set_xticks(x); ax.set_xticklabels(metric_keys)
ax.set_ylim(0, 1.18); ax.set_ylabel('Score')
ax.set_title('Game-Level Metric Comparison — SF+Maia vs SF-only, Per-Move vs Per-Game',
             fontweight='bold')
ax.legend(fontsize=9)
# Delta annotations
for i, mk in enumerate(metric_keys):
    delta_pm = vals['A1'][i] - vals['A2'][i]
    delta_pg = vals['B1'][i] - vals['B2'][i]
    ax.text(i - 0.3, 1.08, f'ΔA={delta_pm:+.3f}', fontsize=7, ha='center', color='#2980b9')
    ax.text(i + 0.1, 1.08, f'ΔB={delta_pg:+.3f}', fontsize=7, ha='center', color='#c0392b')
plt.tight_layout()
figures['fig_v2_metrics.png'] = savefig('fig_v2_metrics.png')

# Fig 6: Move-level score distributions (A1 vs A2)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (res, key) in zip(axes, [(res_A1,'A1'),(res_A2,'A2')]):
    ax.hist(res['probs'][res['true']==0], bins=60, alpha=0.6, color='#2ecc71',
            density=True, label='Clean Move')
    ax.hist(res['probs'][res['true']==1], bins=60, alpha=0.6, color='#e74c3c',
            density=True, label='Cheat Move')
    ax.set_title(f'{LABELS[key]} — Move Score Distribution', fontweight='bold')
    ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Density'); ax.legend()
plt.suptitle('Move-Level Score Distributions — SF+Maia vs SF-only', fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_scores.png'] = savefig('fig_v2_scores.png')

# Fig 7: K&L Proponents & Opponents
mean_inf = influence_mat.mean(axis=0)
top10  = np.argsort(mean_inf)[:10]
bot10  = np.argsort(mean_inf)[-10:]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, idx_list, title, color in [
        (axes[0], top10, 'Top-10 Proponents\n(most negative = most helpful)', '#2ecc71'),
        (axes[1], bot10, 'Top-10 Opponents\n(most positive = most harmful)',  '#e74c3c')]:
    scores = mean_inf[idx_list]
    lbls   = [f'G{game_ids_seq[tr_idx[i]]}\n({"cheat" if tr_labels[i]==1 else "clean"})'
              for i in idx_list]
    ax.barh(range(10), scores[::-1], color=color, alpha=0.85)
    ax.set_yticks(range(10)); ax.set_yticklabels(lbls[::-1], fontsize=8)
    ax.set_title(title, fontweight='bold'); ax.set_xlabel('K&L Influence Score')
    ax.axvline(0, color='black', lw=0.8, ls='--')
plt.suptitle('Koh & Liang (2017) Influence Functions — Proponents & Opponents',
             fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_proponents.png'] = savefig('fig_v2_proponents.png')

# Fig 8: Self-influence
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.hist(self_inf[tr_labels==0], bins=40, alpha=0.7, color='#2ecc71', density=True, label='Clean')
ax.hist(self_inf[tr_labels==1], bins=40, alpha=0.7, color='#e74c3c', density=True, label='Cheat')
ax.axvline(mean_clean_si, color='#27ae60', ls='--', lw=1.5,
           label=f'Clean mean={mean_clean_si:.4f}')
ax.axvline(mean_cheat_si, color='#c0392b', ls='--', lw=1.5,
           label=f'Cheat mean={mean_cheat_si:.4f}')
ax.set_xlabel('Self-Influence'); ax.set_ylabel('Density')
ax.set_title('Self-Influence by Class', fontweight='bold'); ax.legend(fontsize=8)

ax = axes[1]
bottom20 = np.argsort(np.abs(self_inf))[:20]
colors20 = ['#e74c3c' if tr_labels[i]==1 else '#2ecc71' for i in bottom20]
ax.bar(range(20), np.abs(self_inf[bottom20]), color=colors20, alpha=0.85)
ax.set_xticks(range(20))
ax.set_xticklabels([f'G{game_ids_seq[tr_idx[i]]}' for i in bottom20],
                    rotation=45, ha='right', fontsize=7)
ax.set_title('Lowest |Self-Influence| — Least Informative Samples', fontweight='bold')
ax.set_ylabel('|Self-Influence|')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#e74c3c', label='Cheat'),
                   Patch(color='#2ecc71', label='Clean')])
plt.suptitle('Koh & Liang Self-Influence Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_selfinf.png'] = savefig('fig_v2_selfinf.png')

# Fig 9: Influence heatmap
var_per_train = influence_mat.var(axis=0)
top30_var = np.argsort(var_per_train)[-30:]
fig, ax = plt.subplots(figsize=(14, 6))
inf_sub = influence_mat[:20, top30_var]
im = ax.imshow(inf_sub, aspect='auto', cmap='RdBu_r',
               vmin=-np.abs(inf_sub).max(), vmax=np.abs(inf_sub).max())
ax.set_xlabel('Training Games (highest influence variance)')
ax.set_ylabel('Suspicious Test Games (rank)')
ax.set_title('K&L Influence Heatmap (top-20 test × top-30 train)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Influence Score')
plt.tight_layout()
figures['fig_v2_heatmap.png'] = savefig('fig_v2_heatmap.png')

# Fig 10: Cheater Zone
sample_moves = moves_df.sample(min(5000, len(moves_df)), random_state=SEED)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, lv, title, color in zip(axes, [0, 1], ['Clean Moves', 'Cheat Moves'],
                                 ['#2ecc71', '#e74c3c']):
    sub = sample_moves[sample_moves['move_label'] == lv]
    ax.scatter(sub['SF_D20_Rank'].clip(0,10), sub['LC0_D20_Rank'].clip(0,10),
               c=color, alpha=0.3, s=8)
    ax.set_xlabel('SF D20 Rank'); ax.set_ylabel('LC0 D20 Rank')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 10.5)
    ax.add_patch(plt.Rectangle((-0.5, 4.5), 6, 6, lw=2,
                               edgecolor='navy', facecolor='none', ls='--'))
    ax.text(0.05, 9.5, 'Cheater Zone', fontsize=8, color='navy', va='top')
plt.suptitle('Cheater Zone — SF vs LC0 D20 Rank', fontsize=13, fontweight='bold')
plt.tight_layout()
figures['fig_v2_cheaterzone.png'] = savefig('fig_v2_cheaterzone.png')

print(f'\n  {len(figures)} figures generated.')
