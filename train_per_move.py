import ast, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

warnings.filterwarnings('ignore')
DEVICE = torch.device('cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BASE = r'c:\Users\Zura\Desktop\Selieri'

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading combined_features_labels.xlsx ...")
df = pd.read_excel(f'{BASE}\\combined_features_labels.xlsx')
print(f"  Shape: {df.shape}")

# ── 2. Parse serialised array columns ────────────────────────────────────────
def parse_array(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    if isinstance(s, (list, np.ndarray)):
        return list(s)
    s = str(s).strip()
    if not s or s in ('nan', 'None', '[]'):
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return []

ARRAY_COLS = [
    'Side', 'MoveNo',
    'SF_D10_Rank','SF_D10_CPL','SF_D10_AdvWP','SF_D10_BestWP','SF_D10_WCL',
    'SF_D10_Ambiguity05','SF_D10_difNextBest','SF_D10_difNextWorst','SF_D10_Sharpness',
    'SF_D15_Rank','SF_D15_CPL','SF_D15_AdvWP','SF_D15_BestWP','SF_D15_WCL',
    'SF_D15_Ambiguity05','SF_D15_difNextBest','SF_D15_difNextWorst','SF_D15_Sharpness',
    'SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
    'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness',
    'LC0_D10_Rank','LC0_D10_CPL','LC0_D10_AdvWP','LC0_D10_BestWP','LC0_D10_WCL',
    'LC0_D10_Ambiguity05','LC0_D10_difNextBest','LC0_D10_difNextWorst','LC0_D10_Sharpness',
    'LC0_D15_Rank','LC0_D15_CPL','LC0_D15_AdvWP','LC0_D15_BestWP','LC0_D15_WCL',
    'LC0_D15_Ambiguity05','LC0_D15_difNextBest','LC0_D15_difNextWorst','LC0_D15_Sharpness',
    'LC0_D20_Rank','LC0_D20_CPL','LC0_D20_AdvWP','LC0_D20_BestWP','LC0_D20_WCL',
    'LC0_D20_Ambiguity05','LC0_D20_difNextBest','LC0_D20_difNextWorst','LC0_D20_Sharpness',
    'white_labels', 'black_labels',
]
ARRAY_COLS = [c for c in ARRAY_COLS if c in df.columns]
print("Parsing arrays ...")
for col in ARRAY_COLS:
    df[col + '_arr'] = df[col].apply(parse_array)

# ── 3. Build move-level dataframe ─────────────────────────────────────────────
FEAT_COLS_D20 = [
    'SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
    'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness',
    'LC0_D20_Rank','LC0_D20_CPL','LC0_D20_AdvWP','LC0_D20_BestWP','LC0_D20_WCL',
    'LC0_D20_Ambiguity05','LC0_D20_difNextBest','LC0_D20_difNextWorst','LC0_D20_Sharpness',
]

print("Building move-level dataframe ...")
rows = []
for _, game in df.iterrows():
    sides    = game['Side_arr']
    w_labels = game['white_labels_arr']
    b_labels = game['black_labels_arr']
    n_moves  = len(sides)
    feat_arrays = {fc: game[fc + '_arr'] for fc in FEAT_COLS_D20 if fc + '_arr' in game.index}
    w_idx = b_idx = 0
    for i, side in enumerate(sides):
        if side == 'W':
            move_label = int(w_labels[w_idx]) if w_idx < len(w_labels) else 0
            w_idx += 1
        else:
            move_label = int(b_labels[b_idx]) if b_idx < len(b_labels) else 0
            b_idx += 1
        row = {'game_id': game['game_id'], 'move_idx': i, 'side': side, 'move_label': move_label}
        for fc, arr in feat_arrays.items():
            row[fc] = arr[i] if i < len(arr) else np.nan
        rows.append(row)

moves_df = pd.DataFrame(rows)
for fc in FEAT_COLS_D20:
    if fc in moves_df.columns:
        moves_df[fc] = pd.to_numeric(moves_df[fc], errors='coerce')
moves_df.dropna(subset=FEAT_COLS_D20, how='all', inplace=True)
moves_df.reset_index(drop=True, inplace=True)
print(f"  Move-level df: {moves_df.shape}")
print(f"  Cheat moves: {moves_df['move_label'].sum():,} / {len(moves_df):,} ({moves_df['move_label'].mean():.1%})")

# ── 4. Build sequences ────────────────────────────────────────────────────────
print("Building sequences ...")
sequences, label_sequences = [], []
for gid, group in sorted(moves_df.groupby('game_id'), key=lambda x: x[0]):
    seq = group[FEAT_COLS_D20].values.astype(np.float32)
    lbl = group['move_label'].values.astype(np.float32)
    if seq.shape[0] < 5:
        continue
    sequences.append(seq)
    label_sequences.append(lbl)

lengths = [s.shape[0] for s in sequences]
total_moves = sum(lengths)
total_cheat = int(sum(l.sum() for l in label_sequences))
print(f"  Games: {len(sequences)}")
print(f"  Move counts: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}")
print(f"  Total moves: {total_moves:,}  Cheat: {total_cheat:,} ({total_cheat/total_moves:.1%})")
print(f"  Imbalance: {(total_moves-total_cheat)/max(total_cheat,1):.1f}:1")

# ── 5. Normalise + split ──────────────────────────────────────────────────────
all_moves_arr = np.concatenate(sequences, axis=0)
scaler = StandardScaler()
scaler.fit(np.nan_to_num(all_moves_arr, nan=0.0))
sequences_norm = [scaler.transform(np.nan_to_num(s, nan=0.0)).astype(np.float32) for s in sequences]

game_has_cheat = np.array([int(l.sum() > 0) for l in label_sequences])
idx = np.arange(len(sequences_norm))
tr_idx, tmp_idx, _, y_tmp = train_test_split(idx, game_has_cheat, test_size=0.30, stratify=game_has_cheat, random_state=SEED)
va_idx, te_idx, _, _      = train_test_split(tmp_idx, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

tr_moves_flat = np.concatenate([label_sequences[i] for i in tr_idx])
pos_weight_val = (tr_moves_flat == 0).sum() / max((tr_moves_flat == 1).sum(), 1)
print(f"\nTrain:{len(tr_idx)}  Val:{len(va_idx)}  Test:{len(te_idx)}")
print(f"pos_weight = {pos_weight_val:.2f}")

# ── 6. Dataset / DataLoader ───────────────────────────────────────────────────
def collate_fn(batch):
    seqs, lbls = zip(*batch)
    lengths  = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len  = lengths.max().item()
    n_feat   = seqs[0].shape[1]
    padded_x = torch.zeros(len(seqs), max_len, n_feat)
    padded_y = torch.zeros(len(seqs), max_len)
    for i, (s, l, length) in enumerate(zip(seqs, lbls, lengths)):
        padded_x[i, :length] = torch.tensor(s)
        padded_y[i, :length] = torch.tensor(l)
    return padded_x, lengths, padded_y

class ChessSeqDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return sequences_norm[idx], label_sequences[idx]

BATCH = 32
tr_loader = DataLoader(ChessSeqDataset(tr_idx), batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
va_loader = DataLoader(ChessSeqDataset(va_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
te_loader = DataLoader(ChessSeqDataset(te_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

# ── 7. Model ──────────────────────────────────────────────────────────────────
class CheatDetectorPerMove(nn.Module):
    def __init__(self, n_features=18, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.classifier(out).squeeze(-1)  # (B, T)

# ── 8. Training helper ────────────────────────────────────────────────────────
def train_model(name, model, tr_loader, va_loader, pos_weight_val, epochs=40, patience=7, lr=1e-3):
    pos_wt    = torch.tensor([pos_weight_val], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    history   = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val, pat_ctr, best_state = float('inf'), 0, None

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, total_batches = 0.0, 0
        all_preds, all_true = [], []
        with torch.set_grad_enabled(train):
            for x, lengths, y in loader:
                x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                logits = model(x, lengths)
                mask   = torch.arange(logits.size(1), device=DEVICE)[None, :] < lengths[:, None]
                loss   = criterion(logits, y)
                loss   = (loss * mask.float()).sum() / mask.float().sum()
                if train:
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                probs = torch.sigmoid(logits).detach()
                all_preds.extend((probs[mask] >= 0.5).long().cpu().numpy())
                all_true.extend(y[mask].long().cpu().numpy())
                total_loss += loss.item(); total_batches += 1
        return total_loss / total_batches, f1_score(all_true, all_preds, zero_division=0)

    print(f'\n{"="*55}')
    print(f'Training: {name}')
    print(f'{"="*55}')
    print(f'{"Epoch":>5} {"Tr Loss":>9} {"Va Loss":>9} {"Tr F1":>8} {"Va F1":>8}')
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_f1 = run_epoch(tr_loader, True)
        va_loss, va_f1 = run_epoch(va_loader, False)
        scheduler.step(va_loss)
        history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
        history['train_f1'].append(tr_f1);     history['val_f1'].append(va_f1)
        elapsed = time.time() - t0
        print(f'{epoch:5d} {tr_loss:9.4f} {va_loss:9.4f} {tr_f1:8.3f} {va_f1:8.3f}  [{elapsed:.0f}s]')
        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_ctr    = 0
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                print(f'Early stopping at epoch {epoch}'); break
    model.load_state_dict(best_state)
    return history

def evaluate(model, te_loader):
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    game_probs_list, game_true_list = [], []
    with torch.no_grad():
        for x, lengths, y in te_loader:
            x, lengths = x.to(DEVICE), lengths.to(DEVICE)
            logits = model(x, lengths)
            probs  = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(lengths)):
                L = lengths[i].item()
                p = probs[i, :L]; t = y[i, :L].numpy().astype(int)
                all_probs.extend(p); all_preds.extend((p >= 0.5).astype(int)); all_true.extend(t)
                game_probs_list.append(float(p.max())); game_true_list.append(int(t.max()))
    all_probs  = np.array(all_probs);  all_preds = np.array(all_preds);  all_true = np.array(all_true)
    game_probs = np.array(game_probs_list); game_true = np.array(game_true_list)
    game_preds = (game_probs >= 0.5).astype(int)
    return {
        'probs': all_probs, 'preds': all_preds, 'true': all_true,
        'game_probs': game_probs, 'game_preds': game_preds, 'game_true': game_true,
        'move_acc':  accuracy_score(all_true, all_preds),
        'move_prec': precision_score(all_true, all_preds, zero_division=0),
        'move_rec':  recall_score(all_true, all_preds, zero_division=0),
        'move_f1':   f1_score(all_true, all_preds, zero_division=0),
        'move_auc':  roc_auc_score(all_true, all_probs),
        'game_acc':  accuracy_score(game_true, game_preds),
        'game_f1':   f1_score(game_true, game_preds, zero_division=0),
        'game_auc':  roc_auc_score(game_true, game_probs),
    }

# ── 9. Train full model (SF + Maia) ──────────────────────────────────────────
full_model = CheatDetectorPerMove(n_features=18).to(DEVICE)
print(f'Parameters: {sum(p.numel() for p in full_model.parameters()):,}')
full_history = train_model('Full (SF + Maia)', full_model, tr_loader, va_loader, pos_weight_val)
full_res = evaluate(full_model, te_loader)
full_res['name'] = 'Full (SF + Maia)'
full_res['history'] = full_history

# ── 10. Train SF-only control ─────────────────────────────────────────────────
SF_ONLY = ['SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
           'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness']

sf_sequences = [
    group[SF_ONLY].values.astype(np.float32)
    for _, group in sorted(moves_df.groupby('game_id'), key=lambda x: x[0])
    if group.shape[0] >= 5
]
sf_all = np.concatenate(sf_sequences, axis=0)
sf_scaler = StandardScaler()
sf_scaler.fit(np.nan_to_num(sf_all, nan=0.0))
sf_sequences_norm = [sf_scaler.transform(np.nan_to_num(s, nan=0.0)).astype(np.float32) for s in sf_sequences]

class SFDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return sf_sequences_norm[idx], label_sequences[idx]

sf_tr = DataLoader(SFDataset(tr_idx), batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
sf_va = DataLoader(SFDataset(va_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
sf_te = DataLoader(SFDataset(te_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

sf_model   = CheatDetectorPerMove(n_features=9).to(DEVICE)
sf_history = train_model('Control (SF only)', sf_model, sf_tr, sf_va, pos_weight_val)
sf_res     = evaluate(sf_model, sf_te)
sf_res['name']    = 'Control (SF only)'
sf_res['history'] = sf_history

# ── 11. Print results ─────────────────────────────────────────────────────────
print('\n' + '='*55)
print('RESULTS SUMMARY')
print('='*55)
header = f'{"Metric":<16} {"Full (SF+Maia)":>15} {"SF only":>12} {"Delta":>10}'
print(header)
print('-'*55)
metrics = [
    ('Move Accuracy',  'move_acc'),
    ('Move Precision', 'move_prec'),
    ('Move Recall',    'move_rec'),
    ('Move F1',        'move_f1'),
    ('Move AUC',       'move_auc'),
    ('Game Accuracy',  'game_acc'),
    ('Game F1',        'game_f1'),
    ('Game AUC',       'game_auc'),
]
for label, key in metrics:
    f = full_res[key]; s = sf_res[key]; d = f - s
    print(f'{label:<16} {f:>15.4f} {s:>12.4f} {d:>+10.4f}')
print('='*55)

print('\nFull model — classification report (per move):')
print(classification_report(full_res['true'], full_res['preds'], target_names=['Clean Move','Cheat Move']))

# ── 12. Save plots ────────────────────────────────────────────────────────────
sns.set_theme(style='darkgrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Metric comparison bar chart
met_keys  = ['move_acc','move_prec','move_rec','move_f1','move_auc']
met_names = ['Accuracy','Precision','Recall','F1','AUC']
x = np.arange(len(met_keys)); w = 0.35
ax = axes[0,0]
for i, (res, col) in enumerate(zip([full_res, sf_res], ['#3498db','#e74c3c'])):
    vals = [res[m] for m in met_keys]
    bars = ax.bar(x + i*w, vals, width=w, color=col, alpha=0.85, label=res['name'])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x+w/2); ax.set_xticklabels(met_names); ax.set_ylim(0,1.12)
ax.set_title('Per-Move Metrics', fontweight='bold'); ax.legend()

# ROC curves
ax = axes[0,1]
for res, col in zip([full_res, sf_res], ['#3498db','#e74c3c']):
    fpr, tpr, _ = roc_curve(res['true'], res['probs'])
    ax.plot(fpr, tpr, color=col, lw=2, label=f'{res["name"]} AUC={res["move_auc"]:.3f}')
ax.plot([0,1],[0,1],'k--',alpha=0.4); ax.set_title('ROC — Per-Move', fontweight='bold'); ax.legend()
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')

# Val loss + F1
ax = axes[0,2]
for res, col in zip([full_res, sf_res], ['#3498db','#e74c3c']):
    ax.plot(res['history']['val_loss'], color=col, lw=2, label=res['name']+' loss')
    ax.plot(res['history']['val_f1'],   color=col, lw=2, ls='--', alpha=0.7, label=res['name']+' F1')
ax.set_title('Val Loss (solid) & F1 (dashed)', fontweight='bold'); ax.legend(fontsize=7); ax.set_xlabel('Epoch')

# Confusion matrices
for idx2, (res, cmap) in enumerate(zip([full_res, sf_res], ['Blues','Reds'])):
    ax = axes[1, idx2]
    cm = confusion_matrix(res['true'], res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Clean','Cheat'], yticklabels=['Clean','Cheat'])
    ax.set_title(f'{res["name"]}\nMove F1={res["move_f1"]:.3f}  AUC={res["move_auc"]:.3f}', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

# Score distribution
ax = axes[1,2]
ax.hist(full_res['probs'][full_res['true']==0], bins=60, alpha=0.6, color='#2ecc71', density=True, label='Clean')
ax.hist(full_res['probs'][full_res['true']==1], bins=60, alpha=0.6, color='#e74c3c', density=True, label='Cheat')
ax.set_xlabel('Predicted Cheat Probability'); ax.set_ylabel('Density')
ax.set_title('Score Distribution (Full Model)', fontweight='bold'); ax.legend()

plt.suptitle(f'Per-Move BiLSTM  |  Move F1={full_res["move_f1"]:.3f}  Game AUC={full_res["game_auc"]:.3f}', fontsize=13, fontweight='bold')
plt.tight_layout()
out_path = f'{BASE}\\results_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nPlot saved: {out_path}')

# ── 13. Save models ───────────────────────────────────────────────────────────
import joblib

torch.save({
    'model_state_dict': full_model.state_dict(),
    'n_features': 18,
    'hidden': 128,
    'n_layers': 2,
    'dropout': 0.3,
    'feature_cols': FEAT_COLS_D20,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'results': {k: v for k, v in full_res.items() if k not in ('probs','preds','true','game_probs','game_preds','game_true','history')},
}, f'{BASE}\\selieri_model_full.pt')
print(f'Model saved: selieri_model_full.pt')

torch.save({
    'model_state_dict': sf_model.state_dict(),
    'n_features': 9,
    'hidden': 128,
    'n_layers': 2,
    'dropout': 0.3,
    'feature_cols': SF_ONLY,
    'scaler_mean': sf_scaler.mean_.tolist(),
    'scaler_scale': sf_scaler.scale_.tolist(),
    'results': {k: v for k, v in sf_res.items() if k not in ('probs','preds','true','game_probs','game_preds','game_true','history')},
}, f'{BASE}\\selieri_model_sfonly.pt')
print(f'Model saved: selieri_model_sfonly.pt')
