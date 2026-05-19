"""
influence_analysis.py
Implements TracIn influence functions on CheatDetectorPerMove.
Reference: Koh & Liang (2017) "Understanding Black-box Predictions via Influence Functions"
           https://arxiv.org/abs/1703.04730

TracIn approximation: Pruthi et al. (2020) — sum of gradient dot products across checkpoints.
influence(z_train, z_test) = Σ_k  lr_k * ∇L(z_test; θ_k) · ∇L(z_train; θ_k)

Only classifier-head gradients are used (16k params) for tractability on CPU.
"""

import ast, json, warnings, time, os, pickle
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
sns.set_theme(style='darkgrid')
DEVICE = torch.device('cpu')
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)
BASE   = r'c:\Users\Zura\Desktop\Selieri'

# ── 1. Load & parse data ──────────────────────────────────────────────────────
print("=" * 60)
print("SELIERI — Influence Function Analysis")
print("=" * 60)
print("\n[1] Loading data ...")
df = pd.read_excel(f'{BASE}\\combined_features_labels.xlsx')
print(f"    Shape: {df.shape}")

def parse_array(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return []
    if isinstance(s, (list, np.ndarray)): return list(s)
    s = str(s).strip()
    if not s or s in ('nan','None','[]'): return []
    try:    return ast.literal_eval(s)
    except:
        try:    return json.loads(s)
        except: return []

ARRAY_COLS = [c for c in [
    'Side','MoveNo',
    'SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
    'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness',
    'LC0_D20_Rank','LC0_D20_CPL','LC0_D20_AdvWP','LC0_D20_BestWP','LC0_D20_WCL',
    'LC0_D20_Ambiguity05','LC0_D20_difNextBest','LC0_D20_difNextWorst','LC0_D20_Sharpness',
    'white_labels','black_labels',
] if c in df.columns]

for col in ARRAY_COLS:
    df[col + '_arr'] = df[col].apply(parse_array)

FEAT_COLS = [
    'SF_D20_Rank','SF_D20_CPL','SF_D20_AdvWP','SF_D20_BestWP','SF_D20_WCL',
    'SF_D20_Ambiguity05','SF_D20_difNextBest','SF_D20_difNextWorst','SF_D20_Sharpness',
    'LC0_D20_Rank','LC0_D20_CPL','LC0_D20_AdvWP','LC0_D20_BestWP','LC0_D20_WCL',
    'LC0_D20_Ambiguity05','LC0_D20_difNextBest','LC0_D20_difNextWorst','LC0_D20_Sharpness',
]

# ── 2. Build move-level df ────────────────────────────────────────────────────
print("[2] Building move-level dataframe ...")
rows = []
for _, game in df.iterrows():
    sides    = game['Side_arr']
    w_labels = game['white_labels_arr']
    b_labels = game['black_labels_arr']
    feat_arrays = {fc: game[fc + '_arr'] for fc in FEAT_COLS if fc + '_arr' in game.index}
    w_idx = b_idx = 0
    for i, side in enumerate(sides):
        if side == 'W':
            ml = int(w_labels[w_idx]) if w_idx < len(w_labels) else 0
            w_idx += 1
        else:
            ml = int(b_labels[b_idx]) if b_idx < len(b_labels) else 0
            b_idx += 1
        row = {'game_id': game['game_id'], 'move_idx': i, 'side': side, 'move_label': ml,
               'cheater_side': game['cheater_side'], 'phase': game['phase']}
        for fc, arr in feat_arrays.items():
            row[fc] = arr[i] if i < len(arr) else np.nan
        rows.append(row)

moves_df = pd.DataFrame(rows)
for fc in FEAT_COLS:
    moves_df[fc] = pd.to_numeric(moves_df[fc], errors='coerce')
moves_df.dropna(subset=FEAT_COLS, how='all', inplace=True)
moves_df.reset_index(drop=True, inplace=True)
print(f"    Moves: {len(moves_df):,}  |  Cheat: {moves_df['move_label'].sum():,} ({moves_df['move_label'].mean():.1%})")

# ── 3. Build sequences ────────────────────────────────────────────────────────
print("[3] Building sequences ...")
sequences, label_sequences, game_ids_ordered, phases_ordered = [], [], [], []
for gid, group in sorted(moves_df.groupby('game_id'), key=lambda x: x[0]):
    seq = group[FEAT_COLS].values.astype(np.float32)
    lbl = group['move_label'].values.astype(np.float32)
    if seq.shape[0] < 5: continue
    sequences.append(seq); label_sequences.append(lbl)
    game_ids_ordered.append(gid)
    phases_ordered.append(group['phase'].iloc[0])

game_ids_ordered = np.array(game_ids_ordered)
phases_ordered   = np.array(phases_ordered)
lengths_all      = [s.shape[0] for s in sequences]
total_moves      = sum(lengths_all)
total_cheat      = int(sum(l.sum() for l in label_sequences))
pos_weight_val   = (total_moves - total_cheat) / max(total_cheat, 1)
print(f"    Games: {len(sequences)}  |  Imbalance {pos_weight_val:.1f}:1")

# Normalise
all_moves_arr = np.concatenate(sequences, axis=0)
scaler = StandardScaler()
scaler.fit(np.nan_to_num(all_moves_arr, nan=0.0))
sequences_norm = [scaler.transform(np.nan_to_num(s, nan=0.0)).astype(np.float32) for s in sequences]

# Split
game_has_cheat = np.array([int(l.sum() > 0) for l in label_sequences])
idx = np.arange(len(sequences_norm))
tr_idx, tmp_idx, _, y_tmp = train_test_split(idx, game_has_cheat, test_size=0.30, stratify=game_has_cheat, random_state=SEED)
va_idx, te_idx, _, _      = train_test_split(tmp_idx, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

tr_moves_flat = np.concatenate([label_sequences[i] for i in tr_idx])
pos_weight_val_train = (tr_moves_flat == 0).sum() / max((tr_moves_flat == 1).sum(), 1)

# ── 4. Model & DataLoaders ────────────────────────────────────────────────────
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
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return sequences_norm[idx], label_sequences[idx]

BATCH = 32
tr_loader = DataLoader(ChessSeqDataset(tr_idx), batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
va_loader = DataLoader(ChessSeqDataset(va_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
te_loader = DataLoader(ChessSeqDataset(te_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

# Single-example loaders for gradient computation
tr_loader_single = DataLoader(ChessSeqDataset(tr_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)
te_loader_single = DataLoader(ChessSeqDataset(te_idx), batch_size=1, shuffle=False, collate_fn=collate_fn)

class CheatDetectorPerMove(nn.Module):
    def __init__(self, n_features=18, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers>1 else 0.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.classifier(out).squeeze(-1)

# ── 5. Train with checkpoint saving ──────────────────────────────────────────
print("\n[4] Training with checkpoint saving ...")
model     = CheatDetectorPerMove(n_features=18).to(DEVICE)
pos_wt    = torch.tensor([pos_weight_val_train], device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_wt, reduction='none')
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

EPOCHS   = 40
PATIENCE = 7
CKPT_EVERY = 4   # save checkpoint every N epochs

checkpoints = []  # list of (state_dict, lr, epoch)
history = {'train_loss':[],'val_loss':[],'train_f1':[],'val_f1':[]}
best_val, pat_ctr, best_state = float('inf'), 0, None

def run_epoch(loader, train=True):
    model.train(train)
    total_loss, total_batches = 0.0, 0
    all_preds, all_true = [], []
    with torch.set_grad_enabled(train):
        for x, lengths, y in loader:
            x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            logits = model(x, lengths)
            mask   = torch.arange(logits.size(1), device=DEVICE)[None,:] < lengths[:,None]
            loss   = criterion(logits, y)
            loss   = (loss * mask.float()).sum() / mask.float().sum()
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            probs = torch.sigmoid(logits).detach()
            all_preds.extend((probs[mask]>=0.5).long().cpu().numpy())
            all_true.extend(y[mask].long().cpu().numpy())
            total_loss += loss.item(); total_batches += 1
    return total_loss/total_batches, f1_score(all_true, all_preds, zero_division=0)

print(f'{"Epoch":>5} {"Tr Loss":>9} {"Va Loss":>9} {"Tr F1":>8} {"Va F1":>8} {"LR":>10}')
t0 = time.time()
for epoch in range(1, EPOCHS+1):
    tr_loss, tr_f1 = run_epoch(tr_loader, True)
    va_loss, va_f1 = run_epoch(va_loader, False)
    scheduler.step(va_loss)
    cur_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
    history['train_f1'].append(tr_f1);     history['val_f1'].append(va_f1)
    print(f'{epoch:5d} {tr_loss:9.4f} {va_loss:9.4f} {tr_f1:8.3f} {va_f1:8.3f} {cur_lr:10.6f}  [{time.time()-t0:.0f}s]')

    # Save checkpoint every CKPT_EVERY epochs
    if epoch % CKPT_EVERY == 0:
        checkpoints.append({
            'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'lr': cur_lr, 'epoch': epoch
        })
        print(f'    [ckpt] Checkpoint saved at epoch {epoch}')

    if va_loss < best_val:
        best_val   = va_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pat_ctr    = 0
    else:
        pat_ctr += 1
        if pat_ctr >= PATIENCE:
            print(f'Early stopping at epoch {epoch}'); break

# Always save final best as a checkpoint
checkpoints.append({'state_dict': best_state, 'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch})
model.load_state_dict(best_state)
print(f'Training complete. {len(checkpoints)} checkpoints saved.')

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
print("\n[5] Evaluating model ...")
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
            p = probs[i,:L]; t = y[i,:L].numpy().astype(int)
            all_probs.extend(p); all_preds.extend((p>=0.5).astype(int)); all_true.extend(t)
            game_probs_list.append(float(p.max())); game_true_list.append(int(t.max()))

all_probs  = np.array(all_probs); all_preds = np.array(all_preds); all_true = np.array(all_true)
game_probs = np.array(game_probs_list); game_true = np.array(game_true_list)
game_preds = (game_probs>=0.5).astype(int)

print(f'    Move F1  : {f1_score(all_true,all_preds,zero_division=0):.4f}')
print(f'    Move AUC : {roc_auc_score(all_true,all_probs):.4f}')
print(f'    Game AUC : {roc_auc_score(game_true,game_probs):.4f}')

# Per-move probs per test game (for visualisation)
model.eval()
test_game_move_probs = {}  # te_local_idx -> array of move probs
with torch.no_grad():
    for local_i, global_i in enumerate(te_idx):
        seq = torch.tensor(sequences_norm[global_i]).unsqueeze(0)
        L   = torch.tensor([seq.shape[1]])
        logits = model(seq, L)
        probs  = torch.sigmoid(logits).squeeze(0).numpy()
        test_game_move_probs[local_i] = probs[:L.item()]

# ── 7. TracIn Influence Functions ─────────────────────────────────────────────
print("\n[6] Computing TracIn influence functions ...")
print(f"    Checkpoints: {len(checkpoints)}")
print(f"    Train games: {len(tr_idx)}  |  Test games: {len(te_idx)}")
print(f"    Gradient params: classifier head only ({sum(p.numel() for p in model.classifier.parameters()):,} params)")

def get_classifier_params(m):
    return list(m.classifier.parameters())

def compute_example_grad(m, x, lengths, y, crit, params):
    """Gradient of masked BCE loss w.r.t. classifier params for one example."""
    m.zero_grad()
    logits = m(x, lengths)
    mask   = torch.arange(logits.size(1))[None,:] < lengths[:,None]
    loss   = crit(logits, y)
    loss   = (loss * mask.float()).sum() / mask.float().sum()
    grads  = torch.autograd.grad(loss, params, retain_graph=False)
    return torch.cat([g.detach().flatten() for g in grads]).numpy()

n_train = len(tr_idx)
n_test  = len(te_idx)
influence_matrix = np.zeros((n_test, n_train))  # (test, train)
self_influence   = np.zeros(n_train)

crit_for_grad = nn.BCEWithLogitsLoss(pos_weight=pos_wt, reduction='none')

t_inf = time.time()
for ckpt_idx, ckpt in enumerate(checkpoints):
    print(f"\n  Checkpoint {ckpt_idx+1}/{len(checkpoints)} (epoch={ckpt['epoch']}, lr={ckpt['lr']:.6f})")
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    params = get_classifier_params(model)

    # Compute train gradients
    print(f"    Computing {n_train} train gradients ...")
    train_grads = []
    for i, (x, lengths, y) in enumerate(tr_loader_single):
        g = compute_example_grad(model, x, lengths, y, crit_for_grad, params)
        train_grads.append(g)
        if (i+1) % 200 == 0:
            print(f"      {i+1}/{n_train} [{time.time()-t_inf:.0f}s]")
    train_grads = np.array(train_grads)  # (n_train, n_params)

    # Compute test gradients
    print(f"    Computing {n_test} test gradients ...")
    test_grads = []
    for i, (x, lengths, y) in enumerate(te_loader_single):
        g = compute_example_grad(model, x, lengths, y, crit_for_grad, params)
        test_grads.append(g)
    test_grads = np.array(test_grads)  # (n_test, n_params)

    lr = ckpt['lr']
    influence_matrix += lr * (test_grads @ train_grads.T)
    self_influence   += lr * np.sum(train_grads * train_grads, axis=1)

    print(f"    Influence range: [{influence_matrix.min():.4f}, {influence_matrix.max():.4f}]")

model.load_state_dict(best_state)
print(f"\n  Influence computation complete in {time.time()-t_inf:.0f}s")

# ── 8. Analysis ───────────────────────────────────────────────────────────────
print("\n[7] Running influence analysis ...")

# Map back to game IDs
tr_game_ids = game_ids_ordered[tr_idx]
te_game_ids = game_ids_ordered[te_idx]
tr_phases   = phases_ordered[tr_idx]
te_phases   = phases_ordered[te_idx]
tr_has_cheat= game_has_cheat[tr_idx]
te_has_cheat= game_has_cheat[te_idx]

# Sort test games by suspicion score (game_probs)
sorted_te_by_suspicion = np.argsort(game_probs)[::-1]  # most suspicious first

# ── Figure 1: Influence Heatmap ───────────────────────────────────────────────
print("  Generating Figure 1: Influence Heatmap ...")
# Take top-30 most suspicious test games and sort train by cheat status
top_te  = sorted_te_by_suspicion[:30]
# Sort train: cheat games first, then by self-influence
tr_sort = np.argsort(tr_has_cheat * 1000 + self_influence / self_influence.max())[::-1][:60]

sub_inf = influence_matrix[np.ix_(top_te, tr_sort)]

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(sub_inf, aspect='auto', cmap='RdBu_r',
               vmin=-np.percentile(np.abs(sub_inf), 95),
               vmax= np.percentile(np.abs(sub_inf), 95))
plt.colorbar(im, ax=ax, label='TracIn Influence Score')

# Label axes
ax.set_xlabel('Training Games (sorted: cheating → clean)', fontsize=11)
ax.set_ylabel('Test Games (sorted by suspicion score ↓)', fontsize=11)

# Mark boundary between cheat/clean training games
n_cheat_tr = tr_has_cheat[tr_sort].sum()
ax.axvline(n_cheat_tr - 0.5, color='yellow', lw=2, ls='--', label='Cheat|Clean boundary')

# Colour y-axis labels by true label
for i, te_i in enumerate(top_te):
    color = '#e74c3c' if te_has_cheat[te_i] else '#2ecc71'
    ax.get_yticklabels()  # ensure ticks exist
ax.set_yticks(range(len(top_te)))
ax.set_yticklabels([f'Game {te_game_ids[i]} ({"CHEAT" if te_has_cheat[i] else "clean"}, p={game_probs[i]:.2f})'
                    for i in top_te], fontsize=7)
ax.set_xticks([]); ax.legend(loc='upper right', fontsize=9)
ax.set_title('TracIn Influence Heatmap\n(Red = training game supports test prediction, Blue = opposes)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}\\fig_influence_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: fig_influence_heatmap.png")

# ── Figure 2: Proponents & Opponents ─────────────────────────────────────────
print("  Generating Figure 2: Proponents & Opponents ...")
N_SHOWCASE = 3  # number of test games to showcase
showcase_te = sorted_te_by_suspicion[:N_SHOWCASE]

def get_move_probs_for_train(global_idx):
    model.eval()
    with torch.no_grad():
        seq = torch.tensor(sequences_norm[global_idx]).unsqueeze(0)
        L   = torch.tensor([seq.shape[1]])
        logits = model(seq, L)
        probs  = torch.sigmoid(logits).squeeze(0).numpy()
    return probs[:L.item()]

fig = plt.figure(figsize=(18, 5 * N_SHOWCASE))
gs  = gridspec.GridSpec(N_SHOWCASE, 3, figure=fig, hspace=0.5, wspace=0.35)

for row, te_local in enumerate(showcase_te):
    te_global = te_idx[te_local]
    inf_row   = influence_matrix[te_local]

    # Top-2 proponents (most positive influence) and top-1 opponent (most negative)
    proponent_local = np.argsort(inf_row)[-1]
    opponent_local  = np.argsort(inf_row)[0]

    for col, (tr_local, label, color) in enumerate([
        (None,            f'TEST  Game {te_game_ids[te_local]}\n(predicted={game_probs[te_local]:.2f}, true={"CHEAT" if te_has_cheat[te_local] else "clean"})', '#9b59b6'),
        (proponent_local, f'TOP PROPONENT  Game {tr_game_ids[proponent_local]}\n(influence={inf_row[proponent_local]:+.4f}, {"CHEAT" if tr_has_cheat[proponent_local] else "clean"})', '#e74c3c'),
        (opponent_local,  f'TOP OPPONENT   Game {tr_game_ids[opponent_local]}\n(influence={inf_row[opponent_local]:+.4f}, {"CHEAT" if tr_has_cheat[opponent_local] else "clean"})', '#3498db'),
    ]):
        ax = fig.add_subplot(gs[row, col])

        if tr_local is None:
            probs    = test_game_move_probs[te_local]
            true_lbl = label_sequences[te_global]
        else:
            tr_global = tr_idx[tr_local]
            probs     = get_move_probs_for_train(tr_global)
            true_lbl  = label_sequences[tr_global]

        n = len(probs)
        moves = np.arange(n)
        ax.fill_between(moves, probs, alpha=0.25, color=color)
        ax.plot(moves, probs, color=color, lw=1.2)

        # Mark true cheat moves
        cheat_moves = np.where(true_lbl[:n] == 1)[0]
        if len(cheat_moves):
            ax.scatter(cheat_moves, probs[cheat_moves], color='black', s=15, zorder=5,
                       label='True cheat move')

        ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.7)
        ax.set_ylim(0, 1); ax.set_xlim(0, n)
        ax.set_xlabel('Move index'); ax.set_ylabel('Cheat prob')
        ax.set_title(label, fontsize=8, fontweight='bold', color=color)
        if len(cheat_moves): ax.legend(fontsize=7)

plt.suptitle('Proponent & Opponent Analysis\n(TracIn Influence Functions)', fontsize=13, fontweight='bold')
plt.savefig(f'{BASE}\\fig_proponents_opponents.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: fig_proponents_opponents.png")

# ── Figure 3: Self-Influence Distribution ────────────────────────────────────
print("  Generating Figure 3: Self-Influence Distribution ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram by phase
cheat_si = self_influence[tr_has_cheat == 1]
clean_si = self_influence[tr_has_cheat == 0]
axes[0].hist(clean_si, bins=40, alpha=0.65, color='#2ecc71', density=True, label=f'Clean games (n={len(clean_si)})')
axes[0].hist(cheat_si, bins=40, alpha=0.65, color='#e74c3c', density=True, label=f'Cheat games (n={len(cheat_si)})')
axes[0].set_xlabel('Self-Influence Score'); axes[0].set_ylabel('Density')
axes[0].set_title('Self-Influence Distribution\nby Game Type', fontweight='bold')
axes[0].legend()
axes[0].text(0.62, 0.80,
    f'Cheat mean: {cheat_si.mean():.4f}\nClean mean: {clean_si.mean():.4f}',
    transform=axes[0].transAxes, fontsize=9,
    bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8, edgecolor='white'))

# Sorted self-influence — flag outliers
sorted_si_idx = np.argsort(self_influence)
axes[1].scatter(range(n_train), self_influence[sorted_si_idx],
                c=['#e74c3c' if tr_has_cheat[i] else '#2ecc71' for i in sorted_si_idx],
                s=8, alpha=0.6)
# Highlight bottom-10 (potential noise)
bottom10 = sorted_si_idx[:10]
axes[1].scatter(range(10), self_influence[bottom10],
                c='black', s=40, zorder=5, label='Potential noisy samples')
axes[1].set_xlabel('Training game (sorted by self-influence)')
axes[1].set_ylabel('Self-Influence Score')
axes[1].set_title('Sorted Self-Influence\n(low scores = potentially mislabelled)', fontweight='bold')
axes[1].legend()

plt.suptitle('Self-Influence Analysis — Training Data Quality', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}\\fig_self_influence.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: fig_self_influence.png")

# ── Figure 4: Noisy Sample Deep-Dive ─────────────────────────────────────────
print("  Generating Figure 4: Noisy Sample Detection ...")
bottom10_local = np.argsort(self_influence)[:10]
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
axes = axes.flatten()
for i, tr_local in enumerate(bottom10_local):
    tr_global = tr_idx[tr_local]
    probs     = get_move_probs_for_train(tr_global)
    true_lbl  = label_sequences[tr_global]
    n         = len(probs)
    ax        = axes[i]
    ax.fill_between(range(n), probs, alpha=0.2, color='#f39c12')
    ax.plot(probs, color='#f39c12', lw=1.2)
    cheat_moves = np.where(true_lbl[:n]==1)[0]
    if len(cheat_moves):
        ax.scatter(cheat_moves, probs[cheat_moves], color='black', s=12, zorder=5)
    ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.7)
    ax.set_ylim(0,1); ax.set_xlim(0,n)
    phase_str = tr_phases[tr_local]
    si_val    = self_influence[tr_local]
    n_cheat   = int(true_lbl.sum())
    ax.set_title(f'Game {tr_game_ids[tr_local]}\n{phase_str} | SI={si_val:.4f} | {n_cheat} cheat moves',
                 fontsize=7.5, fontweight='bold')
    ax.set_xlabel('Move', fontsize=7); ax.set_ylabel('Cheat prob', fontsize=7)
plt.suptitle('Bottom-10 Self-Influence Training Games\n(potential labelling noise or simulation artefacts)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{BASE}\\fig_noisy_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Saved: fig_noisy_samples.png")

# ── 9. Print summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("INFLUENCE FUNCTION SUMMARY")
print("="*60)
print(f"\nTop-10 most influential training games (proponents for suspicious test games):")
top_proponents = np.argsort(influence_matrix[sorted_te_by_suspicion[:10]].mean(axis=0))[::-1][:10]
for rank, tr_local in enumerate(top_proponents):
    print(f"  {rank+1}. Game {tr_game_ids[tr_local]:5d}  phase={tr_phases[tr_local]:<6}  "
          f"self_inf={self_influence[tr_local]:.4f}  mean_inf={influence_matrix[:,tr_local].mean():+.4f}")

print(f"\nBottom-10 self-influence training games (potential noise):")
for rank, tr_local in enumerate(np.argsort(self_influence)[:10]):
    print(f"  {rank+1}. Game {tr_game_ids[tr_local]:5d}  phase={tr_phases[tr_local]:<6}  "
          f"self_inf={self_influence[tr_local]:.6f}  cheat_moves={int(label_sequences[tr_idx[tr_local]].sum())}")

print(f"\nMean self-influence — cheat games : {self_influence[tr_has_cheat==1].mean():.4f}")
print(f"Mean self-influence — clean games : {self_influence[tr_has_cheat==0].mean():.4f}")
print(f"\nFigures saved to: {BASE}")
print("  fig_influence_heatmap.png")
print("  fig_proponents_opponents.png")
print("  fig_self_influence.png")
print("  fig_noisy_samples.png")
print("\nDone.")
