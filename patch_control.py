import json

path = r'C:\Users\Zura\Desktop\Selieri\selieri_cheating_detection.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# -- Patch cell 34: store full-model results under named variables -------------
cells[34]['source'] = ["""model.eval()
all_probs, all_preds, all_true = [], [], []

with torch.no_grad():
    for x, lengths, y in te_loader:
        x, lengths = x.to(DEVICE), lengths.to(DEVICE)
        logits = model(x, lengths)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(y.numpy().astype(int))

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_true  = np.array(all_true)

# Store for later comparison
full_results = {
    'name':   'Full Model (SF + Maia)',
    'probs':  all_probs,
    'preds':  all_preds,
    'true':   all_true,
    'acc':    accuracy_score(all_true, all_preds),
    'prec':   precision_score(all_true, all_preds),
    'rec':    recall_score(all_true, all_preds),
    'f1':     f1_score(all_true, all_preds),
    'auc':    roc_auc_score(all_true, all_probs),
    'history': history,
}

print('='*45)
print(f'  Test Accuracy  : {full_results[\"acc\"]:.4f}')
print(f'  Precision      : {full_results[\"prec\"]:.4f}')
print(f'  Recall         : {full_results[\"rec\"]:.4f}')
print(f'  F1 Score       : {full_results[\"f1\"]:.4f}')
print(f'  ROC-AUC        : {full_results[\"auc\"]:.4f}')
print('='*45)
print()
print(classification_report(all_true, all_preds, target_names=['Clean','Cheat']))
"""]

# -- New cells to insert after cell 37 (attention) ----------------------------
new_cells = [
    # Markdown header
    {
        "cell_type": "markdown",
        "metadata": {},
        "outputs": [],
        "source": [
            "## 5 - Control: Stockfish-Only Model\n",
            "\n",
            "To isolate the contribution of Maia/LC0 features, we train an identical BiLSTM using **only the 9 Stockfish D20 features** (no Maia data). "
            "The two models are then compared directly - if Maia adds signal, the full model should outperform the SF-only control."
        ]
    },
    # SF-only training code
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": ["""SF_ONLY_FEATURES = [
    'SF_D20_Rank', 'SF_D20_CPL', 'SF_D20_AdvWP', 'SF_D20_BestWP', 'SF_D20_WCL',
    'SF_D20_Ambiguity05', 'SF_D20_difNextBest', 'SF_D20_difNextWorst', 'SF_D20_Sharpness',
]

# Build SF-only sequences
sf_sequences, sf_labels = [], []
for gid, group in moves_df.groupby('game_id'):
    if gid not in game_labels:
        continue
    seq = group[SF_ONLY_FEATURES].values.astype(np.float32)
    if seq.shape[0] < 5:
        continue
    sf_sequences.append(seq)
    sf_labels.append(game_labels[gid])

# Normalise
sf_all = np.concatenate(sf_sequences, axis=0)
sf_scaler = StandardScaler()
sf_scaler.fit(np.nan_to_num(sf_all, nan=0.0))
sf_sequences_norm = [sf_scaler.transform(np.nan_to_num(s, nan=0.0)).astype(np.float32) for s in sf_sequences]

# Datasets (same split indices as full model)
class SFDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return sf_sequences_norm[idx], sf_labels[idx]

sf_tr_loader = DataLoader(SFDataset(tr_idx), batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
sf_va_loader = DataLoader(SFDataset(va_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
sf_te_loader = DataLoader(SFDataset(te_idx), batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

# Identical architecture, 9 input features
sf_model = CheatDetectorRNN(n_features=len(SF_ONLY_FEATURES)).to(DEVICE)
sf_criterion = nn.BCEWithLogitsLoss()
sf_optimizer = optim.Adam(sf_model.parameters(), lr=LR, weight_decay=1e-4)
sf_scheduler = optim.lr_scheduler.ReduceLROnPlateau(sf_optimizer, patience=3, factor=0.5)

sf_history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
sf_best_val  = float('inf')
sf_patience  = 0
sf_best_state = None

def run_sf_epoch(loader, train=True):
    sf_model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, lengths, y in loader:
            x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            logits = sf_model(x, lengths)
            loss   = sf_criterion(logits, y)
            if train:
                sf_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(sf_model.parameters(), 1.0)
                sf_optimizer.step()
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct    += (preds == y.long()).sum().item()
            total      += y.size(0)
            total_loss += loss.item() * y.size(0)
    return total_loss / total, correct / total

print('Training SF-only control model...')
print(f'{"Epoch":>5} {"Tr Loss":>9} {"Va Loss":>9} {"Tr Acc":>8} {"Va Acc":>8}')
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_sf_epoch(sf_tr_loader, train=True)
    va_loss, va_acc = run_sf_epoch(sf_va_loader, train=False)
    sf_scheduler.step(va_loss)

    sf_history['train_loss'].append(tr_loss)
    sf_history['val_loss'].append(va_loss)
    sf_history['train_acc'].append(tr_acc)
    sf_history['val_acc'].append(va_acc)

    print(f'{epoch:5d} {tr_loss:9.4f} {va_loss:9.4f} {tr_acc:8.3f} {va_acc:8.3f}')

    if va_loss < sf_best_val:
        sf_best_val   = va_loss
        sf_best_state = {k: v.cpu().clone() for k, v in sf_model.state_dict().items()}
        sf_patience   = 0
    else:
        sf_patience += 1
        if sf_patience >= PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

sf_model.load_state_dict(sf_best_state)
print('SF-only model trained.')

# Evaluate
sf_model.eval()
sf_probs, sf_preds, sf_true = [], [], []
with torch.no_grad():
    for x, lengths, y in sf_te_loader:
        x, lengths = x.to(DEVICE), lengths.to(DEVICE)
        logits = sf_model(x, lengths)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs >= 0.5).astype(int)
        sf_probs.extend(probs)
        sf_preds.extend(preds)
        sf_true.extend(y.numpy().astype(int))

sf_probs = np.array(sf_probs)
sf_preds = np.array(sf_preds)
sf_true  = np.array(sf_true)

sf_results = {
    'name':   'Control (SF only)',
    'probs':  sf_probs,
    'preds':  sf_preds,
    'true':   sf_true,
    'acc':    accuracy_score(sf_true, sf_preds),
    'prec':   precision_score(sf_true, sf_preds),
    'rec':    recall_score(sf_true, sf_preds),
    'f1':     f1_score(sf_true, sf_preds),
    'auc':    roc_auc_score(sf_true, sf_probs),
    'history': sf_history,
}

print()
print('='*45)
print(f'  SF-only Accuracy : {sf_results[\"acc\"]:.4f}')
print(f'  Precision        : {sf_results[\"prec\"]:.4f}')
print(f'  Recall           : {sf_results[\"rec\"]:.4f}')
print(f'  F1 Score         : {sf_results[\"f1\"]:.4f}')
print(f'  ROC-AUC          : {sf_results[\"auc\"]:.4f}')
print('='*45)
"""]
    },
    # Comparison plots
    {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": ["""# -- Side-by-side model comparison --------------------------------------------
models = [full_results, sf_results]
colors = ['#3498db', '#e74c3c']

fig = plt.figure(figsize=(18, 12))
gs  = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# -- TOP LEFT: Metric bar chart ------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
metrics   = ['acc', 'prec', 'rec', 'f1', 'auc']
met_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(metrics))
w = 0.35
for i, (res, col) in enumerate(zip(models, colors)):
    vals = [res[m] for m in metrics]
    bars = ax1.bar(x + i*w, vals, width=w, color=col, alpha=0.85, label=res['name'])
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
ax1.set_xticks(x + w/2)
ax1.set_xticklabels(met_names, fontsize=9)
ax1.set_ylim(0, 1.12)
ax1.set_title('Metric Comparison', fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_ylabel('Score')

# -- TOP MIDDLE: ROC curves overlaid ------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
for res, col in zip(models, colors):
    fpr, tpr, _ = roc_curve(res['true'], res['probs'])
    ax2.plot(fpr, tpr, color=col, lw=2, label=f'{res[\"name\"]} (AUC={res[\"auc\"]:.3f})')
ax2.plot([0,1],[0,1],'k--', alpha=0.4)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves', fontweight='bold')
ax2.legend(fontsize=8)

# -- TOP RIGHT: Training loss curves ------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
for res, col in zip(models, colors):
    ax3.plot(res['history']['val_loss'], color=col, lw=2, label=res['name'])
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Validation Loss')
ax3.set_title('Validation Loss Curves', fontweight='bold')
ax3.legend(fontsize=8)

# -- BOTTOM LEFT: Confusion matrix - Full model -------------------------------
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(full_results['true'], full_results['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Clean','Cheat'], yticklabels=['Clean','Cheat'])
ax4.set_title(f'Full Model (SF + Maia)\\nAcc={full_results[\"acc\"]:.1%}', fontweight='bold', color='#3498db')
ax4.set_xlabel('Predicted'); ax4.set_ylabel('Actual')

# -- BOTTOM MIDDLE: Confusion matrix - SF only --------------------------------
ax5 = fig.add_subplot(gs[1, 1])
cm = confusion_matrix(sf_results['true'], sf_results['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax5,
            xticklabels=['Clean','Cheat'], yticklabels=['Clean','Cheat'])
ax5.set_title(f'Control (SF only)\\nAcc={sf_results[\"acc\"]:.1%}', fontweight='bold', color='#e74c3c')
ax5.set_xlabel('Predicted'); ax5.set_ylabel('Actual')

# -- BOTTOM RIGHT: Delta summary text -----------------------------------------
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
delta_lines = ['Maia contribution (Full - SF only):\\n']
for m, name in zip(metrics, met_names):
    delta = full_results[m] - sf_results[m]
    sign  = '+' if delta >= 0 else ''
    delta_lines.append(f'  {name:<12}: {sign}{delta:.4f}\\n')
delta_lines.append('\\n')
delta_lines.append('Positive = Maia features help\\n')
delta_lines.append('Negative = SF alone was better\\n')
summary = ''.join(delta_lines)
ax6.text(0.05, 0.92, summary, transform=ax6.transAxes,
         fontsize=11, va='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

plt.suptitle('Full Model (SF + Maia) vs Control (SF only) -- Head-to-Head', fontsize=14, fontweight='bold')
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Comparison saved to model_comparison.png')
"""]
    },
]

# Attach new cells at position 38 (before old Results Summary)
insert_at = 38
for i, cell in enumerate(new_cells):
    cells.insert(insert_at + i, cell)

# Update the old Results Summary markdown (now shifted by 3)
old_summary_idx = insert_at + len(new_cells)
cells[old_summary_idx]['source'] = [
    "## 6 - Results Summary\n",
    "\n",
    "See the comparison plot above for the full metric breakdown.\n",
    "\n",
    "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n",
    "|-------|----------|-----------|--------|----|---------|\n",
    "| Full (SF + Maia) | see output | see output | see output | see output | see output |\n",
    "| Control (SF only) | see output | see output | see output | see output | see output |\n",
    "\n",
    "**Interpretation:** If the full model outperforms the SF-only control, Maia features carry independent signal. "
    "If they perform similarly, Stockfish features alone explain most of the detectable variance."
]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Done. Notebook now has {len(cells)} cells.')
