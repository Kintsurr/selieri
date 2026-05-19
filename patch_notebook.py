import json

path = r'C:\Users\Zura\Desktop\Selieri\selieri_cheating_detection.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# -- Cell 11: markdown header -------------------------------------------------
nb['cells'][11]['source'] = [
    "### 3.2 SF vs Maia 2D Quadrant Analysis\n",
    "\n",
    "The central hypothesis in 2D space: every move lives at a point (SF_Rank, LC0_Rank).\n",
    "Cheater moves cluster in the **top-left quadrant** - low SF rank (engine best) and high LC0 rank (Maia disagrees).\n",
    "Human moves cluster in the **bottom-right** - Maia agrees, Stockfish does not find them optimal.\n",
    "We then split by population to compare Stockfish vs Maia in isolation on each group."
]

# -- Cell 12: 2D quadrant scatter + occupancy bar -----------------------------
nb['cells'][12]['source'] = ["""CPL_CLIP = 300
moves_df['SF_D20_CPL_c']  = moves_df['SF_D20_CPL'].clip(0, CPL_CLIP)
moves_df['LC0_D20_CPL_c'] = moves_df['LC0_D20_CPL'].clip(0, CPL_CLIP)

sf_med  = moves_df['SF_D20_Rank'].median()
lc0_med = moves_df['LC0_D20_Rank'].median()

cheat_moves = moves_df[moves_df['move_label'] == 1]
human_moves = moves_df[moves_df['move_label'] == 0]

sample_cheat = cheat_moves.dropna(subset=['SF_D20_Rank','LC0_D20_Rank']).sample(min(3000, len(cheat_moves)), random_state=42)
sample_human = human_moves.dropna(subset=['SF_D20_Rank','LC0_D20_Rank']).sample(min(3000, len(human_moves)), random_state=42)

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(sample_human['SF_D20_Rank'], sample_human['LC0_D20_Rank'],
            alpha=0.18, s=8, c='#2ecc71', label='Human move')
ax1.scatter(sample_cheat['SF_D20_Rank'], sample_cheat['LC0_D20_Rank'],
            alpha=0.35, s=8, c='#e74c3c', label='Cheat move')

ax1.axvline(sf_med,  color='white', lw=1.2, ls='--', alpha=0.6)
ax1.axhline(lc0_med, color='white', lw=1.2, ls='--', alpha=0.6)

quadrant_labels = [
    (0.25, 0.78, 'LOW SF / HIGH LC0\\n(Cheater Zone)', '#e74c3c'),
    (0.75, 0.78, 'HIGH SF / HIGH LC0\\n(Both agree bad)', '#aaaaaa'),
    (0.25, 0.22, 'LOW SF / LOW LC0\\n(Both agree good)', '#aaaaaa'),
    (0.75, 0.22, 'HIGH SF / LOW LC0\\n(Human Zone)', '#2ecc71'),
]
for (xp, yp, txt, col) in quadrant_labels:
    ax1.text(xp, yp, txt, transform=ax1.transAxes,
             ha='center', va='center', fontsize=8.5, color=col,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', alpha=0.7))

ax1.set_xlabel('Stockfish Rank (1 = engine best move)')
ax1.set_ylabel('Maia/LC0 Rank')
ax1.set_xlim(0.5, 10.5)
ax1.set_ylim(0.5, 10.5)
ax1.set_title('2D Quadrant: SF Rank vs Maia Rank', fontweight='bold')
ax1.legend(markerscale=3, loc='upper right')

ax2 = fig.add_subplot(1, 2, 2)

def quadrant_pct(df):
    sf  = df['SF_D20_Rank']
    lc0 = df['LC0_D20_Rank']
    return [
        ((sf <= sf_med) & (lc0 > lc0_med)).mean(),   # low SF, high LC0 = cheat zone
        ((sf >  sf_med) & (lc0 <= lc0_med)).mean(),  # high SF, low LC0 = human zone
        ((sf <= sf_med) & (lc0 <= lc0_med)).mean(),  # both good
        ((sf >  sf_med) & (lc0 >  lc0_med)).mean(),  # both bad
    ]

q_labels = [
    'Low SF + High LC0\\n(Cheater Zone)',
    'High SF + Low LC0\\n(Human Zone)',
    'Low SF + Low LC0\\n(Both Good)',
    'High SF + High LC0\\n(Both Bad)',
]
q_cheat_pct = quadrant_pct(cheat_moves.dropna(subset=['SF_D20_Rank','LC0_D20_Rank']))
q_human_pct = quadrant_pct(human_moves.dropna(subset=['SF_D20_Rank','LC0_D20_Rank']))

x = list(range(len(q_labels)))
w = 0.35
ax2.bar([i - w/2 for i in x], q_cheat_pct, width=w, color='#e74c3c', alpha=0.85, label='Cheat moves')
ax2.bar([i + w/2 for i in x], q_human_pct, width=w, color='#2ecc71', alpha=0.85, label='Human moves')
ax2.set_xticks(x)
ax2.set_xticklabels(q_labels, fontsize=8)
ax2.set_ylabel('Proportion of moves in quadrant')
ax2.set_title('Quadrant Occupancy: Cheat vs Human', fontweight='bold')
ax2.legend()
ax2.set_ylim(0, 0.5)

plt.suptitle('Stockfish vs Maia 2D Analysis -- The Cheater Zone', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Cheat moves in Cheater Zone (low SF, high LC0): {q_cheat_pct[0]:.1%}")
print(f"Human moves in Human Zone   (high SF, low LC0): {q_human_pct[1]:.1%}")
"""]

# -- Cell 13: explicit SF vs Maia split by population -------------------------
nb['cells'][13]['source'] = ["""from scipy.stats import mannwhitneyu

cheat_sf  = cheat_moves['SF_D20_Rank'].dropna().clip(1, 10)
human_sf  = human_moves['SF_D20_Rank'].dropna().clip(1, 10)
cheat_lc0 = cheat_moves['LC0_D20_Rank'].dropna().clip(1, 10)
human_lc0 = human_moves['LC0_D20_Rank'].dropna().clip(1, 10)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# TOP LEFT: Stockfish rank - cheaters vs humans
ax = axes[0][0]
for data, label, color, offset in [(cheat_sf,'Cheat','#e74c3c',0), (human_sf,'Human','#2ecc71',0.4)]:
    counts = data.value_counts().sort_index()
    ax.bar(counts.index + offset, counts / counts.sum(), width=0.4, alpha=0.8, color=color, label=label)
_, p = mannwhitneyu(cheat_sf, human_sf, alternative='less')
ax.set_title(f'Stockfish Rank: Cheat vs Human  (p={p:.2e})', fontweight='bold')
ax.set_xlabel('Rank (1 = engine picks this move)')
ax.set_ylabel('Proportion')
ax.set_xticks(range(1, 11))
ax.legend()
ax.text(0.62, 0.82,
        f'Cheat Rank-1: {(cheat_sf==1).mean():.1%}\\nHuman Rank-1: {(human_sf==1).mean():.1%}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8))

# TOP RIGHT: Maia rank - cheaters vs humans
ax = axes[0][1]
for data, label, color, offset in [(cheat_lc0,'Cheat','#e74c3c',0), (human_lc0,'Human','#2ecc71',0.4)]:
    counts = data.value_counts().sort_index()
    ax.bar(counts.index + offset, counts / counts.sum(), width=0.4, alpha=0.8, color=color, label=label)
_, p = mannwhitneyu(human_lc0, cheat_lc0, alternative='less')
ax.set_title(f'Maia/LC0 Rank: Cheat vs Human  (p={p:.2e})', fontweight='bold')
ax.set_xlabel('Rank')
ax.set_ylabel('Proportion')
ax.set_xticks(range(1, 11))
ax.legend()
ax.text(0.62, 0.82,
        f'Cheat Rank-1: {(cheat_lc0==1).mean():.1%}\\nHuman Rank-1: {(human_lc0==1).mean():.1%}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8))

# BOTTOM LEFT: CLEAN GAMES ONLY - SF vs Maia CPL
ax = axes[1][0]
clean_game_ids = df[df['phase'] == 'clean']['game_id'].values
clean_moves = moves_df[moves_df['game_id'].isin(clean_game_ids)]
ax.hist(clean_moves['SF_D20_CPL_c'].dropna(),  bins=50, alpha=0.65, color='#e74c3c', density=True, label='Stockfish CPL')
ax.hist(clean_moves['LC0_D20_CPL_c'].dropna(), bins=50, alpha=0.65, color='#3498db', density=True, label='Maia/LC0 CPL')
ax.set_title('CLEAN GAMES ONLY: SF vs Maia CPL', fontweight='bold', color='#2ecc71')
ax.set_xlabel('CPL (clipped 300)')
ax.set_ylabel('Density')
ax.legend()
sf_m = clean_moves['SF_D20_CPL_c'].mean()
lc_m = clean_moves['LC0_D20_CPL_c'].mean()
ax.text(0.52, 0.80,
        f'SF mean CPL:   {sf_m:.1f}\\nMaia mean CPL: {lc_m:.1f}\\nMaia advantage: {sf_m - lc_m:+.1f}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8))

# BOTTOM RIGHT: CHEAT GAMES ONLY - SF vs Maia CPL
ax = axes[1][1]
cheat_game_ids = df[df['phase'] == 'cheat']['game_id'].values
cheat_game_moves = moves_df[moves_df['game_id'].isin(cheat_game_ids)]
ax.hist(cheat_game_moves['SF_D20_CPL_c'].dropna(),  bins=50, alpha=0.65, color='#e74c3c', density=True, label='Stockfish CPL')
ax.hist(cheat_game_moves['LC0_D20_CPL_c'].dropna(), bins=50, alpha=0.65, color='#3498db', density=True, label='Maia/LC0 CPL')
ax.set_title('CHEAT GAMES ONLY: SF vs Maia CPL', fontweight='bold', color='#e74c3c')
ax.set_xlabel('CPL (clipped 300)')
ax.set_ylabel('Density')
ax.legend()
sf_m = cheat_game_moves['SF_D20_CPL_c'].mean()
lc_m = cheat_game_moves['LC0_D20_CPL_c'].mean()
ax.text(0.52, 0.80,
        f'SF mean CPL:   {sf_m:.1f}\\nMaia mean CPL: {lc_m:.1f}\\nSF advantage: {lc_m - sf_m:+.1f}',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#222', alpha=0.8))

plt.suptitle('Engine-vs-Engine Comparison Split by Population', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\nKey takeaway:")
print("  Clean games -> Maia should have LOWER CPL (agrees more with human moves)")
print("  Cheat games -> Stockfish should have LOWER CPL (agrees more with engine moves)")
"""]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook patched successfully.')
