# -*- coding: utf-8 -*-
"""WP2 cross-condition fate redistribution & nonunion block metrics (GSE292578).
Three conditions (WT Intact, WT Ischemic, CD47-KO) x day4/day7 stromal cells.
Fate states from the same module signatures; quantify failure-to-commit:
  commitment = chondro+hypertrophic+osteo ; arrest = prog ; terminal = hyp+osteo ;
  fibrotic = fibrous ; ischemic intermediate = Col14a1+/Pdzrn4+.
Block score = standardized deviation from WT Intact day7 (bootstrap CIs).
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, gc
from scipy import sparse
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = ''+ROOT+''
z = np.load(f'{BASE}/data/gse292578/built/stromal_lognorm.npz', allow_pickle=True)
shape = tuple(z['shape'])
X = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=shape)
genes = pd.read_csv(f'{BASE}/data/gse292578/built/genes.csv')['gene'].astype(str).values
gidx = {g: i for i, g in enumerate(genes)}
meta = pd.read_csv(f'{BASE}/data/gse292578/built/stromal_meta.csv')
print('stromal', X.shape)

modules = {
    'prog_SSPC': ['Prrx1', 'Pdgfra', 'Cxcl12', 'Lepr', 'Postn', 'Pdgfrb'],
    'chondro': ['Sox9', 'Col2a1', 'Acan', 'Col11a1', 'Sox5', 'Sox6'],
    'fibrous': ['Col1a1', 'Col3a1', 'Dcn', 'Lum', 'Lgals1', 'Col1a2'],
    'hypertrophic': ['Col10a1', 'Runx2', 'Mmp13', 'Ibsp', 'Mmp9'],
    'osteo': ['Sp7', 'Spp1', 'Alpl', 'Bglap'],
    'proliferation': ['Mki67', 'Top2a', 'Cdk1', 'Pcna', 'Mcm2'],
}
def col_score(names):
    cols = [gidx[g] for g in names if g in gidx]
    if not cols: return np.zeros(X.shape[0])
    return np.asarray(X[:, cols].mean(1)).ravel()
S = np.column_stack([col_score(v) for v in modules.values()])
mnames = list(modules.keys())
# per-cell background = mean over a broad housekeeping-ish set (use overall module mean)
bg = S.mean(1, keepdims=True)
Sa = S - bg
fate_lab = np.array(['prog_SSPC', 'chondro', 'fibrous', 'hypertrophic', 'osteo'])[
    np.argmax(Sa[:, [mnames.index(m) for m in ['prog_SSPC', 'chondro', 'fibrous',
                                               'hypertrophic', 'osteo']]], 1)]
# ischemic fibroblast-like intermediate gate
def gate(g):
    return np.asarray(X[:, gidx[g]].toarray()).ravel() if g in gidx else np.zeros(X.shape[0])
inter = ((gate('Col14a1') > 0.3) & (gate('Pdzrn4') > 0.2)).astype(float)
prolif = S[:, mnames.index('proliferation')]

df = meta.copy(); df['fate'] = fate_lab; df['inter'] = inter; df['prolif'] = prolif

# ---- composition by condition x day ----
order = [('WT_Intact', 'day7'), ('WT_Ischemic', 'day7'), ('CD47_KO', 'day7'),
         ('WT_Intact', 'day4'), ('WT_Ischemic', 'day4'), ('CD47_KO', 'day4')]
fates = ['prog_SSPC', 'fibrous', 'chondro', 'hypertrophic', 'osteo']
comp = {}
for cond, day in order:
    sub = df[(df.condition == cond) & (df.day == day)]
    if len(sub) == 0: continue
    comp[f'{cond}|{day}'] = (sub.fate.value_counts(normalize=True)
                             .reindex(fates).fillna(0).values)
comp = pd.DataFrame(comp, index=fates).T
print('fate composition (rows condition|day):'); print(comp.round(3))

# ---- block metrics ----
def metrics(sub):
    n = len(sub); c = sub.fate.value_counts(normalize=True)
    chondro = c.get('chondro', 0); hyp = c.get('hypertrophic', 0); ost = c.get('osteo', 0)
    prog = c.get('prog_SSPC', 0); fib = c.get('fibrous', 0)
    return pd.Series({
        'commitment': chondro + hyp + ost,
        'arrest_prog': prog,
        'terminal': hyp + ost,
        'fibrotic': fib,
        'inter_ischemic': sub.inter.mean(),
        'proliferation': sub.prolif.mean(),
        'n': n})
M = pd.DataFrame({f'{c}|{d}': metrics(df[(df.condition == c) & (df.day == d)])
                  for c, d in order if len(df[(df.condition == c) & (df.day == d)])})
print('block metrics:'); print(M.round(3))

# bootstrap CIs for day7 key metrics
def boot(cond, day, B=200):
    sub = df[(df.condition == cond) & (df.day == day)].reset_index(drop=True)
    n = len(sub); rows = []
    rng = np.random.default_rng(1)
    for _ in range(B):
        s = sub.iloc[rng.integers(0, n, n)]; rows.append(metrics(s).drop('n'))
    return pd.DataFrame(rows)
ci = {}
for c in ['WT_Intact', 'WT_Ischemic', 'CD47_KO']:
    b = boot(c, 'day7')
    ci[c] = pd.DataFrame({'lo': b.quantile(0.025), 'hi': b.quantile(0.975)})
print('bootstrap 95% CI (day7) available')

# ---- block score vs normal (WT Intact day7): standardized adverse deviation ----
key = ['commitment', 'terminal', 'proliferation']   # lower is worse
adv = ['arrest_prog', 'fibrotic', 'inter_ischemic']  # higher is worse
ref = M['WT_Intact|day7']
sdref = M.std(1) + 1e-6
block = {}
for col in M.columns:
    if not col.endswith('day7'): continue
    z = 0.0
    for k in key:
        z += (ref[k] - M[col][k]) / sdref[k]
    for k in adv:
        z += (M[col][k] - ref[k]) / sdref[k]
    block[col] = z / (len(key) + len(adv))
block = pd.Series(block).sort_values(ascending=False)
print('nonunion block score (higher = more blocked, vs WT Intact day7):')
print(block.round(3))

# ---- save ----
comp.to_csv(f'{BASE}/wp2_out/condition_composition.csv')
M.to_csv(f'{BASE}/wp2_out/condition_metrics.csv')
block.to_csv(f'{BASE}/wp2_out/block_score.csv')
for c, d in ci.items():
    d.to_csv(f'{BASE}/wp2_out/ci_{c}.csv')

# ---- figures ----
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
day7cols = [c for c in comp.index if c.endswith('day7')]
bottom = np.zeros(len(day7cols))
cmap = {'prog_SSPC': '#9ecae1', 'fibrous': '#fdae6b', 'chondro': '#74c476',
        'hypertrophic': '#bc80bd', 'osteo': '#66c2a5'}
for f in fates:
    v = comp.loc[day7cols, f].values
    axes[0].bar([c.replace('|day7', '') for c in day7cols], v, bottom=bottom,
                label=f, color=cmap[f]); bottom += v
axes[0].set_title('Day7 stromal fate composition'); axes[0].legend(fontsize=8)
axes[0].tick_params(axis='x', rotation=20)
mk = ['commitment', 'arrest_prog', 'terminal', 'fibrotic', 'inter_ischemic', 'proliferation']
Md = M[[c for c in M.columns if c.endswith('day7')]].loc[mk]
im = axes[1].imshow(Md.values, cmap='RdYlBu_r', aspect='auto')
axes[1].set_xticks(range(Md.shape[1]), [c.replace('|day7', '') for c in Md.columns],
                   rotation=20, ha='right')
axes[1].set_yticks(range(len(mk)), mk)
for i in range(Md.shape[0]):
    for j in range(Md.shape[1]):
        axes[1].text(j, i, round(Md.values[i, j], 2), ha='center', va='center', fontsize=8)
axes[1].set_title('Day7 nonunion block metrics')
plt.tight_layout(); plt.savefig(f'{BASE}/wp2_out/fig_conditions.png', dpi=140)
print('WP2 conditions saved')
