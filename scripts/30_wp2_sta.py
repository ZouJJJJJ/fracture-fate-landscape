# -*- coding: utf-8 -*-
"""WP2 Stochastic Transition Analysis (fate probabilities).
Fate transport = empirical entropic OT (Waddington-OT style Sinkhorn) between adjacent
daily clouds in latent space -- robust to the neural drift's near-identity regularization.
The neural DiffusionOT model supplies the continuous landscape / attractors / score.
Outputs: forward fate probabilities (E11.5->E14.5), backward ancestry, terminal module
extension (hypertrophic/osteogenic via probability layer).
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = ''+ROOT+''
stages = ['1', '2', '3', '4']; stage_names = ['E11.5', 'E12.5', 'E13.5', 'E14.5']
labs = ['P', 'F', 'C', 'O']; lab_names = {'P': 'progenitor', 'F': 'fibrous',
                                          'C': 'chondrocyte', 'O': 'other'}
li = {l: i for i, l in enumerate(labs)}

op = np.load(f'{BASE}/wp1_out/pca_operator.npz', allow_pickle=True)
latent = op['latent']
tl = np.load(f'{BASE}/data/gse185940/gse185940_latent_full.npz',
             allow_pickle=True)['time_label'].astype(str)
at = np.load(f'{BASE}/wp0_out/atlas.npz', allow_pickle=True); cs = at['state']
fmap = {'mesenchyme_condensation': 'P', 'limb_mesenchyme': 'P',
        'fibrous_perichondrium': 'F', 'tendon': 'F',
        'committed_chondro': 'C', 'proliferative_chondro': 'C', 'immune': 'O'}
cls = np.array([fmap.get(s, 'O') for s in cs])
rng = np.random.default_rng(0)
idx = {s: np.where(tl == s)[0] for s in stages}
NS = 700
S = {s: rng.choice(idx[s], min(NS, len(idx[s])), replace=False) for s in stages}

# ---- entropic OT coupling (Sinkhorn) between adjacent stages ----
def sinkhorn(X, Y, eps=0.08, iters=200):
    C = ((X[:, None, :] - Y[None]) ** 2).sum(-1); C = C / C.mean()
    K = np.exp(-C / eps)
    a = np.ones(len(X)) / len(X); b = np.ones(len(Y)) / len(Y)
    u = a.copy(); v = b.copy()
    for _ in range(iters):
        u = a / (K @ v + 1e-30); v = b / (K.T @ u + 1e-30)
    return u[:, None] * K * v[None]

class_mass = []   # raw class-to-class mass per interval (4x4)
for a, b in zip(stages[:-1], stages[1:]):
    P = sinkhorn(latent[S[a]], latent[S[b]])
    ca, cb = cls[S[a]], cls[S[b]]
    T = np.zeros((4, 4))
    for i, la in enumerate(ca):
        Pi = P[i]
        for j, lb in enumerate(cb):
            T[li[la], li[lb]] += Pi[j]
    class_mass.append(T)

# forward row-normalized transitions; compose
Tf = [T / T.sum(1, keepdims=True) for T in class_mass]
Ff = Tf[0]
for T in Tf[1:]:
    Ff = Ff @ T
fate_df = pd.DataFrame(Ff, index=[lab_names[l] for l in labs],
                       columns=[lab_names[l] for l in labs])
print('forward fate probabilities (rows=E11.5 start class):')
print(fate_df.round(3))
start_comp = np.array([(cls[S['1']] == l).mean() for l in labs])
overall = start_comp @ Ff
print('overall E11.5->E14.5 fate:', {lab_names[l]: round(float(overall[i]), 3)
      for i, l in enumerate(labs)})

# backward ancestry: column-normalized coupling, compose from stage4 to stage1
Tb = [T / T.sum(0, keepdims=True) for T in class_mass]  # rows=source, cols=target
anc_rows = []
for tc in labs:  # terminal class at stage4
    v = np.zeros(4); v[li[tc]] = 1
    for T in reversed(Tb):   # v over target -> source = T @ v
        v = T @ v
    anc_rows.append(v)
anc = pd.DataFrame(np.array(anc_rows), index=[lab_names[l] for l in labs],
                   columns=[lab_names[l] for l in labs])
print('backward ancestry (rows=E14.5 fate, cols=E11.5 ancestor):')
print(anc.round(3))

# ---- terminal module extension (probability layer) ----
# normal day7 fracture SSPC hypertrophic/osteogenic potential from module scores
ms = np.load(f'{BASE}/data/gse234451/module_scores.npz', allow_pickle=True)
Smod, mnames, tp, sspc = ms['S'], list(ms['mnames']), ms['tp'], ms['sspc']
sel = sspc & (tp == 'day7')
if sel.sum() < 50:
    sel = sspc
# terminal modules (exclude the transient chondro program for the terminal decision)
term = ['prog_SSPC', 'fibrous', 'hypertrophic', 'osteo']
tci = [mnames.index(t) for t in term]
Zt = Smod[tci][:, sel].T
lab_t = np.array(term)[np.argmax(Zt, 1)]
mix = pd.Series(lab_t).value_counts(normalize=True)
print('normal day7 SSPC terminal module mix:'); print(mix.round(3))
# extend geometric fates: chondrocyte endpoint -> split into hypertrophic/osteogenic
hyp = float(mix.get('hypertrophic', 0)); ost = float(mix.get('osteo', 0))
ss = hyp + ost
remain_chondro = max(0.0, 1 - ss) if ss > 0 else 1.0
ext_fate = {'progenitor': float(overall[li['P']]), 'fibrous': float(overall[li['F']]),
            'other': float(overall[li['O']]),
            'chondrocyte': float(overall[li['C']]) * remain_chondro,
            'hypertrophic': float(overall[li['C']]) * hyp,
            'osteoblast': float(overall[li['C']]) * ost}
print('extended terminal fate probabilities (probability layer):')
for k, v in ext_fate.items():
    print(' ', k, round(v, 3))

# ---- save ----
fate_df.to_csv(f'{BASE}/wp2_out/forward_fate.csv')
anc.to_csv(f'{BASE}/wp2_out/backward_ancestry.csv')
pd.Series(ext_fate).to_csv(f'{BASE}/wp2_out/extended_fate.csv')
np.savez(f'{BASE}/wp2_out/transitions.npz', class_mass=np.array(class_mass),
         start_comp=start_comp, overall=overall, S=np.array([S[s] for s in stages],
         dtype=object))

# ---- figures ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
# forward fate stacked bars by start class
bottom = np.zeros(4)
colors = {'progenitor': '#9ecae1', 'fibrous': '#fdae6b',
          'chondrocyte': '#74c476', 'other': '#bdbdbd'}
for col in ['progenitor', 'fibrous', 'chondrocyte', 'other']:
    vals = fate_df[col].values
    axes[0].bar(fate_df.index, vals, bottom=bottom, label=col, color=colors[col])
    bottom += vals
axes[0].set_title('Forward fate by E11.5 start class'); axes[0].legend(fontsize=8)
axes[0].tick_params(axis='x', rotation=30)
# backward ancestry heatmap
im = axes[1].imshow(anc.values, cmap='Blues', aspect='auto', vmin=0, vmax=1)
axes[1].set_xticks(range(4), anc.columns, rotation=30, ha='right'); axes[1].set_yticks(range(4), anc.index)
axes[1].set_title('Backward ancestry (E14.5 fate -> E11.5 ancestor)')
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, round(anc.values[i, j], 2), ha='center', va='center', fontsize=8)
# extended terminal fate
ek = ['progenitor', 'fibrous', 'chondrocyte', 'hypertrophic', 'osteoblast', 'other']
ev = [ext_fate[k] for k in ek]
axes[2].bar(ek, ev, color='#6baed6')
axes[2].set_title('Extended terminal fate (probability layer)')
axes[2].tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.savefig(f'{BASE}/wp2_out/fig_sta.png', dpi=140)
print('WP2 STA saved')
