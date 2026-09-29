# -*- coding: utf-8 -*-
"""WP1 fields + stochastic fate landscape:
diffusion D, velocity/growth/score in a 2D view, density potential L=-log p,
attractors (density minima of L), escape depth and pairwise barriers (minimax/Kruskal)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, types, gc
BASE = ''+ROOT+''
# (third-party dependencies are installed via requirements.txt)
sys.path.append(os.path.join(BASE, 'vendor/DiffusionOT'))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, pandas as pd
from sklearn.neighbors import KernelDensity
from scipy import ndimage
import utility as U
from utility import RUOT

torch.manual_seed(1)
device = torch.device('cpu')
op = np.load(f'{BASE}/wp1_out/pca_operator.npz', allow_pickle=True)
latent, coords2, Q = op['latent'], op['coords2'], op['Q']
full = np.load(f'{BASE}/data/gse185940/gse185940_latent_full.npz', allow_pickle=True)
tl = full['time_label'].astype(str)
stages = ['E11.5', 'E12.5', 'E13.5', 'E14.5']
data_train = [torch.from_numpy(latent[tl == str(k + 1)]).float() for k in range(4)]
integral_time = [0., 1., 2., 3.]; train_time = list(range(4))

func = RUOT(in_out_dim=8, hidden_dim=16, n_hiddens=3, activation='Tanh', d=0.001)
ck = torch.load(f'{BASE}/wp1_out/ckpt_final.pth', map_location='cpu')
func.load_state_dict(ck['func_state_dict']); func.eval()

args = types.SimpleNamespace()
# ---- diffusion ----
# ---- diffusion diagnostic (unexplained residual variance) ----
Dt0 = U.diffusion_fit(func, args, data_train, train_time, integral_time, device, time_tt=0.05).numpy()
Dt = Dt0.reshape(-1, Dt0.shape[-1]).mean(0)  # per-dimension mean over sampled paths
# forward SDE uses the model's own noise parameter d (fixed at 0.001 during training);
# the near-zero residual diagnostic means drift explains almost all cloud evolution.
D_scalar = float(func.d.detach())
print('per-dim residual D:', np.round(Dt, 6), 'SDE D (model d) =', round(D_scalar, 5))

# ---- 2D grid ----
ng = 120; pad = 0.6
xmin, xmax = coords2[:, 0].min() - pad, coords2[:, 0].max() + pad
ymin, ymax = coords2[:, 1].min() - pad, coords2[:, 1].max() + pad
gx = np.linspace(xmin, xmax, ng); gy = np.linspace(ymin, ymax, ng)
GX, GY = np.meshgrid(gx, gy)
grid2 = np.column_stack([GX.ravel(), GY.ravel()])
grid_lat = grid2 @ Q.T  # least-norm lift (Q orthonormal)

L_all, v2_all, g_all, score_all = [], [], [], []
zt = torch.from_numpy(grid_lat).float()
z0 = torch.zeros(zt.shape[0], 1); g0 = torch.zeros(zt.shape)
with torch.no_grad():
    for s, t in enumerate(integral_time):
        out = func(torch.tensor(t).float(), (zt, z0, z0, g0))
        v = out[0].numpy(); gg = out[1].numpy().ravel(); sc = out[3].numpy()
        v2 = v @ Q
        kde = KernelDensity(bandwidth=0.18, kernel='gaussian', rtol=1e-3)
        kde.fit(coords2[tl == str(s + 1)])
        logp = kde.score_samples(grid2)
        L = -logp
        L_all.append(L.reshape(ng, ng)); v2_all.append(v2); g_all.append(gg); score_all.append(sc)
        print('stage', stages[s], 'mean|v2|', round(np.linalg.norm(v2, axis=1).mean(), 3),
              'mean g', round(gg.mean(), 4))
L_all = np.array(L_all); v2_all = np.array(v2_all); g_all = np.array(g_all)

# ---- attractors: local minima of L ----
L_CAP = 4.0  # absolute ceiling: real density peaks L~1.5-2.5, empty/edge regions L~7+
def find_attractors(L):
    mp = ndimage.minimum_filter(L, size=9)
    cand = (L == mp)
    lab, n = ndimage.label(cand)
    pts = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        j = np.argmin(L[ys, xs]); pts.append((ys[j], xs[j], L[ys[j], xs[j]]))
    # only genuine, deep peaks: absolute cap and relative prominence
    thr = min(L_CAP, np.median(L) - 0.4)
    pts = [p for p in pts if p[2] < thr]
    pts.sort(key=lambda p: p[2])
    keep = []
    for p in pts:
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= 10 for q in keep): keep.append(p)
    return keep

# ---- minimax barrier via Kruskal union-find over grid graph ----
def build_edges(L):
    edges = []
    for r in range(ng):
        for c in range(ng):
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < ng and 0 <= c2 < ng:
                    w = max(L[r, c], L[r2, c2])
                    edges.append((w, r * ng + c, r2 * ng + c2))
    edges.sort(key=lambda e: e[0])
    return edges

class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb; return True
        return False

# WP0 states for attractor identity
atlas = np.load(f'{BASE}/wp0_out/atlas.npz', allow_pickle=True)
cell_state = atlas['state']
state_palette = ['#4C78A8', '#F58518', '#54A24B', '#B279A2', '#E45756',
                 '#72B7B2', '#9D755D', '#BAB0AC', '#FF9DA6']
state_names = sorted(np.unique(cell_state))
state_color = {s: state_palette[i % len(state_palette)] for i, s in enumerate(state_names)}

attractor_records = []
barrier_records = []
for s in range(4):
    L = L_all[s]
    pts = find_attractors(L)
    edges = build_edges(L)
    node_ids = [r * ng + c for r, c, _ in pts]
    # map each real cell to nearest grid node for basin labels
    cell_xy = np.column_stack([(coords2[tl == str(s + 1)][:, 0] - xmin) / (xmax - xmin) * (ng - 1),
                               (coords2[tl == str(s + 1)][:, 1] - ymin) / (ymax - ymin) * (ng - 1)]).astype(int)
    cell_node = np.clip(cell_xy[:, 1], 0, ng - 1) * ng + np.clip(cell_xy[:, 0], 0, ng - 1)
    states_here = cell_state[tl == str(s + 1)]
    # Kruskal, record pairwise connect threshold
    uf = UF(ng * ng)
    pair_barrier = {}
    target = {n: i for i, n in enumerate(node_ids)}
    connected_at = {}
    for w, a, b in edges:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb: continue
        # collect attractor roots before merge
        roots_att = {}
        for nid in node_ids:
            roots_att.setdefault(uf.find(nid), []).append(nid)
        uf.union(a, b)
        # check newly connected attractor pairs
        seen_root = {}
        for nid in node_ids:
            r = uf.find(nid)
            if r in seen_root:
                other = seen_root[r]
                key = tuple(sorted([target[nid], target[other]]))
                if key not in connected_at: connected_at[key] = w
            else:
                seen_root[r] = nid
        if len(connected_at) == len(node_ids) * (len(node_ids) - 1) // 2: break
    for i, (r, c, lv) in enumerate(pts):
        # escape depth = smallest barrier to another attractor
        others = [connected_at[tuple(sorted([i, j]))] for j in range(len(pts)) if j != i
                  and tuple(sorted([i, j])) in connected_at]
        depth = (min(others) - lv) if others else float('nan')
        # label by majority state of nearest 60 cells
        d2 = np.hypot(coords2[tl == str(s + 1)][:, 0] - gx[c],
                      coords2[tl == str(s + 1)][:, 1] - gy[r])
        near = np.argsort(d2)[:80]
        lab_state = pd.Series(states_here[near]).value_counts().idxmax()
        attractor_records.append({'stage': stages[s], 'id': i, 'x': gx[c], 'y': gy[r],
                                  'L': round(lv, 3), 'depth': round(float(depth), 3),
                                  'state': lab_state})
        for key, w in connected_at.items():
            if i in key:
                j = key[0] if key[1] == i else key[1]
                barrier_records.append({'stage': stages[s], 'a': i, 'b': j,
                                        'barrier': round(w - min(lv, pts[j][2]), 3)})
    print(stages[s], 'attractors:', [(p[2], a['state']) for p, a in
          zip(pts, [r for r in attractor_records if r['stage'] == stages[s]])])

att_df = pd.DataFrame(attractor_records)
bar_df = pd.DataFrame(barrier_records)
att_df.to_csv(f'{BASE}/wp1_out/attractors.csv', index=False)
bar_df.to_csv(f'{BASE}/wp1_out/barriers.csv', index=False)
np.savez(f'{BASE}/wp1_out/fields.npz', L=L_all, v2=v2_all, g=g_all, grid2=grid2,
         coords2=coords2, tl=tl, D_per=Dt, D_scalar=D_scalar, Q=Q,
         gx=gx, gy=gy)
print('fields saved')

# ============ figures ============
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for s, ax in enumerate(axes.ravel()):
    L = L_all[s]
    Lc = np.clip(L, np.percentile(L, 2), np.percentile(L, 98))
    im = ax.imshow(Lc, origin='lower', extent=[xmin, xmax, ymin, ymax],
                   cmap='terrain', aspect='auto', alpha=0.9)
    # coarse-grid velocity arrows
    cg = np.linspace(xmin, xmax, 15); cgy = np.linspace(ymin, ymax, 15)
    CGX, CGY = np.meshgrid(cg, cgy); cgrid = np.column_stack([CGX.ravel(), CGY.ravel()])
    clat = cgrid @ Q.T
    ct = torch.from_numpy(clat).float()
    cz0 = torch.zeros(ct.shape[0], 1); cg0 = torch.zeros(ct.shape)
    with torch.no_grad():
        cv = func(torch.tensor(float(s)).float(), (ct, cz0, cz0, cg0))[0].numpy() @ Q
    ax.quiver(cgrid[:, 0], cgrid[:, 1], cv[:, 0], cv[:, 1],
              color='k', scale=12, width=0.0025, alpha=0.65)
    sub = att_df[att_df.stage == stages[s]]
    for _, r in sub.iterrows():
        ax.scatter(r.x, r.y, s=130, marker='*', edgecolor='k',
                   color=state_color.get(r.state, 'k'), zorder=5)
        ax.annotate(str(int(r.id)), (r.x, r.y), textcoords='offset points',
                    xytext=(5, 5), fontsize=9, zorder=6)
    ax.set_title(f'{stages[s]}  fate landscape (L=-log p)')
handles = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=c,
                       markeredgecolor='k', markersize=13, label=st)
           for st, c in state_color.items()]
fig.legend(handles=handles, loc='lower center', ncol=min(len(state_color), 5), frameon=False)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(f'{BASE}/wp1_out/fig_landscape.png', dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
piv = att_df.pivot_table(index='stage', columns='id', values='depth', aggfunc='first')
piv.plot(kind='bar', ax=axes[0], width=0.8)
axes[0].set_title('Attractor escape depth'); axes[0].set_ylabel('depth (L units)')
axes[0].legend(title='attractor', fontsize=8)
axes[1].bar(range(1, 9), Dt, color='#4C78A8')
axes[1].axhline(D_scalar, color='#E45756', linestyle='--', label=f'scalar D={D_scalar:.4f}')
axes[1].set_title('Diffusion coefficient by latent dimension')
axes[1].set_xlabel('latent dimension'); axes[1].set_ylabel('D'); axes[1].legend()
fig.tight_layout(); fig.savefig(f'{BASE}/wp1_out/fig_depth_diffusion.png', dpi=180); plt.close(fig)
print('figures saved')
