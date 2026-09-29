# -*- coding: utf-8 -*-
"""Build low-dimensional coordinate npz for DiffusionOT from GSE185940.
Trunk = Tamoxifen four-gate daily E11.5->E14.5 (uniform protocol, balanced gates).
Output npz keys: pca_scaled (n x d, float32), time_label (str '1'..'4'), type_label (str).
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, gc
from scipy import sparse
from sklearn.decomposition import PCA
import sys, os

BASE = ''+ROOT+''
n_per_tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1200   # per-timepoint subsample (smoke)
n_pc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
n_hvg = 1800
rng = np.random.RandomState(0)

# --- reconstruct CSR (cells x genes) ---
d = np.load(f'{BASE}/data/gse185940/gse185940_lognorm.npz')
shape = tuple(d['shape'])
X = sparse.csr_matrix((d['data'], d['indices'], d['indptr']), shape=shape)
if shape[1] == 39551:      # genes x cells -> transpose
    X = X.T.tocsr()
print('matrix cells x genes =', X.shape)
genes = pd.read_csv(f'{BASE}/data/gse185940/genelist.csv', header=None)[0].astype(str).values
meta = pd.read_csv(f'{BASE}/data/gse185940/cell_meta.csv', keep_default_na=False)
assert X.shape[0] == len(meta)

# --- subset Tam four-gate daily ---
tps = ['E11.5', 'E12.5', 'E13.5', 'E14.5']
keep_t = meta['treatment'].eq('Tamoxifen')
sel_idx, sel_tp, sel_type = [], [], []
for k, tp in enumerate(tps):
    idx = np.where(keep_t.values & meta['timepoint'].eq(tp).values)[0]
    if n_per_tp and len(idx) > n_per_tp:
        idx = rng.choice(idx, size=n_per_tp, replace=False)
    sel_idx.append(idx)
    sel_tp.append(np.full(len(idx), str(k + 1)))
    sel_type.append(meta['selection'].values[idx])
sel_idx = np.concatenate(sel_idx)
time_label = np.concatenate(sel_tp).astype(object)
type_label = np.concatenate(sel_type).astype(object)
print('selected cells:', len(sel_idx), 'per tp:', [int((time_label == str(k + 1)).sum()) for k in range(4)])

Xs = X[sel_idx]
del X; gc.collect()

# --- HVG by per-gene variance on lognorm (sparse) ---
# mean and E[x^2] per gene
mu = np.asarray(Xs.mean(0)).ravel()
sq = Xs.copy(); sq.data **= 2
var = np.asarray(sq.mean(0)).ravel() - mu ** 2
del sq; gc.collect()
hvg = np.argsort(var)[::-1][:n_hvg]
print('HVG var range:', var[hvg].min().round(3), var[hvg].max().round(3))

# --- densify, standardize genes, PCA ---
D = Xs[:, hvg].toarray().astype(np.float64)
del Xs; gc.collect()
mu2 = D.mean(0); sd = D.std(0); sd[sd == 0] = 1
D = (D - mu2) / sd
p = PCA(n_components=n_pc, random_state=0)
Z = p.fit_transform(D)
print('PC explained var ratio:', p.explained_variance_ratio_.round(4))
# scale each PC to unit std (mimic pca_scaled)
Z = Z / Z.std(0)
Z = Z.astype(np.float32)
print('coord shape', Z.shape, 'std', Z.std(0).round(3))

out = f'{BASE}/data/gse185940/gse185940_latent_smoke.npz' if n_per_tp <= 2000 else f'{BASE}/data/gse185940/gse185940_latent_full.npz'
np.savez(out, pca_scaled=Z, time_label=time_label, type_label=type_label)
print('saved', out)
