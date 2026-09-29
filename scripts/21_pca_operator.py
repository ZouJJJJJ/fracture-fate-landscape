# -*- coding: utf-8 -*-
"""Recompute the exact linear PCA operator of the GSE185940 trunk used in training,
with encode/decode and Jacobians; verify it reproduces the saved latent."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, gc
BASE = ''+ROOT+''
# (third-party dependencies are installed via requirements.txt)
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

d = np.load(f'{BASE}/data/gse185940/gse185940_lognorm.npz')
X = sparse.csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
if tuple(d['shape'])[1] == 39551: X = X.T.tocsr()
genes = pd.read_csv(f'{BASE}/data/gse185940/genelist.csv', header=None)[0].astype(str).values
meta = pd.read_csv(f'{BASE}/data/gse185940/cell_meta.csv', keep_default_na=False)
# trunk = Tam daily, SAME grouped selection/order as make_latent (E11.5 then E12.5 ...)
keep_t = meta['treatment'].eq('Tamoxifen').values
tps = ['E11.5', 'E12.5', 'E13.5', 'E14.5']
sel_parts, stage_parts = [], []
for k, tp in enumerate(tps):
    idx = np.where(keep_t & meta['timepoint'].eq(tp).values)[0]
    sel_parts.append(idx); stage_parts.append(np.full(len(idx), k + 1))
keep = np.concatenate(sel_parts)
stage = np.concatenate(stage_parts)
Xc = X[keep]; del X; gc.collect()

mu0 = np.asarray(Xc.mean(0)).ravel()
sq = Xc.copy(); sq.data **= 2
var0 = np.asarray(sq.mean(0)).ravel() - mu0 ** 2; del sq
hvg_idx = np.argsort(var0)[::-1][:1800]
D = Xc[:, hvg_idx].toarray().astype(np.float64)
mean = D.mean(0); sd = D.std(0); sd[sd == 0] = 1
Ds = (D - mean) / sd
pca = PCA(n_components=8, random_state=0).fit(Ds)
V = pca.components_.T  # HVG x L
pc = Ds @ V
pcstd = pc.std(0)
latent = pc / pcstd

saved = np.load(f'{BASE}/data/gse185940/gse185940_latent_full.npz')
maxerr = np.abs(latent - saved['pca_scaled']).max()
print('re-encode max abs err vs saved latent:', maxerr)
assert maxerr < 1e-2  # residual is float32 rounding of the saved latent

# 2D view of the latent
pca2 = PCA(n_components=2, random_state=0).fit(latent)
Q = pca2.components_.T  # L x 2
coords2 = latent @ Q

# operator Jacobians
# encode: z = ((x-mean)/sd @ V)/pcstd  -> dz/dx[i,j] = V[i,j]/pcstd[j]/sd[i]
Jenc = V / pcstd[None, :] / sd[:, None]                 # HVG x L
# decode least-norm: x = (z*pcstd @ V.T)*sd + mean -> dx/dz[j,i] = V[i,j]*sd[i]*pcstd[j]
Jdec = (V * sd[:, None] * pcstd[None, :]).T             # L x HVG

os.makedirs(f'{BASE}/wp1_out', exist_ok=True)
np.savez(f'{BASE}/wp1_out/pca_operator.npz',
         hvg_idx=hvg_idx, hvg_genes=genes[hvg_idx], mean=mean, sd=sd, V=V,
         pcstd=pcstd, Q=Q, coords2=coords2, latent=latent, keep=keep, stage=stage,
         Jenc=Jenc, Jdec=Jdec)
print('operator saved; latent', latent.shape, '2d', coords2.shape)
print('2d view explained:', pca2.explained_variance_ratio_)
