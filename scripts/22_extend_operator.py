# -*- coding: utf-8 -*-
"""Extend the PCA operator's gene-level interface with a curated skeletal/fracture panel
(TFs, signaling & literature targets excluded by variance HVG). Each extra gene is
standardized over the selected cells and regressed onto the 8 unit-std PC scores; the
regression coefficients serve as its linear encode/decode rows. Does NOT retrain / change
the latent coordinates. Saves pca_operator_ext.npz.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, gc, sys
from scipy import sparse

BASE = ''+ROOT+''
d = np.load(f'{BASE}/data/gse185940/gse185940_lognorm.npz', allow_pickle=True)
shape = tuple(d['shape'])
M = sparse.csr_matrix((d['data'], d['indices'], d['indptr']), shape=shape)  # cells x genes
gc.collect()
allgenes = pd.read_csv(f'{BASE}/data/gse185940/genelist.csv', header=None)[0].astype(str).values[:shape[1]]
meta = pd.read_csv(f'{BASE}/data/gse185940/cell_meta.csv', keep_default_na=False)

# grouped selection identical to make_latent
keep_t = meta['treatment'].eq('Tamoxifen').values
tps = ['E11.5', 'E12.5', 'E13.5', 'E14.5']
parts = []
for tp in tps:
    parts.append(np.where(keep_t & meta['timepoint'].eq(tp).values)[0])
sel = np.concatenate(parts)
Xs = M[sel]
del M; gc.collect()
print('selected', Xs.shape)

op = np.load(f'{BASE}/wp1_out/pca_operator.npz', allow_pickle=True)
Z = op['latent'].astype(np.float64)
Jenc = op['Jenc'].astype(np.float64); Jdec = op['Jdec'].astype(np.float64)
hvg_idx = op['hvg_idx']; hvg_genes = list(op['hvg_genes'])
mean0 = op['mean']; sd0 = op['sd']

curated = [
    'Prrx1','Prrx2','Pdgfra','Pdgfrb','Cxcl12','Lepr','Postn','Pi16','Ly6a','Scx','Mkx',
    'Twist1','Twist2','Msx1','Msx2','Hoxa11','Hoxd13',
    'Sox9','Sox5','Sox6','Col2a1','Acan','Col11a1','Col9a1','Col9a2',
    'Col1a1','Col1a2','Col3a1','Dcn','Lum','Lgals1','Col14a1','Pdzrn4','Col5a1','Col6a1',
    'Fbln1','Fbln2','Acta2',
    'Col10a1','Runx2','Mmp13','Mmp9','Ibsp',
    'Sp7','Spp1','Alpl','Bglap','Dlx5','Dlx6',
    'Cdk8','Inhba','Inhbb','Bmp2','Bmp4','Bmp7','Bmpr1a','Bmpr1b','Bmpr2',
    'Acvr1','Acvr1l','Acvr2a','Acvr2b','Smad1','Smad4','Smad5','Smad9',
    'Nog','Chrd','Grem1','Fst','Fstl1',
    'Map2k1','Map2k2','Mapk1','Mapk3','Ptpn11','Nf1','Kras',
    'Cd47','Thbs1','Sparc','Clec11a','Itga11',
    'Wnt10b','Wnt5a','Ctnnb1','Axin2','Lef1','Tcf7',
    'Mki67','Top2a','Cdk1','Pcna','Mcm2','Ccnb1','Ccnd1',
    'Sost','Dkk1','Dkk3',
]
gpos = {g: i for i, g in enumerate(allgenes)}
hvg_set = set(hvg_idx)
extra_genes, extra_idx, beta_rows, emean, esd = [], [], [], [], []
ZtZ_inv = np.linalg.inv(Z.T @ Z / len(Z))
for g in curated:
    if g not in gpos: continue
    gi = gpos[g]
    if gi in hvg_set: continue
    col = np.asarray(Xs[:, gi].todense()).ravel()
    mu = col.mean(); sd = col.std()
    if sd < 1e-8: continue
    x = (col - mu) / sd
    beta = ZtZ_inv @ (Z.T @ x / len(Z))  # x ~ Z beta
    extra_genes.append(g); extra_idx.append(gi); beta_rows.append(beta)
    emean.append(mu); esd.append(sd)
beta_rows = np.array(beta_rows)
print('extra genes projected:', len(extra_genes))

# concatenate: Jenc rows = HVG then extra; Jdec cols likewise
Jenc_ext = np.vstack([Jenc, beta_rows])
Jdec_ext = np.hstack([Jdec, beta_rows.T])
genes_ext = np.array(hvg_genes + extra_genes)
mean_ext = np.concatenate([mean0, np.array(emean)])
sd_ext = np.concatenate([sd0, np.array(esd)])
is_extra = np.concatenate([np.zeros(len(hvg_genes), bool), np.ones(len(extra_genes), bool)])
np.savez(f'{BASE}/wp1_out/pca_operator_ext.npz',
         Jenc=Jenc_ext.astype(np.float32), Jdec=Jdec_ext.astype(np.float32),
         genes=genes_ext, mean=mean_ext.astype(np.float32), sd=sd_ext.astype(np.float32),
         is_extra=is_extra, latent=Z.astype(np.float32))
print('saved pca_operator_ext.npz; total genes', len(genes_ext))
