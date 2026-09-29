# -*- coding: utf-8 -*-
"""Build GSE292578: QC filter, stromal-lineage selection, lognorm matrix + condition metadata."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, gc, gzip
BASE = ''+ROOT+''
# (third-party dependencies are installed via requirements.txt)
import numpy as np, pandas as pd
from scipy import sparse
from scipy.io import mmread

D = f'{BASE}/data/gse292578'
meta = pd.read_csv(f'{D}/GSE292578_all_unfiltered_cell_metadata.csv.gz')
genes = pd.read_csv(f'{D}/GSE292578_all_unfiltered_genes.csv.gz')
symbols = genes['gene_name'].astype(str).values
g2i = {g: i for i, g in enumerate(symbols)}

# parse condition/day
def parse(s):
    if 'Periosteum' in s:
        cond = 'CD47_KO' if s.startswith('CD47') else 'WT_Intact'
        return cond, 'Periosteum'
    cond = 'CD47_KO' if s.startswith('CD47') else ('WT_Ischemic' if 'Ischemic' in s else 'WT_Intact')
    day = 'day4' if 'Day_4' in s else 'day7'
    return cond, day
cd = meta['sample'].map(parse)
meta['condition'] = [x[0] for x in cd]; meta['day'] = [x[1] for x in cd]

keep = (meta['gene_count'] >= 400) & (meta['tscp_count'] <= 30000)
keep_idx = np.where(keep.values)[0]
print('QC pass cells:', len(keep_idx))
print(meta.loc[keep].groupby(['condition', 'day']).size())

X = mmread(gzip.open(f'{D}/GSE292578_all_unfiltered_DGE.mtx.gz')).tocsr()
print('full DGE', X.shape)
assert X.shape[0] == len(meta)
Xq = X[keep_idx].astype(np.float32)
del X; gc.collect()
print('QC matrix nnz', Xq.nnz)

# lineage gate
def cnt(gs):
    cols = [g2i[g] for g in gs if g in g2i]
    return np.asarray(Xq[:, cols].sum(1)).ravel() if cols else np.zeros(Xq.shape[0])
mes = cnt(['Col1a1','Col1a2','Col2a1','Col3a1','Pdgfra','Prrx1','Dcn','Lum','Acan','Postn'])
imm = cnt(['Ptprc','Lyz2','C1qa','C1qb','Cd3d','Cd3e','Cd14','S100a8','S100a9'])
end = cnt(['Pecam1','Kdr','Cdh5','Emcn'])
ery = cnt(['Hba-a1','Hbb-bs','Hba-a2'])
stromal = (mes >= 2) & (imm == 0) & (end == 0) & (ery < mes)
print('stromal cells:', stromal.sum())
print(pd.crosstab(meta.loc[keep_idx].condition, stromal))

Xs = Xq[stromal]
ms = meta.loc[keep_idx].reset_index(drop=True)
ms = ms.loc[stromal].reset_index(drop=True)
# lognorm
lib = np.asarray(Xs.sum(1)).ravel(); lib[lib == 0] = 1
Xn = Xs.multiply(1 / lib[:, None]) * 1e4; Xn.data = np.log1p(Xn.data)
Xn = Xn.tocsr().astype(np.float32)
os.makedirs(f'{BASE}/data/gse292578/built', exist_ok=True)
sparse.save_npz(f'{BASE}/data/gse292578/built/stromal_lognorm.npz', Xn)
ms[['condition', 'day', 'sample', 'gene_count', 'tscp_count']].to_csv(
    f'{BASE}/data/gse292578/built/stromal_meta.csv', index=False)
pd.DataFrame({'gene': symbols}).to_csv(f'{BASE}/data/gse292578/built/genes.csv', index=False)
print('saved stromal lognorm', Xn.shape, 'nnz', Xn.nnz)
print(ms.groupby(['condition', 'day']).size())
