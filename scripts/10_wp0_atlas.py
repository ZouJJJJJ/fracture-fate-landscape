# -*- coding: utf-8 -*-
"""WP0: annotate GSE185940 trunk (Tam four-gate daily) into cell states using canonical
markers; produce cluster table, lineage scores, composition dynamics and 2D plot coords."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, gc
BASE = ''+ROOT+''
# (third-party dependencies are installed via requirements.txt)
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans

d = np.load(f'{BASE}/data/gse185940/gse185940_lognorm.npz')
X = sparse.csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
if tuple(d['shape'])[1] == 39551: X = X.T.tocsr()
genes = pd.read_csv(f'{BASE}/data/gse185940/genelist.csv', header=None)[0].astype(str).values
gix = {g: i for i, g in enumerate(genes)}
meta = pd.read_csv(f'{BASE}/data/gse185940/cell_meta.csv', keep_default_na=False)

# trunk = Tam daily, grouped order matching the trained latent
keep_t = meta['treatment'].eq('Tamoxifen').values
tps = ['E11.5', 'E12.5', 'E13.5', 'E14.5']
_parts, _st = [], []
for k, tp in enumerate(tps):
    idx = np.where(keep_t & meta['timepoint'].eq(tp).values)[0]
    _parts.append(idx); _st.append(np.full(len(idx), k + 1))
keep = np.concatenate(_parts); stage = np.concatenate(_st)
Xc = X[keep]; del X; gc.collect()
print('trunk cells', Xc.shape)

# HVG + PCA (reuse simple)
mu = np.asarray(Xc.mean(0)).ravel()
sq = Xc.copy(); sq.data **= 2
var = np.asarray(sq.mean(0)).ravel() - mu ** 2; del sq
hvg = np.argsort(var)[::-1][:1800]
D = Xc[:, hvg].toarray().astype(np.float64)
sd = D.std(0); sd[sd == 0] = 1
D = (D - D.mean(0)) / sd
from sklearn.decomposition import PCA
Z = PCA(n_components=20, random_state=0).fit_transform(D)
del D; gc.collect()

K = 14
lab = KMeans(K, n_init=15, random_state=0).fit_predict(Z)

marker_sets = {
 'mesenchyme': ['Prrx1','Prrx2','Msx1','Msx2','Pitx1','Meis2','Pbx1'],
 'condensation': ['Sox9','Sox5','Sox6','Nkx3-2','Fgfr3'],
 'chondrocyte': ['Col2a1','Col9a1','Col9a2','Col11a1','Acan','Hapln1','Matn1'],
 'fibrous': ['Col1a1','Col1a2','Col3a1','Dcn','Lum','Postn','Fn1','Col5a1'],
 'osteogenic': ['Runx2','Sp7','Spp1','Ibsp','Col10a1'],
 'proliferation': ['Mki67','Top2a','Cdk1','Ccnb1','Cenpa'],
 'tendon': ['Scx','Tnmd','Mkx','Thbs2'],
 'muscle': ['Myod1','Myog','Myh3','Tnnt3','Actc1'],
 'endothelial': ['Pecam1','Kdr','Cdh5','Emcn','Kdr'],
 'immune': ['Ptprc','Lyz2','C1qa','Cd52'],
}
# per-cluster mean lognorm expression for markers
rows = []
gscore = {name: np.zeros(K) for name in marker_sets}
present = {name: [g for g in gs if g in gix] for name, gs in marker_sets.items()}
for c in range(K):
    cells = np.where(lab == c)[0]
    sub = Xc[cells]
    r = {'cluster': c, 'n': len(cells)}
    for name, gs in marker_sets.items():
        use = present[name]
        cols = [gix[g] for g in use]
        m = np.asarray(sub[:, cols].mean(0)).ravel().mean() if cols else 0.0
        gscore[name][c] = m
        r[name] = round(float(m), 3)
    rows.append(r)
tab = pd.DataFrame(rows)

# assign state label using cluster-RELATIVE (z-score) lineage activation
lineage_names = ['mesenchyme','chondrocyte','fibrous','tendon','muscle','endothelial','immune']
raw = {n: gscore[n] for n in lineage_names}
zrel = {n: (raw[n] - raw[n].mean()) / (raw[n].std() + 1e-9) for n in lineage_names}
def label_for(c):
    if all(raw[n][c] < 0.04 for n in lineage_names): return 'other'
    dom = max(lineage_names, key=lambda n: zrel[n][c])
    prol = gscore['proliferation'][c]; cond = gscore['condensation'][c]
    if dom == 'chondrocyte':
        return 'proliferative_chondro' if prol > 0.30 else 'committed_chondro'
    if dom == 'fibrous': return 'fibrous_perichondrium'
    if dom == 'mesenchyme':
        return 'mesenchyme_condensation' if (cond > 0.12 or zrel['mesenchyme'][c] > 0.5) else 'limb_mesenchyme'
    return {'tendon': 'tendon', 'muscle': 'muscle', 'endothelial': 'endothelial', 'immune': 'immune'}[dom]
state = np.array([label_for(c) for c in range(K)])
tab['state'] = state
print(tab[['cluster','n','mesenchyme','condensation','chondrocyte','fibrous','proliferation','tendon','muscle','endothelial','immune','state']].to_string(index=False))

cell_state = state[lab]
# composition per stage
comp = pd.crosstab(stage, cell_state, normalize='index').round(4)
print('\ncomposition:\n', comp)

os.makedirs(f'{BASE}/wp0_out', exist_ok=True)
tab.to_csv(f'{BASE}/wp0_out/cluster_table.csv', index=False)
comp.to_csv(f'{BASE}/wp0_out/composition.csv')
np.savez(f'{BASE}/wp0_out/atlas.npz', Z2=Z[:, :2], Z20=Z, cluster=lab,
         stage=stage, state=cell_state, keep=keep)
print('WP0 saved')
