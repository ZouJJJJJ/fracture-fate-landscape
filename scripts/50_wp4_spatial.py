# -*- coding: utf-8 -*-
"""WP4 spatial in-situ validation (GSE218046): control C1 vs NF1 pseudarthrosis D1.
- Load Visium filtered h5, library-normalize/log1p; tissue positions for hex geometry.
- Per-spot fate modules + BMP & MAPK pathway readout scores; spatial maps.
- Global Moran's I (hex adjacency) for spatial structure; niche classification.
- Fibrocartilage-bone junction niche: compare BMP / MAPK control vs NF1 (expect
  NF1 = BMP-low, MAPK-high), with permutation/CI.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, h5py, json, gzip, sys
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = ''+ROOT+''
EX = f'{BASE}/data/gse218046/extracted'
slices = {'Control': 'GSM6733377', 'NF1': 'GSM6733378'}

def load_slice(gsm):
    with h5py.File(f'{EX}/{gsm}_filtered_feature_bc_matrix.h5', 'r') as f:
        m = f['matrix']
        barcodes = np.array([b.decode() for b in m['barcodes']])
        genes = np.array([b.decode() for b in m['features/name']])
        nspots, ngenes = len(barcodes), len(genes)
        X = sparse.csr_matrix((m['data'][:], m['indices'][:], m['indptr'][:]),
                              shape=(nspots, ngenes))  # already spots x genes
    # positions: barcode,in_tissue,array_row,array_col,pxl_row,pxl_col
    with gzip.open(f'{EX}/{gsm}_tissue_positions_list.csv.gz', 'rt') as f:
        pos = pd.read_csv(f, header=None)
    pos.columns = ['barcode', 'in_tissue', 'arr_row', 'arr_col', 'px_row', 'px_col']
    pos = pos.set_index('barcode').loc[barcodes].reset_index()
    # normalize
    lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
    Xn = X.multiply(1e4 / lib[:, None]); Xn.data = np.log1p(Xn.data)
    return Xn.tocsr(), genes, pos

modules = {
    'prog': ['Prrx1', 'Pdgfra', 'Cxcl12', 'Lepr', 'Postn'],
    'chondro': ['Sox9', 'Col2a1', 'Acan', 'Col11a1', 'Sox5'],
    'fibrous': ['Col1a1', 'Col3a1', 'Dcn', 'Lum', 'Col1a2'],
    'hyp': ['Col10a1', 'Runx2', 'Mmp13', 'Ibsp'],
    'osteo': ['Sp7', 'Spp1', 'Alpl', 'Bglap'],
}
pathways = {
    'BMP': ['Bmp2', 'Bmp4', 'Bmpr1a', 'Smad1', 'Smad5', 'Id1', 'Id2', 'Id3'],
    'MAPK': ['Map2k1', 'Mapk1', 'Mapk3', 'Dusp1', 'Dusp6', 'Egr1', 'Fos'],
    'CDK8': ['Cdk8'],
    'inter_med': ['Col14a1', 'Pdzrn4'],
}
def score(X, genes, names):
    gpos = {g: i for i, g in enumerate(genes)}
    cols = [gpos[g] for g in names if g in gpos]
    if not cols: return np.zeros(X.shape[0]), []
    return np.asarray(X[:, cols].mean(1)).ravel(), cols

data = {}
def safe_corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return (x * y).sum() / den if den > 0 else 0.0
for label, gsm in slices.items():
    X, genes, pos = load_slice(gsm)
    in_t = pos.in_tissue.values.astype(bool)
    sc = {}
    for name, names in {**modules, **pathways}.items():
        sc[name], used = score(X, genes, names)
    Sc = pd.DataFrame(sc)
    data[label] = {'X': X, 'genes': genes, 'pos': pos, 'Sc': Sc, 'in_t': in_t}
    print(label, 'spots', X.shape[0], 'in tissue', in_t.sum())

# within-slice z-score all module/pathway scores (in-tissue) -> comparable relative enrichment
for label in data:
    d = data[label]; k = d['in_t']; Sc = d['Sc']; Scz = Sc.copy()
    for c in Sc.columns:
        v = Sc[c].values[k]; mu, sd = v.mean(), v.std() + 1e-9
        Scz[c] = (Sc[c].values - mu) / sd
    d['Scz'] = Scz

# ---- Moran's I on hex array coords (in-tissue spots) ----
def morans_i(vals, coords):
    nn = NearestNeighbors(radius=1.6).fit(coords)
    g = nn.radius_neighbors_graph(coords).tocsr()
    g.setdiag(0); g.eliminate_zeros()
    x = vals - vals.mean()
    xs = (x ** 2).sum()
    num = (g.multiply(x[None, :]).multiply(x[:, None])).sum()
    W = g.nnz
    n = len(vals)
    return (n / W) * num / xs if W > 0 and xs > 0 else np.nan
moran = {}
for label in data:
    d = data[label]; k = d['in_t']
    coords = d['pos'][['arr_row', 'arr_col']].values[k]
    moran[label] = {m: morans_i(d['Sc'][m].values[k], coords)
                    for m in ['chondro', 'fibrous', 'hyp', 'BMP', 'MAPK']}
moran = pd.DataFrame(moran).round(3)
print("Moran's I:"); print(moran)

# ---- niche classification: relative (within-slice z) module enrichment ----
mod_cols = list(modules)
for label in data:
    d = data[label]
    d['niche'] = np.array(mod_cols)[np.argmax(d['Scz'][mod_cols].values, 1)]

# ---- fibrocartilage-bone junction: high fibrous AND high chondro/hyp ----
def junction_mask(Sc):
    fz = Sc.fibrous > np.percentile(Sc.fibrous, 60)
    cz = (Sc.chondro + Sc.hyp) > np.percentile(Sc.chondro + Sc.hyp, 60)
    return (fz & cz).values
junc = {label: junction_mask(data[label]['Sc']) & data[label]['in_t']
        for label in data}
jtab = []
for label in data:
    k = junc[label]; Scz = data[label]['Scz']; Sc = data[label]['Sc']
    # within-slice junction vs non-junction enrichment (z units)
    rest = data[label]['in_t'] & ~k
    jtab.append({'slice': label, 'n_junction': int(k.sum()),
                 'BMP_z': Scz.BMP[k].mean(), 'MAPK_z': Scz.MAPK[k].mean(),
                 'CDK8_z': Scz.CDK8[k].mean(), 'inter_z': Scz.inter_med[k].mean(),
                 'MAPK_junc_enrich': Scz.MAPK[k].mean() - Scz.MAPK[rest].mean(),
                 'BMP_junc_enrich': Scz.BMP[k].mean() - Scz.BMP[rest].mean(),
                 'corr_BMP_MAPK': safe_corr(Sc.BMP.values[data[label]['in_t']],
                                            Sc.MAPK.values[data[label]['in_t']])})
jtab = pd.DataFrame(jtab).set_index('slice')
print('junction pathway z-means & within-slice enrichment:'); print(jtab.round(3))

# permutation test for NF1 vs control junction (on within-slice z scores)
rng = np.random.default_rng(7)
def perm(col):
    a = data['NF1']['Scz'][col].values[junc['NF1']]
    b = data['Control']['Scz'][col].values[junc['Control']]
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a); diffs = []
    for _ in range(1000):
        p = rng.permutation(pool)
        diffs.append(p[:na].mean() - p[na:].mean())
    diffs = np.array(diffs)
    pval = (np.abs(diffs) >= abs(obs)).mean()
    return obs, pval
for col in ['BMP', 'MAPK', 'CDK8', 'inter_med']:
    obs, p = perm(col)
    print(f'junction {col}: NF1-Control z-delta {round(obs,3)} p={round(p,4)}')

# ---- spatial maps ----
plot_mods = ['chondro', 'fibrous', 'hyp', 'BMP', 'MAPK', 'CDK8']
fig, axes = plt.subplots(2, len(plot_mods), figsize=(3.0 * len(plot_mods), 6.4))
for r, label in enumerate(['Control', 'NF1']):
    d = data[label]; pos = d['pos']; Sc = d['Sc']
    x, y = pos.px_col.values, -pos.px_row.values
    for c, m in enumerate(plot_mods):
        ax = axes[r, c]; k = d['in_t']
        v = Sc[m].values
        sc = ax.scatter(x[k], y[k], c=v[k], s=12, cmap='viridis')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(m)
        if c == 0: ax.set_ylabel(label, fontsize=12)
plt.suptitle('Spatial module & pathway maps (top Control, bottom NF1)', y=1.02)
plt.tight_layout(); plt.savefig(f'{BASE}/wp4_out/fig_spatial_maps.png', dpi=140,
                                bbox_inches='tight')

# niche composition comparison
niche_comp = pd.DataFrame({label: data[label]['Sc'].assign(
    niche=data[label]['niche']).groupby('niche').size()
    for label in data}).fillna(0)
niche_comp = niche_comp / niche_comp.sum()
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(mod_cols)); w = 0.38
ax.bar(x - w/2, niche_comp.reindex(mod_cols)['Control'].fillna(0), w, label='Control')
ax.bar(x + w/2, niche_comp.reindex(mod_cols)['NF1'].fillna(0), w, label='NF1')
ax.set_xticks(x, mod_cols); ax.set_title('Spatial niche composition'); ax.legend()
plt.tight_layout(); plt.savefig(f'{BASE}/wp4_out/fig_niche.png', dpi=140)

# ---- save tables ----
moran.to_csv(f'{BASE}/wp4_out/morans_I.csv')
jtab.to_csv(f'{BASE}/wp4_out/junction_pathways.csv')
# per-spot scores + coords for supplementary spatial figures
spot = {}
for label in data:
    d = data[label]; pos = d['pos']
    spot[label] = np.column_stack([pos.px_col.values, pos.px_row.values,
                                   d['in_t'].astype(float),
                                   d['Sc'][['prog','chondro','fibrous','hyp','osteo',
                                            'BMP','MAPK','CDK8','inter_med']].values])
np.savez(f'{BASE}/wp4_out/spot_scores.npz',
         cols=np.array(['x','y','in_t','prog','chondro','fibrous','hyp','osteo',
                        'BMP','MAPK','CDK8','inter_med']),
         Control=spot['Control'], NF1=spot['NF1'])
print('WP4 saved')
