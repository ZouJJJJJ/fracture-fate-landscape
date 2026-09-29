# -*- coding: utf-8 -*-
"""WP0 full pipeline under Plan B (main line = GSE185940 Tam daily E11.5-E14.5).
Adds QC, Scrublet-lite doublet score, one-vs-rest markers, lineage module scores,
unified stage mapping, differential-abundance tests, and two WP0 figures.
Reuses cluster/state from wp0_out/atlas.npz to stay consistent with WP1-WP4.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys, os, gc
BASE = ''+ROOT+''
# (third-party dependencies are installed via requirements.txt)
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

OUTD = f'{BASE}/wp0_out'; os.makedirs(OUTD, exist_ok=True)
tps = ['E11.5','E12.5','E13.5','E14.5']

# ---- rebuild identical trunk selection ----
d = np.load(f'{BASE}/data/gse185940/gse185940_lognorm.npz')
X = sparse.csr_matrix((d['data'], d['indices'], d['indptr']), shape=tuple(d['shape']))
if tuple(d['shape'])[1] == 39551: X = X.T.tocsr()
genes = pd.read_csv(f'{BASE}/data/gse185940/genelist.csv', header=0).iloc[:,0].astype(str).values
gix = {g: i for i, g in enumerate(genes)}
meta = pd.read_csv(f'{BASE}/data/gse185940/cell_meta.csv', keep_default_na=False)
keep_t = meta['treatment'].eq('Tamoxifen').values
_parts, _st = [], []
for k, tp in enumerate(tps):
    idx = np.where(keep_t & meta['timepoint'].eq(tp).values)[0]
    _parts.append(idx); _st.append(np.full(len(idx), k + 1))
keep = np.concatenate(_parts); stage = np.concatenate(_st)
Xc = X[keep].tocsr(); del X; gc.collect()
N = Xc.shape[0]; print('trunk', Xc.shape)

# ---- reuse clustering / annotation / 20 PCs ----
atlas = np.load(f'{OUTD}/atlas.npz', allow_pickle=True)
lab = atlas['cluster']; cell_state = atlas['state']; Z20 = atlas['Z20']; Z2 = atlas['Z2']
state_order = ['mesenchyme_condensation','committed_chondro','fibrous_perichondrium','tendon','immune']

# ============ QC ============
# nCount: raw-count proxy via expm1; fractions computed on lognorm (raw counts not archived)
Xraw = Xc.copy(); Xraw.data = np.expm1(Xraw.data)
nFeature = Xc.getnnz(axis=1)
nCount = np.asarray(Xraw.sum(1)).ravel()
cell_tot = np.asarray(Xc.sum(1)).ravel()
mito_cols = np.array([g.startswith('mt-') for g in genes])
ribo_cols = np.array([g.startswith(('Rpl','Rps')) for g in genes])
mito = np.asarray(Xc[:, mito_cols].sum(1)).ravel() / (cell_tot + 1e-9)
ribo = np.asarray(Xc[:, ribo_cols].sum(1)).ravel() / (cell_tot + 1e-9)
del Xraw; gc.collect()

def mad(x): return np.median(np.abs(x - np.median(x)))
nf_lo, nf_hi = 200, np.median(nFeature) + 5*mad(nFeature)
mt_hi = np.median(mito) + 5*mad(mito)   # robust upper tail (data-driven; raw counts unavailable)
qc_flag = (nFeature < nf_lo) | (nFeature > nf_hi) | (mito > mt_hi)

# Scrublet-lite doublet score on 20-PC space
rng = np.random.default_rng(0)
n_syn = N // 5
a = rng.integers(0, N, n_syn); b = rng.integers(0, N, n_syn)
syn = (Z20[a] + Z20[b]) / 2.0
Zall = np.vstack([Z20, syn])
is_syn = np.r_[np.zeros(N, bool), np.ones(n_syn, bool)]
nn = NearestNeighbors(n_neighbors=21).fit(Zall)
nbr = nn.kneighbors(Z20, return_distance=False)[:, 1:]
doub_score = is_syn[nbr].mean(1)
dthr = np.median(doub_score) + 3*mad(doub_score)
doub_flag = doub_score > dthr

qc_cell = pd.DataFrame({'stage': stage, 'nFeature': nFeature, 'nCount': nCount,
                        'mito_frac': mito, 'ribo_frac': ribo,
                        'doublet_score': doub_score, 'qc_flag': qc_flag, 'doublet_flag': doub_flag})
qc_summary = qc_cell.groupby('stage').agg(
    n=('nFeature','size'),
    nFeature_median=('nFeature','median'), nFeature_q05=('nFeature', lambda x: np.percentile(x,5)),
    nFeature_q95=('nFeature', lambda x: np.percentile(x,95)),
    nCount_median=('nCount','median'),
    mito_pct_median=('mito_frac', lambda x: 100*np.median(x)),
    ribo_pct_median=('ribo_frac', lambda x: 100*np.median(x)),
    doublet_pct=('doublet_flag', lambda x: 100*x.mean()),
    qc_fail_pct=('qc_flag', lambda x: 100*x.mean())).round(2)
qc_summary.index = tps; qc_summary.to_csv(f'{OUTD}/qc_summary.csv')
print('qc fail %.2f%%, doublet %.2f%%' % (100*qc_flag.mean(), 100*doub_flag.mean()))

# ============ lineage module scores per state ============
modules = {
 'prog_SSPC': ['Prrx1','Pdgfra','Cxcl12','Lepr','Postn','Pdgfrb'],
 'chondro': ['Sox9','Col2a1','Acan','Col11a1','Sox5','Sox6'],
 'fibrous': ['Col1a1','Col3a1','Dcn','Lum','Lgals1','Col1a2'],
 'hypertrophic': ['Col10a1','Runx2','Mmp13','Ibsp','Mmp9'],
 'osteoblast': ['Sp7','Spp1','Alpl','Bglap'],
 'proliferation': ['Mki67','Top2a','Cdk1','Pcna','Mcm2'],
}
mod_score = pd.DataFrame(index=state_order, columns=list(modules), dtype=float)
for mn, gs in modules.items():
    cols = [gix[g] for g in gs if g in gix]
    if not cols: continue
    gene_mean = np.asarray(Xc[:, cols].mean(1)).ravel()
    for s in state_order:
        k = cell_state == s
        mod_score.loc[s, mn] = gene_mean[k].mean() if k.sum() else np.nan
mod_score = mod_score.round(3); mod_score.to_csv(f'{OUTD}/state_module_scores.csv')

# ============ one-vs-rest markers (z-test on sparse) ============
# global sum & sum-of-squares per gene
Gsum = np.asarray(Xc.sum(0)).ravel().astype(np.float64)
Xsq = Xc.copy(); Xsq.data **= 2
Gss = np.asarray(Xsq.sum(0)).ravel().astype(np.float64); del Xsq; gc.collect()
named = ~np.array([g.startswith(('mt-','ERCC','ENSMUST')) for g in genes])
rows = []
for s in state_order:
    kin = np.where(cell_state == s)[0]
    if len(kin) == 0: continue
    Xs = Xc[kin]
    n_in = len(kin); n_out = N - n_in
    s_in = np.asarray(Xs.sum(0)).ravel().astype(np.float64)
    ss_in = Xs.copy(); ss_in.data **= 2; ss_in = np.asarray(ss_in.sum(0)).ravel().astype(np.float64)
    s_out = Gsum - s_in; ss_out = Gss - ss_in
    mi = s_in / n_in; mo = s_out / n_out
    vi = ss_in/n_in - mi**2; vo = ss_out/n_out - mo**2
    vi = np.clip(vi, 1e-9, None); vo = np.clip(vo, 1e-9, None)
    z = (mi - mo) / np.sqrt(vi/n_in + vo/n_out)
    pct_in = np.array(Xs.getnnz(axis=0)).ravel() / n_in
    pct_out = (Xc.getnnz(axis=0) - np.array(Xs.getnnz(axis=0)).ravel()) / n_out
    z = np.nan_to_num(z, nan=0.0)
    p = 2*stats.norm.sf(np.abs(z))
    for gi_ in np.where(named)[0]:
        rows.append((s, genes[gi_], mi[gi_], mo[gi_], mi[gi_]-mo[gi_],
                     pct_in[gi_], pct_out[gi_], z[gi_], p[gi_]))
mk = pd.DataFrame(rows, columns=['state','gene','mean_in','mean_out','logFC',
                                 'pct_in','pct_out','z','p'])
mk['p_adj'] = np.clip(mk['p'] * len(mk), 0, 1)
mk = mk.sort_values(['state','z'], ascending=[True, False])
mk.round(4).to_csv(f'{OUTD}/markers_all.csv', index=False)
top = mk.groupby('state', group_keys=False).head(30)
top.round(4).to_csv(f'{OUTD}/markers_top.csv', index=False)
top10 = mk.groupby('state', group_keys=False).head(10)
print(top10.groupby('state').apply(lambda d: ', '.join(d.gene.head(6))).to_string())

# ============ unified stage mapping ============
bio_stage = ['limb mesenchyme / mesenchymal condensation',
             'mesenchymal condensation / chondrogenic commitment',
             'committed proliferative chondrocytes',
             'committed chondrocytes + fibrous perichondrium']
stage_map = pd.DataFrame({
    'dataset': ['GSE185940']*4, 'sampling_time': tps,
    'unified_biological_stage': bio_stage,
    'n_cells': [int((stage==k).sum()) for k in range(1,5)]})
stage_map.to_csv(f'{OUTD}/stage_mapping.csv', index=False)

# ============ differential abundance ============
counts = pd.crosstab(stage, cell_state).reindex(columns=state_order, fill_value=0)
chi2, chi_p, _, _ = stats.chi2_contingency(counts.values)
ab_rows = []
for s in state_order:
    prop = counts[s].values / counts.sum(1).values
    rho, rp = stats.spearmanr(np.arange(4), prop)
    ab_rows.append((s, *prop.round(4), round(rho,3), rp))
ab = pd.DataFrame(ab_rows, columns=['state']+tps+['spearman_rho','trend_p'])
ab.loc[len(ab)] = ['GLOBAL chi-square', chi2.round(2), chi_p, '', '', '', '']
ab.round(4).to_csv(f'{OUTD}/abundance_test.csv', index=False)
print('chi2 %.1f p=%.2e' % (chi2, chi_p))

# ============ figures ============
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
state_color = dict(zip(state_order, ['#1f77b4','#2ca02c','#ff7f0e','#9467bd','#d62728']))
# fig WP0 atlas
fig, ax = plt.subplots(2, 2, figsize=(12,10))
for k in range(4):
    m = stage == k+1
    ax[0,0].scatter(Z2[m,0], Z2[m,1], s=4, label=tps[k], alpha=.6)
ax[0,0].legend(markerscale=3); ax[0,0].set_title('A  Atlas by stage')
for s in state_order:
    m = cell_state == s
    ax[0,1].scatter(Z2[m,0], Z2[m,1], s=4, label=s, color=state_color[s], alpha=.6)
ax[0,1].legend(markerscale=3, fontsize=8); ax[0,1].set_title('B  Atlas by annotated state')
# C dotplot: rows = top6 markers per state, cols = states; color = per-gene relative enrichment
dot = top10.groupby('state', group_keys=False).head(6)
ygenes = list(dict.fromkeys(dot.gene.values))
piv = mk[mk.gene.isin(ygenes)].pivot_table(index='gene', columns='state', values='mean_in')
piv = piv.reindex(index=ygenes, columns=state_order)
pct_piv = mk[mk.gene.isin(ygenes)].pivot_table(index='gene', columns='state', values='pct_in').reindex(index=ygenes, columns=state_order)
enr = ((piv.sub(piv.min(1), axis=0)).div((piv.max(1)-piv.min(1)).replace(0,np.nan), axis=0))
for yi, g in enumerate(ygenes):
    for xi, s in enumerate(state_order):
        if np.isnan(enr.iloc[yi, xi]): continue
        pp = pct_piv.iloc[yi, xi]
        ax[1,0].scatter(xi, yi, s=60+300*(pp if not np.isnan(pp) else 0),
                        c=[enr.iloc[yi, xi]], cmap='magma', vmin=0, vmax=1,
                        edgecolors='gray', linewidths=.3)
ax[1,0].set_xticks(range(5), state_order, rotation=25, fontsize=8)
ax[1,0].set_yticks(range(len(ygenes)), ygenes, fontsize=6)
ax[1,0].invert_yaxis(); ax[1,0].set_title('C  State marker dot plot (size=pct, color=relative enrichment)')
# D composition stacked
comp = counts.div(counts.sum(1), axis=0).values
x = np.arange(4); bottom = np.zeros(4)
for j, s in enumerate(state_order):
    ax[1,1].bar(x, comp[:,j], bottom=bottom, color=state_color[s], label=s); bottom += comp[:,j]
ax[1,1].set_xticks(x, tps); ax[1,1].set_ylim(0,1)
ax[1,1].set_title('D  State composition'); ax[1,1].legend(fontsize=7)
for a in [ax[0,0],ax[0,1]]: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(f'{OUTD}/fig_wp0_atlas.png', dpi=140); plt.close()

# fig WP0 QC
fig, ax = plt.subplots(2, 2, figsize=(12,9))
data_box = [nFeature[stage==k] for k in range(1,5)]
ax[0,0].boxplot(data_box, showfliers=False); ax[0,0].set_xticks(range(1,5), tps)
ax[0,0].set_title('A  nFeature per cell by stage')
ax[0,1].boxplot([nCount[stage==k] for k in range(1,5)], showfliers=False)
ax[0,1].set_xticks(range(1,5), tps); ax[0,1].set_title('B  nCount (approx UMI) by stage')
w = .38; xx = np.arange(4)
ax[1,0].bar(xx-w/2, 100*np.array([mito[stage==k].mean() for k in range(1,5)]), w, label='mito %')
ax[1,0].bar(xx+w/2, 100*np.array([ribo[stage==k].mean() for k in range(1,5)]), w, label='ribo %')
ax[1,0].set_xticks(xx, tps); ax[1,0].set_title('C  Mitochondrial / ribosomal fraction'); ax[1,0].legend()
ax[1,1].hist(doub_score, bins=50, color='steelblue')
ax[1,1].axvline(dthr, color='red', ls='--', label=f'threshold={dthr:.2f}')
ax[1,1].set_title('D  Doublet score (Scrublet-lite)'); ax[1,1].legend()
plt.tight_layout(); plt.savefig(f'{OUTD}/fig_wp0_qc.png', dpi=140); plt.close()
print('WP0 full saved')
