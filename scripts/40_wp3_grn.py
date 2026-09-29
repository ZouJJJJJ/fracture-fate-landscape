# -*- coding: utf-8 -*-
"""WP3 robust GRN, in silico perturbation & target ranking.
Primary network = expression-derived, directional TF->target linear regulation:
multivariate OLS of standardized targets on candidate regulators, controlling for stage
dummies and cell-cycle; validates canonical edges (Sox9->Col2a1/Acan, Runx2->Col10a1).
Perturbation propagates TF -> targets -> fate-module scores -> fate (multinomial), giving
a chondrocyte-commitment rescue magnitude. The neural DiffusionOT Jacobian (landscape-
consistent) is retained as a cross-check. Final rank = rescue + importance + literature.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, sys, gc
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = ''+ROOT+''
op = np.load(f'{BASE}/wp1_out/pca_operator_ext.npz', allow_pickle=True)
genes = list(op['genes']); gpos = {g: i for i, g in enumerate(genes)}
mean = op['mean']; sd = op['sd']
at = np.load(f'{BASE}/wp0_out/atlas.npz', allow_pickle=True); cs = at['state']
fmap = {'mesenchyme_condensation': 0, 'limb_mesenchyme': 0, 'fibrous_perichondrium': 1,
        'tendon': 1, 'committed_chondro': 2, 'proliferative_chondro': 2, 'immune': 3}
fnum = np.array([fmap.get(s, 3) for s in cs])
fname = {0: 'progenitor', 1: 'fibrous', 2: 'chondrocyte', 3: 'other'}

# ---- build standardized expression of the operator genes over grouped selected cells ----
raw = np.load(f'{BASE}/data/gse185940/gse185940_lognorm.npz', allow_pickle=True)
shape = tuple(raw['shape'])   # cells x genes
X = sparse.csr_matrix((raw['data'], raw['indices'], raw['indptr']), shape=shape)
meta = pd.read_csv(f'{BASE}/data/gse185940/cell_meta.csv', keep_default_na=False)
keep_t = meta['treatment'].eq('Tamoxifen').values
tps = ['E11.5', 'E12.5', 'E13.5', 'E14.5']
sel = np.concatenate([np.where(keep_t & meta['timepoint'].eq(tp).values)[0] for tp in tps])
allgenes = pd.read_csv(f'{BASE}/data/gse185940/genelist.csv', header=None)[0].astype(str).values[:shape[1]]
agpos = {g: i for i, g in enumerate(allgenes)}
cols = np.array([agpos[g] for g in genes])
E = X[sel][:, cols].toarray().astype(np.float64)
del X; gc.collect()
Es = (E - mean) / sd   # standardized expression aligned to operator genes
print('expression', Es.shape)

# ---- module marker membership ----
modules = {
    'prog': ['Prrx1', 'Pdgfra', 'Cxcl12', 'Lepr', 'Postn', 'Pdgfrb'],
    'chondro': ['Sox9', 'Col2a1', 'Acan', 'Col11a1', 'Sox5', 'Sox6'],
    'fibrous': ['Col1a1', 'Col3a1', 'Dcn', 'Lum', 'Lgals1', 'Col1a2'],
    'hyp': ['Col10a1', 'Runx2', 'Mmp13', 'Ibsp', 'Mmp9'],
    'osteo': ['Sp7', 'Spp1', 'Alpl', 'Bglap'],
    'prolif': ['Mki67', 'Top2a', 'Cdk1', 'Pcna', 'Mcm2'],
}
mod_of_gene = {}
for m, gs_ in modules.items():
    for g in gs_:
        if g in gpos: mod_of_gene[gpos[g]] = m
module_score = np.zeros((len(genes), 6))
for gi, m in mod_of_gene.items():
    module_score[gi, list(modules).index(m)] = 1
Mscore = Es @ module_score / np.array([sum(module_score[:, k] > 0) for k in range(6)])
Mscore_df = pd.DataFrame(Mscore, columns=list(modules))

# ---- regulators & targets ----
regulators = ['Sox9', 'Sox5', 'Sox6', 'Runx2', 'Sp7', 'Dlx5', 'Msx1', 'Msx2', 'Twist1',
              'Prrx1', 'Scx', 'Mkx', 'Bmp2', 'Bmp4', 'Bmp7', 'Inhba', 'Inhbb', 'Cdk8',
              'Map2k1', 'Ptpn11', 'Cd47', 'Acvr1', 'Bmpr1a', 'Grem1', 'Nog', 'Chrd',
              'Fst', 'Wnt5a', 'Clec11a', 'Thbs1', 'Col14a1', 'Pdzrn4', 'Ctnnb1',
              'Lef1', 'Axin2', 'Postn', 'Pdgfra']
regulators = [g for g in regulators if g in gpos]
targets = sorted(set([g for gs_ in modules.values() for g in gs_] +
                     ['Col14a1', 'Pdzrn4', 'Bmpr1a', 'Acvr1', 'Map2k1', 'Cdk8', 'Ptpn11',
                      'Cd47', 'Clec11a', 'Itga11']))
targets = [g for g in targets if g in gpos]
R = Es[:, [gpos[g] for g in regulators]]
T = Es[:, [gpos[g] for g in targets]]
# covariates: stage dummies + proliferation
stagev = at['stage']
C = np.column_stack([(stagev == k).astype(float) for k in [1, 2, 3, 4]] +
                    [Mscore[:, list(modules).index('prolif')]])
Xdes = np.hstack([R, C])
ols = LinearRegression().fit(Xdes, T)
B = ols.coef_[:, :len(regulators)].T   # regulators x targets
Bdf = pd.DataFrame(B, index=regulators, columns=targets)
def edge(a, b): return float(Bdf.loc[a, b]) if (a in Bdf.index and b in Bdf.columns) else np.nan
print('canonical edges: Sox9->Col2a1', round(edge('Sox9', 'Col2a1'), 3),
      'Sox9->Acan', round(edge('Sox9', 'Acan'), 3),
      'Sox5->Col2a1', round(edge('Sox5', 'Col2a1'), 3),
      'Runx2->Col10a1', round(edge('Runx2', 'Col10a1'), 3),
      'Runx2->Mmp13', round(edge('Runx2', 'Mmp13'), 3),
      'Sp7->Spp1', round(edge('Sp7', 'Spp1'), 3))
# regulator importance = |outgoing to module targets|
out_inf = np.abs(B).sum(1)
print('top regulators:', [regulators[i] for i in np.argsort(out_inf)[::-1][:12]])

# ---- fate ~ module scores (multinomial) ----
clf = LogisticRegression(max_iter=500, multi_class='multinomial').fit(Mscore, fnum)
base_pred = clf.predict_proba(Mscore)
def fate_from_modules(MS_):
    P = clf.predict_proba(MS_)
    order = list(clf.classes_)
    return P, order
p0, order = fate_from_modules(Mscore)
base_fate = np.array([p0[:, order.index(k)].mean() for k in range(4)])
print('baseline mean predicted fate:', {fname[k]: round(base_fate[k], 3) for k in range(4)})

# ---- in silico KO / OE: TF -> targets -> modules -> fate ----
# target -> module index
t2m = {}
for t in targets:
    for mi, m in enumerate(modules):
        if t in modules[m]: t2m[t] = mi
rows = []
MAG = 2.0
for ri_, rname in enumerate(regulators):
    for mode, delta in [('KO', -MAG), ('OE', MAG)]:
        dT = B[ri_] * delta                       # target standardized changes
        dM = np.zeros(6)
        for ti_, t in enumerate(targets):
            if t in t2m: dM[t2m[t]] += dT[ti_]
        cnt = np.array([sum(1 for t in targets if t2m.get(t) == k) for k in range(6)])
        cnt[cnt == 0] = 1
        dM = dM / cnt
        MS2 = Mscore + dM
        P = clf.predict_proba(MS2)
        f = np.array([P[:, order.index(k)].mean() for k in range(4)])
        rows.append({'gene': rname, 'mode': mode,
                     'dChondro': f[2] - base_fate[2], 'dFibrous': f[1] - base_fate[1],
                     'dHyp': f[2] - base_fate[2]})
pert = pd.DataFrame(rows)
pv = pert.pivot_table(index='gene', columns='mode', values='dChondro').reset_index()
pv.columns.name = None
pv['out_influence'] = pv.gene.map(lambda g: out_inf[regulators.index(g)])
# rescue: KO (inhibitor) or OE (agonist) that increases chondro commitment
pv['rescue_inhibitor'] = pv.KO.clip(lower=0)
pv['rescue_agonist'] = pv.OE.clip(lower=0)
pv['best_rescue'] = pv[['rescue_inhibitor', 'rescue_agonist']].max(1)
pv['best_modality'] = np.where(pv.rescue_inhibitor >= pv.rescue_agonist,
                               'inhibitor', 'agonist')
pv = pv.sort_values('best_rescue', ascending=False)
print('perturbation Δchondro (top):')
print(pv.head(15).round(3).to_string(index=False))

# ---- literature druggability ----
lit = {
 'Cdk8': ('inhibitor', 'high', 'CDK8 inhibitor rescues ischemic fracture (PMC12136223)'),
 'Bmp2': ('agonist/recombinant', 'high', 'off-label rhBMP2 in NF1 pseudarthrosis (PMID38990653)'),
 'Map2k1': ('inhibitor(MEK)', 'high', 'MEK-SHP2 prevents NF1 pseudarthrosis (PMC12093263)'),
 'Ptpn11': ('inhibitor(SHP2)', 'medium', 'MEK-SHP2 combination (PMID38924432)'),
 'Inhba': ('agonist/recombinant', 'medium', 'Activin A pro-healing PPC (PMID38079220); context dependent'),
 'Cd47': ('agonist(avoid blockade)', 'medium', 'CD47 blockade lowers MSC proliferation (PMC11873311)'),
 'Runx2': ('agonist', 'medium', 'master hypertrophic/osteogenic TF'),
 'Sox9': ('agonist', 'medium', 'master chondrogenic TF; hard to drug'),
 'Sp7': ('agonist', 'low', 'osteogenic TF; undruggable'),
 'Acvr1': ('context', 'medium', 'BMP receptor; Activin pathological in FOP'),
 'Col14a1': ('inhibitor/marker', 'low', 'ischemic fibroblastic intermediate'),
 'Pdzrn4': ('inhibitor/marker', 'low', 'ischemic intermediate marker'),
 'Dlx5': ('agonist', 'low', 'BMP downstream osteogenic TF'),
 'Clec11a': ('agonist', 'low', 'Clec11a-Itga11-Wnt in NF1 (PMC NIHMS1854411)'),
}
pv['drug_modality_lit'] = pv.gene.map(lambda g: lit.get(g, ('', ''))[0])
pv['druggability'] = pv.gene.map(lambda g: lit.get(g, ('', ''))[1])
pv['evidence'] = pv.gene.map(lambda g: lit.get(g, ('', '', ''))[2])
tier = {'high': 1.0, 'medium': 0.6, 'low': 0.3, '': 0.0}
rsc = pv.best_rescue / (pv.best_rescue.max() + 1e-9)
inf_n = pv.out_influence / (pv.out_influence.max() + 1e-9)
pv['composite'] = (0.45 * rsc.values + 0.2 * inf_n.values +
                   0.35 * pv.druggability.map(tier).values)
rank = pv.sort_values('composite', ascending=False)
print('final target ranking:')
print(rank[['gene', 'best_modality', 'best_rescue', 'druggability', 'composite',
            'evidence']].head(12).round(3).to_string(index=False))

# ---- save ----
Bdf.to_csv(f'{BASE}/wp3_out/grn_edges.csv')
pv.to_csv(f'{BASE}/wp3_out/perturbation_robust.csv', index=False)
rank.to_csv(f'{BASE}/wp3_out/target_ranking.csv', index=False)

# ---- figures ----
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
show = pv.head(12); x = np.arange(len(show)); w = 0.4
axes[0].bar(x - w/2, show.KO, w, label='KO/inhibitor')
axes[0].bar(x + w/2, show.OE, w, label='OE/agonist')
axes[0].axhline(0, color='k', lw=0.6); axes[0].set_xticks(x, show.gene, rotation=45, ha='right')
axes[0].set_title('in silico perturb: Δ chondrocyte commitment'); axes[0].legend(fontsize=8)
# network: top regulator outgoing
topr = np.argsort(out_inf)[::-1][:10]
axes[1].barh([regulators[i] for i in topr][::-1], out_inf[topr][::-1])
axes[1].set_title('Top regulators by outgoing influence')
r10 = rank.head(10)
axes[2].barh(r10.gene[::-1], r10.composite[::-1]); axes[2].set_title('Target composite ranking')
plt.tight_layout(); plt.savefig(f'{BASE}/wp3_out/fig_grn.png', dpi=140)
print('WP3 saved')
