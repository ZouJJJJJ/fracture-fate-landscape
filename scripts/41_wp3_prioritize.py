# -*- coding: utf-8 -*-
"""WP3 driver/target prioritization (robust, convergent evidence).
1) Pseudotime along progenitor->chondrocyte; driver genes by pseudotime correlation.
2) Direction of fate control = per-gene correlation with the chondrocyte module
   (positive=pro-commitment agonist; negative=blocker -> inhibitor rescues).
3) Failure-condition differential expression (GSE292578 day7): mediators altered in
   ischemia / CD47-KO with bootstrap significance.
4) in silico KO/OE directional readout from a stable univariate response (single-target
   magnitudes are small = distributed control; evidence is convergent, not large single
   effects). Final rank = condition mediator + trajectory driver + literature druggability.
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, sys, gc
from scipy import sparse
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

BASE = ''+ROOT+''
op = np.load(f'{BASE}/wp1_out/pca_operator_ext.npz', allow_pickle=True)
genes = list(op['genes']); gpos = {g: i for i, g in enumerate(genes)}
mean, sd = op['mean'], op['sd']
at = np.load(f'{BASE}/wp0_out/atlas.npz', allow_pickle=True); cs = at['state']
fmap = {'mesenchyme_condensation': 0, 'limb_mesenchyme': 0, 'fibrous_perichondrium': 1,
        'tendon': 1, 'committed_chondro': 2, 'proliferative_chondro': 2, 'immune': 3}
fnum = np.array([fmap.get(s, 3) for s in cs])

raw = np.load(f'{BASE}/data/gse185940/gse185940_lognorm.npz', allow_pickle=True)
shape = tuple(raw['shape'])
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
Es = (E - mean) / sd

modules = {'prog': ['Prrx1', 'Pdgfra', 'Cxcl12', 'Lepr', 'Postn', 'Pdgfrb'],
           'chondro': ['Sox9', 'Col2a1', 'Acan', 'Col11a1', 'Sox5', 'Sox6'],
           'fibrous': ['Col1a1', 'Col3a1', 'Dcn', 'Lum', 'Lgals1', 'Col1a2'],
           'hyp': ['Col10a1', 'Runx2', 'Mmp13', 'Ibsp', 'Mmp9'],
           'osteo': ['Sp7', 'Spp1', 'Alpl', 'Bglap'],
           'prolif': ['Mki67', 'Top2a', 'Cdk1', 'Pcna', 'Mcm2']}
mk = np.zeros((len(genes), 6))
for mi, (m, gs_) in enumerate(modules.items()):
    for g in gs_:
        if g in gpos: mk[gpos[g], mi] = 1
Mscore = Es @ mk / np.maximum(mk.sum(0), 1)

pt = (Mscore[:, 1] + Mscore[:, 3] + Mscore[:, 4]) - Mscore[:, 0]
pt = (pt - pt.min()) / (pt.max() - pt.min())

def safe_corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    return (x * y).sum() / den if den > 0 else 0.0
c_pt = np.array([safe_corr(Es[:, i], pt) for i in range(len(genes))])
c_ch = np.array([safe_corr(Es[:, i], Mscore[:, 1]) for i in range(len(genes))])
detect = np.abs(Es).mean(0)
driver = np.abs(c_pt) * detect
Ddf = pd.DataFrame({'gene': genes, 'corr_pseudotime': c_pt, 'corr_chondro': c_ch,
                    'driver': driver, 'detect': detect})
print('pro-commitment drivers:', Ddf.nlargest(10, 'driver').gene.tolist())
print('anti-commitment drivers:', Ddf[Ddf.corr_chondro < -0.05]
      .nlargest(8, 'driver').gene.tolist())

# condition differential
zc = np.load(f'{BASE}/data/gse292578/built/stromal_lognorm.npz', allow_pickle=True)
shc = tuple(zc['shape'])
XC = sparse.csr_matrix((zc['data'], zc['indices'], zc['indptr']), shape=shc)
gcenes = pd.read_csv(f'{BASE}/data/gse292578/built/genes.csv')['gene'].astype(str).values
gcpos = {g: i for i, g in enumerate(gcenes)}
cm = pd.read_csv(f'{BASE}/data/gse292578/built/stromal_meta.csv')
panel = sorted(set([g for gs_ in modules.values() for g in gs_] +
                   ['Cdk8', 'Col14a1', 'Pdzrn4', 'Map2k1', 'Ptpn11', 'Cd47', 'Inhba',
                    'Bmp2', 'Acvr1', 'Bmpr1a', 'Clec11a', 'Dlx5', 'Grem1', 'Nog']))
panel = [g for g in panel if g in gcpos]
Pcol = np.array([gcpos[g] for g in panel])
def cond_mean(cond, day):
    k = np.where(((cm.condition == cond) & (cm.day == day)).values)[0]
    return np.asarray(XC[k][:, Pcol].mean(0)).ravel(), len(k)
means = {}
for cond in ['WT_Intact', 'WT_Ischemic', 'CD47_KO']:
    means[cond], _ = cond_mean(cond, 'day7')
Cmean = pd.DataFrame(means, index=panel)
Cmean['d_ischemic'] = Cmean.WT_Ischemic - Cmean.WT_Intact
Cmean['d_CD47'] = Cmean.CD47_KO - Cmean.WT_Intact
rng = np.random.default_rng(5)
def boot_delta(cond, B=120):
    k = np.where(((cm.condition == cond) & (cm.day == 'day7')).values)[0]
    kr = np.where(((cm.condition == 'WT_Intact') & (cm.day == 'day7')).values)[0]
    acc = []
    for _ in range(B):
        ks = rng.choice(k, len(k)); krs = rng.choice(kr, len(kr))
        a = np.asarray(XC[ks][:, Pcol].mean(0)).ravel()
        b = np.asarray(XC[krs][:, Pcol].mean(0)).ravel()
        acc.append(a - b)
    acc = np.array(acc)
    return pd.DataFrame({'lo': np.percentile(acc, 2.5, 0),
                         'hi': np.percentile(acc, 97.5, 0)}, index=panel)
ciI, ciK = boot_delta('WT_Ischemic'), boot_delta('CD47_KO')
Cmean['sig_ischemic'] = (np.sign(ciI.lo) == np.sign(ciI.hi))
Cmean['sig_CD47'] = (np.sign(ciK.lo) == np.sign(ciK.hi))
print('day7 differential ischemic (significant):')
print(Cmean[Cmean.sig_ischemic][['d_ischemic']].round(3).head(12))
print('day7 differential CD47 (significant):')
print(Cmean[Cmean.sig_CD47][['d_CD47']].round(3).head(12))

# in silico directional perturb (univariate)
slope = c_ch
rows = []
for g in panel:
    if g not in gpos: continue
    i = gpos[g]; s = slope[i]
    KO = (-2.0) * s; OE = (2.0) * s
    rows.append({'gene': g, 'KO_dChondro': KO, 'OE_dChondro': OE,
                 'rescue_inhibitor': max(KO, 0), 'rescue_agonist': max(OE, 0)})
Pv = pd.DataFrame(rows)
Pv['best_rescue'] = Pv[['rescue_inhibitor', 'rescue_agonist']].max(1)
Pv['best_modality'] = np.where(Pv.rescue_inhibitor >= Pv.rescue_agonist,
                               'inhibitor', 'agonist')

ev = Pv.set_index('gene')
ev['corr_pseudotime'] = Ddf.set_index('gene').corr_pseudotime
ev['driver'] = Ddf.set_index('gene').driver
ev['d_ischemic'] = Cmean.d_ischemic; ev['d_CD47'] = Cmean.d_CD47
ev['sig_ischemic'] = Cmean.sig_ischemic; ev['sig_CD47'] = Cmean.sig_CD47
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
ev['drug_modality_lit'] = [lit.get(g, ('', ''))[0] for g in ev.index]
ev['druggability'] = [lit.get(g, ('', ''))[1] for g in ev.index]
ev['evidence'] = [lit.get(g, ('', '', ''))[2] for g in ev.index]
# intervention direction from failure-condition differential (up->inhibitor, down->agonist),
# falling back to literature modality then the intrinsic univariate direction.
def intervention(g):
    if g in Cmean.index:
        r = Cmean.loc[g]
        d = r.d_ischemic if bool(r.sig_ischemic) else (r.d_CD47 if bool(r.sig_CD47) else np.nan)
        if not np.isnan(d):
            return 'inhibitor' if d > 0 else 'agonist'
    lm = lit.get(g, ('', ''))[0]
    if lm: return lm.split('(')[0].strip()
    return ev.loc[g].best_modality
ev['best_modality'] = [intervention(g) for g in ev.index]
med_score = (ev.d_ischemic.abs() * ev.sig_ischemic.astype(float) +
             ev.d_CD47.abs() * ev.sig_CD47.astype(float))
def nz(x): return x / (x.max() + 1e-9)
tier = {'high': 1.0, 'medium': 0.6, 'low': 0.3, '': 0.0}
ev['composite'] = (0.30 * nz(med_score).values + 0.20 * nz(ev.driver).values +
                   0.15 * nz(ev.best_rescue).values +
                   0.35 * ev.druggability.map(tier).values)
rank = ev.sort_values('composite', ascending=False)
print('final target ranking:')
print(rank[['best_modality', 'best_rescue', 'driver', 'druggability', 'composite',
            'evidence']].head(13).round(3).to_string())

Ddf.to_csv(f'{BASE}/wp3_out/pseudotime_drivers.csv', index=False)
Cmean.to_csv(f'{BASE}/wp3_out/condition_differential.csv')
rank.to_csv(f'{BASE}/wp3_out/target_ranking.csv')

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
show = ev.sort_values('best_rescue', ascending=False).head(12)
x = np.arange(len(show)); w = 0.4
axes[0].bar(x - w/2, show.KO_dChondro, w, label='KO/inhibitor')
axes[0].bar(x + w/2, show.OE_dChondro, w, label='OE/agonist')
axes[0].axhline(0, color='k', lw=0.6); axes[0].set_xticks(x, show.index, rotation=45, ha='right')
axes[0].set_title('in silico directional perturb: d chondrocyte module'); axes[0].legend(fontsize=8)
dd = Ddf.nlargest(10, 'driver')
axes[1].barh(dd.gene[::-1], dd.driver[::-1]); axes[1].set_title('Pseudotime driver genes')
r10 = rank.head(10)
axes[2].barh(r10.index[::-1], r10.composite[::-1]); axes[2].set_title('Target composite ranking')
plt.tight_layout(); plt.savefig(f'{BASE}/wp3_out/fig_grn.png', dpi=140)
print('WP3 saved')
