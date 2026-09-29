# fracture-fate-landscape

Reproducible analysis code for:

**Reconstructing the stochastic Waddington landscape of bone regeneration
identifies CDK8 as a pan-non-union therapeutic target**

Jie Zou, Bo Guo, Liang Ma, Weibo Zheng, Xiaoai Zhou, Meiyun Tan\*

Department of Orthopedics, The Affiliated Hospital of Southwest Medical
University, Luzhou 646000, Sichuan, China.
\*Corresponding author: Meiyun Tan (drtmy169@swmu.edu.cn)

The repository reconstructs a developmental baseline (mouse limb-bud
endochondral development, E11.5–E14.5) with a regularized unbalanced optimal
transport (RUOT) neural model, derives a stochastic Waddington landscape and
lineage probabilities, maps three adult fracture conditions onto that baseline
to quantify a non-union blockage, prioritizes targets with a multi-evidence
strategy (ITPS), and validates the top hit on spatial transcriptomics.

---

## 1. What is included

| Path | Contents |
|------|----------|
| `scripts/` | All data-preparation, modelling, figure and supplemental-table scripts, numbered in run order |
| `vendor/DiffusionOT/` | The subset of the third-party **DiffusionOT** package imported by this study (MIT; see below) |
| `manifests/` | The GSE185940 plate design (`plate_design.csv`) used to label plates |
| `wp0_out … wp4_out/` | Committed **lightweight results** (`.csv` tables and `.png` figures) for immediate inspection |
| `supp_figures/` | Composite supporting figures rendered from the pipeline outputs |
| `SI_build2/` | The four panel-cropped Supplemental Figures (`.png`) and Tables S1–S12 (`.csv`) |
| `requirements.txt` | Pinned CPU environment |
| `LICENSE` | MIT license for the code in this repository |

Raw data (several GB), large intermediate arrays (`*.npz`) and trained model
weights (`*.pth`) are **not** committed (see `.gitignore`); they are downloaded
and regenerated with the commands below. The committed `.csv/.png` outputs let
reviewers audit every reported number and figure without rerunning anything.

---

## 2. Data (all public, no controlled-access data)

| GEO series | Role in the study | Key objects |
|------------|-------------------|-------------|
| **GSE185940** | Mouse forelimb developmental scRNA-seq; the Tamoxifen-labelled gates at **E11.5–E14.5** form the developmental baseline (27,264 cells) | one `GSM*.txt.gz` per plate |
| **GSE292578** | Adult fracture stromal scRNA-seq: intact vs ischemic vs CD47-null, day 4 / day 7 | sparse DGE, genes, cell metadata |
| **GSE218046** | 10x Visium fracture callus: control (`GSM6733377`) vs NF1 (`GSM6733378`) | filtered feature-barcode `.h5`, spatial positions, scalefactors |

Download and unpack all three (uses `curl`/`tar`):

```bash
bash scripts/00_download_data.sh
```

This populates `data/gse185940/`, `data/gse292578/` and
`data/gse218046/extracted/` and copies the plate manifest into place.

---

## 3. Installation

Linux, CPython 3.12, CPU-only (a GPU is not required).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the environment that generated the committed results
(`torch 2.14.0` CPU, `TorchDiffEqPack 1.0.1`, `numpy 1.26.4`, `scipy 1.17.1`,
`pandas 2.2.3`, `scikit-learn 1.4.0`, etc.). The neural-ODE solver
`TorchDiffEqPack` and the RUOT model are required for the training/field steps.

The DiffusionOT code used here is vendored verbatim under
`vendor/DiffusionOT/` (only `utility.py`, `training.py` and the `AE/` package),
so no separate download is needed. It is distributed under the MIT license,
Copyright (c) 2025 Juntan Liu; upstream:
<https://github.com/liujuntan/DiffusionOT>.

---

## 4. Reproduction (run order)

Run from the repository root. Steps 01–03 build the data matrices; WP0–WP4 are
the five analysis work packages; steps 91–92 assemble the Supplemental
Information.

```bash
# --- data preparation ---
bash scripts/00_download_data.sh
python scripts/01_build_gse185940.py          # -> data/gse185940/gse185940_lognorm.npz, genelist.csv, cell_meta.csv
python scripts/02_make_latent_gse185940.py 10000 8   # args: per-stage cap, n_PCs; cap>2000 writes gse185940_latent_full.npz
python scripts/03_build_gse292578.py          # -> data/gse292578/built/ (stromal_lognorm.npz, genes.csv, stromal_meta.csv)

# --- WP0: developmental atlas, state annotation, QC (Figure 2; Tables S1-S4) ---
python scripts/10_wp0_atlas.py                # initial clustering -> wp0_out/atlas.npz
python scripts/11_wp0_full.py                 # markers, composition, abundance trends, QC -> wp0_out/

# --- WP1: RUOT training, PCA operator, landscape/fields (Figure 3; Figure S1) ---
python scripts/20_full_train.py               # 3-stage RUOT, CPU -> wp1_out/ckpt_*.pth, loss_history.csv
python scripts/21_pca_operator.py             # gene<->latent PCA operator -> wp1_out/pca_operator.npz
python scripts/22_extend_operator.py          # curated bone/fracture gene panel -> wp1_out/pca_operator_ext.npz
python scripts/23_wp1_fields.py               # potential L, velocity v, growth g, diffusion D -> wp1_out/fields.npz, figures

# --- WP2: stochastic trajectory analysis + adult conditions (Figures 4-5; Tables S5-S8b) ---
python scripts/30_wp2_sta.py                  # entropic-OT fates/ancestry -> wp2_out/
python scripts/31_wp2_conditions.py           # WT/ischemic/CD47 block metrics -> wp2_out/

# --- WP3: GRN, in-silico perturbation, ITPS (Figure 6; Tables S9-S10) ---
python scripts/40_wp3_grn.py                  # directed GRN, KO/OE, dose-response -> wp3_out/
python scripts/41_wp3_prioritize.py           # pseudotime drivers, condition DE, target ranking -> wp3_out/
python scripts/42_make_itps.py                # ITPS funnel figure -> wp3_out/fig_itps.png

# --- supporting composite figures (Figures S1-S7 source composites) ---
python scripts/43_make_supp_figures.py        # -> supp_figures/figS1_*.png … figS7_*.png

# --- WP4: spatial validation (Figure 7; Tables S11-S12) ---
python scripts/50_wp4_spatial.py              # Visium modules, Moran's I, junction tests -> wp4_out/

# --- Supplemental Information ---
python scripts/91_export_si_tables.py         # copy/rename wp outputs -> SI_build2/Supplemental_Tables/
python scripts/92_build_si_docx.py            # embed FigS1-S4 + table legends -> SI_build2/Supplementary_Information.docx
```

### Training effort / smoke test

`20_full_train.py` runs on CPU with the faithful three-stage schedule
(score pretrain → velocity/growth pretrain → total RUOT). Iteration counts and
batch size are environment variables (defaults shown):

```bash
N_SCORE=80 N_PRE=50 N_TOTAL=500 NUM_SAMPLES=128 python scripts/20_full_train.py
```

Reduce them (e.g. `N_SCORE=20 N_PRE=20 N_TOTAL=50`) for a fast smoke run.
`02_make_latent_gse185940.py` writes the *full* latent when the per-stage cap
exceeds 2000; the value `10000` exceeds the largest Tamoxifen gate (7,680 cells)
and therefore retains every cell.

---

## 5. Script-to-figure/table mapping

| Script | Primary outputs | Manuscript element |
|--------|-----------------|--------------------|
| 10/11 `wp0_*` | `wp0_out/fig_wp0_atlas.png`, `fig_wp0_qc.png`, `composition.csv`, `abundance_test.csv`, `markers_top.csv`, `qc_summary.csv` | Figure 2; Tables S1–S4 |
| 20 `full_train` | `wp1_out/ckpt_*.pth`, `loss_history.csv` | RUOT model behind Figures 3–4 |
| 21/22 `*operator*` | `pca_operator.npz`, `pca_operator_ext.npz` | gene↔latent mapping for perturbation (Figure 6) |
| 23 `wp1_fields` | `wp1_out/fig_landscape.png`, `fig_depth_diffusion.png`, `fields.npz`, `attractors.csv`, `barriers.csv` | Figure 3; Figure S1 |
| 30 `wp2_sta` | `wp2_out/fig_sta.png`, `forward_fate.csv`, `backward_ancestry.csv`, `extended_fate.csv`, `transitions.npz` | Figure 4; Tables S5–S7 |
| 31 `wp2_conditions` | `wp2_out/fig_conditions.png`, `condition_metrics.csv`, `block_score.csv` | Figure 5; Tables S8, S8b |
| 40 `wp3_grn` | `wp3_out/fig_grn.png`, `grn_matrix.npz`, `grn_edges.csv`, `perturbation*.csv` | Figure 6D–F |
| 41 `wp3_prioritize` | `wp3_out/fig_itps` inputs, `pseudotime_drivers.csv`, `condition_differential.csv`, `target_ranking.csv` | Figure 6A–C; Tables S9–S10 |
| 42 `make_itps` | `wp3_out/fig_itps.png` | Figure 6A |
| 43 `make_supp_figures` | `supp_figures/figS1_…figS7_*.png` | Supplemental Figure composites |
| 50 `wp4_spatial` | `wp4_out/fig_spatial_maps.png`, `fig_niche.png`, `fig10_assembled.png`, `morans_I.csv`, `junction_pathways.csv` | Figure 7; Tables S11–S12; Figure S4 |
| 91/92 `build_si_*` | `SI_build2/Supplemental_Tables/*.csv`, `SI_build2/Supplementary_Information.docx` | Supplemental Information |

The seven **main** figures were composited for publication from the rendered
`wp*_out` panels (panel placement/lettering is presentation, not computation).
The four final Supplemental Figure PNGs in `SI_build2/Supplementary_Figures/`
are provided as the exact submitted panels; their data-level source composites
are regenerated by `43_make_supp_figures.py`.

---

## 6. Key analysis settings (for reference)

- RUOT model: latent dimension 8, hidden dimension 16, 3 hidden Tanh layers,
  fixed irreducible noise `d = 0.001`; joint inference of drift `v`, diffusion
  `D` and growth `g`.
- Developmental trunk: Tamoxifen-labelled cells at E11.5/E12.5/E13.5/E14.5,
  1,800 highly variable genes, 8 principal components.
- Random seeds are fixed in the scripts (e.g. NumPy/training seed 11,
  `torch.manual_seed(1)` in the field step; bootstrap/OT generators seed 11),
  so the stochastic and resampling steps are reproducible.
- Reported landscape quantities (E14.5 chondrogenic attractor depth ≈ 0.316 vs
  fibrous ≈ 0.019), terminal-fate mix, day-7 condition block metrics, the ITPS
  ranking (CDK8 top, composite 0.658) and the NF1 junction BMP–MAPK statistics
  are all reproduced by the indicated scripts and stored as committed CSVs.

---

## 7. License

The analysis code in this repository is released under the MIT License
(`LICENSE`). The vendored DiffusionOT files under `vendor/DiffusionOT/` retain
their original MIT License, Copyright (c) 2025 Juntan Liu
(<https://github.com/liujuntan/DiffusionOT>).

## 8. Citation and code DOI

A versioned archive of this repository is deposited on Zenodo and assigned a
DOI (the DOI and citation are inserted in the manuscript's Data and Code
Availability statement at submission/acceptance). Please cite both the
manuscript and Liu *et al.* (DiffusionOT) when using this code.

For questions or reproducibility requests, contact the corresponding author at
drtmy169@swmu.edu.cn.
#   f r a c t u r e - f a t e - l a n d s c a p e  
 