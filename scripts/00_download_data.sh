#!/usr/bin/env bash
# Download and unpack the three public GEO datasets used in this study.
# Raw data are NOT stored in the git repository (see .gitignore); run this once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
GEO="https://ftp.ncbi.nlm.nih.gov/geo/series"

mkdir -p "$DATA/gse185940" "$DATA/gse292578" "$DATA/gse218046/extracted"

# ---------------------------------------------------------------------------
# GSE185940 : mouse forelimb developmental scRNA-seq (E10.5-E14.5).
# RAW.tar contains one GSM*.txt.gz per plate (genes x cells, tab-separated).
# The study analyses the Tamoxifen-labelled gates at E11.5-E14.5 (27,264 cells).
# ---------------------------------------------------------------------------
cd "$DATA/gse185940"
if ! ls GSM*.txt.gz >/dev/null 2>&1; then
  curl -L --retry 3 -O "$GEO/GSE185nnn/GSE185940/suppl/GSE185940_RAW.tar"
  tar xf GSE185940_RAW.tar
fi
cp "$ROOT/manifests/gse185940_plate_design.csv" "$DATA/gse185940/plate_design.csv"
echo "GSE185940 ready: $(ls GSM*.txt.gz | wc -l) plates"

# ---------------------------------------------------------------------------
# GSE292578 : ischemic and CD47-KO fracture stromal scRNA-seq (day 4 / day 7).
# Three files: sparse DGE (MatrixMarket), gene index and per-cell metadata.
# ---------------------------------------------------------------------------
cd "$DATA/gse292578"
for f in \
  GSE292578_all_unfiltered_DGE.mtx.gz \
  GSE292578_all_unfiltered_genes.csv.gz \
  GSE292578_all_unfiltered_cell_metadata.csv.gz ; do
  [ -s "$f" ] || curl -L --retry 3 -O "$GEO/GSE292nnn/GSE292578/suppl/$f"
done
echo "GSE292578 ready"

# ---------------------------------------------------------------------------
# GSE218046 : 10x Visium fracture callus, control (GSM6733377) vs NF1 (GSM6733378).
# RAW.tar holds the filtered feature-barcode .h5 plus spatial positions/scalefactors.
# ---------------------------------------------------------------------------
cd "$DATA/gse218046"
if [ ! -s extracted/GSM6733377_filtered_feature_bc_matrix.h5 ]; then
  [ -s GSE218046_RAW.tar ] || curl -L --retry 3 -O "$GEO/GSE218nnn/GSE218046/suppl/GSE218046_RAW.tar"
  tar xf GSE218046_RAW.tar -C extracted
fi
echo "GSE218046 ready"
echo "All datasets downloaded under $DATA"
