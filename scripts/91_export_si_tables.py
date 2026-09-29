#!/usr/bin/env python3
"""Export the machine-readable Supplemental Tables (Table S1-S12, incl. S8b)
from the pipeline outputs in wp0_out/wp2_out/wp3_out/wp4_out into
SI_build2/Supplemental_Tables/.

This is a deterministic copy/rename step; the mapping below was verified by
checksum against the tables shipped with the manuscript.
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'SI_build2', 'Supplemental_Tables')
os.makedirs(OUT, exist_ok=True)

MAP = {
    'wp0_out/composition.csv':            'TableS1_composition.csv',
    'wp0_out/abundance_test.csv':         'TableS2_abundance_trend.csv',
    'wp0_out/markers_top.csv':            'TableS3_state_markers.csv',
    'wp0_out/qc_summary.csv':             'TableS4_qc_summary.csv',
    'wp2_out/forward_fate.csv':           'TableS5_forward_fate.csv',
    'wp2_out/backward_ancestry.csv':      'TableS6_backward_ancestry.csv',
    'wp2_out/extended_fate.csv':          'TableS7_extended_fate.csv',
    'wp2_out/condition_metrics.csv':      'TableS8_condition_metrics.csv',
    'wp2_out/block_score.csv':            'TableS8b_block_score.csv',
    'wp3_out/condition_differential.csv': 'TableS9_condition_differential.csv',
    'wp3_out/target_ranking.csv':         'TableS10_target_ranking.csv',
    'wp4_out/morans_I.csv':               'TableS11_morans_I.csv',
    'wp4_out/junction_pathways.csv':      'TableS12_junction_pathways.csv',
}

for src, dst in MAP.items():
    s = os.path.join(ROOT, src)
    if not os.path.exists(s):
        raise FileNotFoundError(f'Missing {src}; run the upstream pipeline step first.')
    shutil.copy(s, os.path.join(OUT, dst))
    print('exported', dst, '<-', src)

print('Supplemental tables written to', OUT)
