#!/usr/bin/env python3
"""
Regenerate Task 4 tables, figure catalog, and CSV/JSON-based plots (no GPU).

Rebuilds: ablation_tables.md, figure_catalog.md, ablation_clean_vs_noisy_r2.png,
noise_sweep_a_vs_b.png, ablation_loss_curves.png (if histories JSON exists).

Usage:
    pip install -r requirements.txt
    python generate_visualizations.py
"""

from visualization import regenerate_static_plots_from_csv

if __name__ == "__main__":
    regenerate_static_plots_from_csv()
