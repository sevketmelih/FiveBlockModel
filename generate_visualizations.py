#!/usr/bin/env python3
"""
Regenerate Task 4 tables and CSV-based plots without retraining models.

Usage:
    python generate_visualizations.py
"""

from visualization import regenerate_static_plots_from_csv

if __name__ == "__main__":
    regenerate_static_plots_from_csv()
