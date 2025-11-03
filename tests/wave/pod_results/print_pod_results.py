#!/usr/bin/env python3
"""
Plot (x, y) data from .pkl files in the current folder using a log y-axis.

Each .pkl file must contain a Python list (or iterable) of pairs/tuples (x, y).
If multiple .pkl files are provided/found, each will be plotted as a separate curve.

Usage:
  python plot_pickle_logy.py                 # auto-detects *.pkl in current dir
  python plot_pickle_logy.py file1.pkl
  python plot_pickle_logy.py file1.pkl file2.pkl
  python plot_pickle_logy.py --save myplot.png
  python plot_pickle_logy.py --no-show       # don't open a window, just save

Notes:
- Non-positive y-values are filtered out (cannot be shown on log scale).
- The plot is saved to "plot_logy.png" by default (or --save path).
"""

from __future__ import annotations
import argparse
import glob
import os
import pickle
from typing import Iterable, Tuple, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def load_xy_from_pickle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a list of (x, y) pairs from a pickle file and return as numpy arrays."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Accept list/iterable of pairs; convert robustly
    try:
        xs, ys = zip(*data)  # may raise if structure is wrong / empty
    except Exception as e:
        raise ValueError(f"{path}: expected an iterable of (x, y) pairs; got error: {e}")

    x_arr = np.asarray(xs, dtype=float).reshape(-1)
    y_arr = np.asarray(ys, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"{path}: x and y lengths differ: {x_arr.shape} vs {y_arr.shape}")

    return x_arr, y_arr


def main():
    parser = argparse.ArgumentParser(description="Plot (x, y) data from .pkl files with log y-axis.")
    parser.add_argument("pickles", nargs="*", help="Paths to .pkl files. Defaults to all *.pkl in current folder.")
    parser.add_argument("--save", default=None, help="Output image filename")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window (save only).")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--markersize", type=float, default=4.0, help="Marker size for points (default: 4.0).")
    args = parser.parse_args()

    paths: List[str] = args.pickles or sorted(glob.glob("*.pkl"))
    if not paths:
        raise SystemExit("No .pkl files provided or found in the current directory.")

    # Use non-interactive backend if --no-show is requested
    if args.no_show:
        matplotlib.use("Agg", force=True)

    plt.figure(figsize=(8, 5))

    for p in paths:
        try:
            x, y = load_xy_from_pickle(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

        # Filter non-positive y for log scale
        mask = y > 0
        if not np.all(mask):
            n_bad = int((~mask).sum())
            print(f"{p}: filtered out {n_bad} non-positive y-values for log scale.")
        x_f = x[mask]
        y_f = y[mask]
        if x_f.size == 0:
            print(f"{p}: nothing to plot after filtering; skipping.")
            continue

        # Sort by x for nicer lines (optional)
        order = np.argsort(x_f)
        x_f = x_f[order]
        y_f = y_f[order]

        plt.plot(x_f, y_f, linestyle='none', marker='o', markersize=args.markersize, label=os.path.basename(p))
        
    plt.yscale("log")
    plt.xlabel("2n")
    plt.ylabel("POD projection error")
    if args.title:
        plt.title(args.title)
    if len(paths) > 1:
        plt.legend()
    plt.grid(True, which="both", linewidth=0.5, alpha=0.5)

    # Save
    if args.save is not None:
        out_path = args.save
    else:
        if len(paths) == 1:
            base = os.path.splitext(os.path.basename(paths[0]))[0]
            out_path = f"{base}.png"
        else:
            out_path = "plot_logy.png"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    print(f"Saved figure to: {out_path}")

    # Show if requested (default behavior)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()