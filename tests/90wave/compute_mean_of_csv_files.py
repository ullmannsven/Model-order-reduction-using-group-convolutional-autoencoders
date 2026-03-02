#!/usr/bin/env python3
"""
Compute the pointwise mean of two CSV files with columns (x, y).

Usage:
    python compute_mean_csv.py run1.csv run2.csv mean.csv
"""

import sys
import csv
from pathlib import Path


def read_csv(path):
    """Read CSV with columns x,y and return list of (x, y)."""
    data = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError(f"{path} must have columns 'x' and 'y'")
        for row in reader:
            data.append((float(row["x"]), float(row["y"])))
    return data


def main(file1, file2, outfile):
    file1 = Path(file1)
    file2 = Path(file2)
    outfile = Path(outfile)

    data1 = read_csv(file1)
    data2 = read_csv(file2)

    if len(data1) != len(data2):
        raise ValueError("CSV files have different number of rows")

    mean_data = []
    for (x1, y1), (x2, y2) in zip(data1, data2):
        if x1 != x2:
            raise ValueError(
                f"x-values do not match: {x1} vs {x2}"
            )
        mean_data.append((x1, 0.5 * (y1 + y2)))

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(mean_data)

    print(f"Wrote mean to {outfile}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_mean_csv.py run1.csv run2.csv mean.csv")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
