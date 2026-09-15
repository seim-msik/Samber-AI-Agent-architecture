#!/usr/bin/env python3
"""Load and inspect a NumPy .npy file (e.g. saved Mamba hidden states)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a .npy file and print its shape, dtype, and summary statistics."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Absolute path to the .npy file (omit to enter it interactively)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of flattened elements to show (default: 10; use 0 to skip)",
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        help="Memory-map the file; useful for very large hidden-state arrays",
    )
    args = parser.parse_args()

    path = args.path
    if path is None:
        entered_path = input("Absolute path to the .npy file: ").strip()
        if not entered_path:
            parser.error("No path was entered.")
        path = Path(entered_path).expanduser()

    if not path.is_absolute():
        parser.error("Enter an absolute path, beginning with '/'.")
    if path.suffix.lower() != ".npy":
        parser.error("Only .npy files are supported.")
    if not path.is_file():
        parser.error(f"File not found: {path}")

    array = np.load(path, mmap_mode="r" if args.mmap else None, allow_pickle=False)
    print(f"file:  {path}")
    print(f"shape: {array.shape}")
    print(f"dtype: {array.dtype}")
    print(f"ndim:  {array.ndim}")
    print(f"size:  {array.size:,} elements ({array.nbytes:,} bytes)")

    if np.issubdtype(array.dtype, np.number) and array.size:
        finite = np.isfinite(array)
        print(f"finite: {np.count_nonzero(finite):,}/{array.size:,}")
        if np.any(finite):
            values = array[finite]
            print(
                "stats: "
                f"min={values.min():.6g}, max={values.max():.6g}, "
                f"mean={values.mean():.6g}, std={values.std():.6g}"
            )

    if args.preview > 0:
        print(f"preview (first {args.preview} flattened values):")
        print(np.asarray(array).reshape(-1)[: args.preview])


if __name__ == "__main__":
    main()
