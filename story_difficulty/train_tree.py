#!/usr/bin/env python3
"""features.csv から決定木（回帰）で難易度を学習し、R2 スコアを表示する。

実行例:
    python train_tree.py
    python train_tree.py --csv data/features.csv --max-depth 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

FEATURE_COLUMNS = [
    "n_tokens",
    "mean_norm",
    "norm_std",
    "mean_mean",
    "mean_var",
    "mean_max",
    "mean_min",
    "sim_mean",
    "sim_std",
    "sim_min",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/features.csv"))
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise SystemExit(
            f"{args.csv} がありません。\n    python build_features.py を先に実行してください。"
        )
    if not (0.0 < args.test_size < 1.0):
        raise SystemExit("--test-size は 0 より大きく 1 未満にしてください。")

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"features.csv に必要な列がありません: {missing}")

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["difficulty"].to_numpy(dtype=np.float64)

    if len(df) < 2:
        raise SystemExit("決定木の学習には最低 2 サンプル必要です。")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = DecisionTreeRegressor(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    r2_train = float(r2_score(y_train, model.predict(X_train)))
    rmse_train = float(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))

    print(f"train: {len(X_train)} samples / R2 = {r2_train:.4f} / RMSE = {rmse_train:.4f}")

    if len(X_test) > 0:
        y_pred = model.predict(X_test)
        r2_test = float(r2_score(y_test, y_pred))
        rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        print(f"test : {len(X_test)} samples / R2 = {r2_test:.4f} / RMSE = {rmse_test:.4f}")
        for i in range(min(len(y_test), 5)):
            print(f"    actual={y_test[i]:.0f} predicted={y_pred[i]:.2f}")

    importances = model.feature_importances_
    print("\nfeature importances:")
    for name, importance in sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda item: item[1], reverse=True
    ):
        print(f"    {name:<10} {importance:.4f}")


if __name__ == "__main__":
    main()
