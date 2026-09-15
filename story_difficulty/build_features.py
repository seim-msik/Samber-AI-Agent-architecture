#!/usr/bin/env python3
"""隠れ状態 .npy から特徴量を集約し、難易度ラベル付きの features.csv を出力する。

特徴量（各物語ごとに 1 行）:
    n_tokens   : トークン数
    mean_norm  : トークンごとの L2 ノルムの平均
    norm_std   : トークンごとの L2 ノルムの標準偏差
    mean_mean  : トークンごとの次元平均の平均
    mean_var   : トークンごとの分散の平均
    mean_max   : トークンごとの最大値の平均
    mean_min   : トークンごとの最小値の平均
    sim_mean   : 隣接トークン間コサイン類似度の平均
    sim_std    : 隣接トークン間コサイン類似度の標準偏差
    sim_min    : 隣接トークン間コサイン類似度の最小値

ラベル（difficulty, 1〜100）:
    GiNZA の文分割で平均文長を、文字種判定で漢字比率を求め、
    平均文長と漢字比率の両方を基準に簡易スコアとして算出する。

実行例:
    python build_features.py
    python build_features.py --data-dir data --output data/features.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def is_kanji(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
    )


def kanji_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(is_kanji(c) for c in chars) / len(chars)


def difficulty_label(text: str, nlp) -> int:
    """平均文長と漢字比率から難易度 (1〜100) を簡易算出する。"""
    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not sentences:
        return 50
    avg_sentence_len = len(text) / len(sentences)
    kr = kanji_ratio(text)

    score = 50.0
    score += 30.0 * (avg_sentence_len - 8.0) / 32.0   # 文が長いほど難しい
    score += 30.0 * (kr - 0.15) / 0.25                # 漢字比率が高いほど難しい
    return int(max(1, min(100, round(score))))


def story_features(emb: np.ndarray) -> dict[str, float]:
    """トークン単位の隠れ状態 [tokens, hidden_size] を集約する。"""
    norms = np.linalg.norm(emb, axis=1)
    means = emb.mean(axis=1)
    variances = emb.var(axis=1)
    maxes = emb.max(axis=1)
    mins = emb.min(axis=1)

    if emb.shape[0] >= 2:
        prev, nxt = emb[:-1], emb[1:]
        numerator = (prev * nxt).sum(axis=1)
        denominator = np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1)
        similarities = numerator / np.maximum(denominator, 1e-12)
    else:
        similarities = np.array([0.0])

    return {
        "n_tokens": int(emb.shape[0]),
        "mean_norm": float(norms.mean()),
        "norm_std": float(norms.std()),
        "mean_mean": float(means.mean()),
        "mean_var": float(variances.mean()),
        "mean_max": float(maxes.mean()),
        "mean_min": float(mins.mean()),
        "sim_mean": float(similarities.mean()),
        "sim_std": float(similarities.std()),
        "sim_min": float(similarities.min()),
    }


def load_ginza():
    try:
        import spacy
    except ImportError:
        raise SystemExit("spacy/ginza がありません。\n    bash setup_metal.sh を実行してください。") from None
    for name in ("ja_ginza", "ja_ginza_electra"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    raise SystemExit('ja_ginza モデルが見つかりません。\n    bash setup_metal.sh を実行してください。')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/features.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hidden_dir = args.data_dir / "hidden_states"
    npy_paths = sorted(hidden_dir.glob("story_*.npy"))
    if not npy_paths:
        raise SystemExit(
            f"{hidden_dir} に story_*.npy がありません。\n"
            "    python extract_hidden.py を先に実行してください。"
        )

    nlp = load_ginza()
    rows = []
    for npy_path in npy_paths:
        story_id = npy_path.stem  # story_XXXX
        txt_path = args.data_dir / f"{story_id}.txt"
        if not txt_path.is_file():
            print(f"warn: {story_id}.txt が見つからないため {story_id}.npy をスキップします")
            continue

        emb = np.load(npy_path)
        if emb.ndim != 2:
            print(f"warn: {npy_path.name} の shape が {emb.shape} のためスキップします")
            continue

        text = txt_path.read_text(encoding="utf-8")
        row = {"story_id": story_id}
        row.update(story_features(emb))
        row["difficulty"] = difficulty_label(text, nlp)
        rows.append(row)

    if not rows:
        raise SystemExit("有効な隠れ状態がありませんでした。")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved {len(df)} rows -> {args.output}")
    print(f"difficulty: min={df['difficulty'].min()} max={df['difficulty'].max()} "
          f"mean={df['difficulty'].mean():.1f}")


if __name__ == "__main__":
    main()
