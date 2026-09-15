#!/usr/bin/env python3
"""GGUF 形式の Mamba モデルから、トークン単位の隠れ状態（埋め込み）を抽出して .npy に保存する。

- llama-cpp-python の Metal ビルドを使い、Apple Silicon の GPU で推論する。
- pooling_type=LLAMA_POOLING_TYPE_NONE により、各トークンの埋め込みを取得する。
- 各物語は data/story_XXXX.txt → data/hidden_states/story_XXXX.npy に保存される。
  .npy の shape は [tokens, hidden_size]、dtype は float32。

実行例:
    python extract_hidden.py
    python extract_hidden.py --model models/mamba-130m-hf-Q3_K_M.gguf --data-dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/mamba-130m-hf-Q3_K_M.gguf"),
                        help="GGUF モデルパス")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="story_*.txt と hidden_states/ の親ディレクトリ")
    parser.add_argument("--n-ctx", type=int, default=4096, help="コンテキスト長")
    parser.add_argument("--n-batch", type=int, default=4096,
                        help="プロンプト評価のバッチサイズ（n_ctx 以上にしないと末尾で切れる）")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="GPU(Metal)にオフロードする層数 (-1 = 全層)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from llama_cpp import LLAMA_POOLING_TYPE_NONE, Llama
    except ImportError:
        raise SystemExit(
            "llama-cpp-python がありません。\n    bash setup_metal.sh を実行してください。"
        ) from None

    if not args.model.is_file():
        raise SystemExit(
            f"モデルが見つかりません: {args.model}\n"
            "    bash setup_metal.sh でモデルを取得してください。"
        )

    story_paths = sorted(args.data_dir.glob("story_*.txt"))
    if not story_paths:
        raise SystemExit(
            f"{args.data_dir} に story_*.txt がありません。\n"
            "    python generate_stories.py --n 10 を実行してください。"
        )

    print(f"モデルを読み込みます: {args.model}")
    llm = Llama(
        model_path=str(args.model),
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE,  # トークン単位の埋め込みを有効化
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        logits_all=False,
        verbose=True,  # Metal バックエンドのログを表示する
    )

    out_dir = args.data_dir / "hidden_states"
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in story_paths:
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        embedding = np.asarray(llm.embed(text, normalize=False), dtype=np.float32)
        if embedding.ndim != 2:
            raise SystemExit(
                f"{txt_path.name}: 埋め込みの shape が {embedding.shape} です。"
                " pooling_type=NONE が有効か llama-cpp-python のバージョンを確認してください。"
            )

        out_path = out_dir / f"{txt_path.stem}.npy"
        np.save(out_path, embedding)
        print(f"saved {out_path.name}: {embedding.shape} float32")


if __name__ == "__main__":
    main()
