"""llama.cpp で Mamba 2.8B GGUF の末尾トークン特徴量を抽出する。

CUDA は使用しない。Apple Silicon の Metal 有効版 llama-cpp-python を使う。

事前に Metal 有効の llama-cpp-python をインストールする。
    CMAKE_ARGS="-DGGML_METAL=on" pip install --upgrade --force-reinstall llama-cpp-python

Samber フォルダーからのコピペ用実行例:
    cd ~/Desktop/samber
    python3 granite.py --model ./mamba-2.8b-q4_k_m.gguf --texts "Hello world" "Mamba is a state space model." --output ./last_token_features.npy

上の実行では、対応する token ID が ./last_token_features_token_ids.npy に自動保存される。

従来の実行例:
    python3 granite.py --model /path/to/mamba-2.8b-q4_k_m.gguf \\
        --texts "Hello world" "Mamba is a state space model." \\
        --output last_token_features.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from llama_cpp import LLAMA_POOLING_TYPE_NONE, Llama
except ImportError as error:
    raise SystemExit(
        "llama-cpp-python が必要です。例: "
        'CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python'
    ) from error


def extract_last_token_features(
    texts: list[str],
    model_path: str | Path,
    *,
    n_metal_layers: int = -1,
    n_ctx: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """各テキストの末尾トークンの最終層埋め込みを返す。

    `pooling_type=NONE` によりプーリングを無効化し、全トークンの特徴量を
    llama.cpp から受け取る。そこから各入力の最後のトークンだけを選択する。

    Returns:
        (features, token_ids) のタプル。
        features は shape: (len(texts), hidden_size)、dtype float32。
        token_ids は各 feature 行に対応する「末尾トークン」の ID。
    """
    if not texts:
        raise ValueError("texts must contain at least one string.")

    # llama.cpp の n_gpu_layers はバックエンド共通の引数名。
    # Metal ビルドの Apple Silicon では、-1 で全層を Metal GPU にオフロードする。
    llm = Llama(
        model_path=str(model_path),
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_NONE,
        n_gpu_layers=n_metal_layers,
        n_ctx=n_ctx,
        logits_all=False,
        verbose=False,
    )

    last_token_features: list[np.ndarray] = []
    last_token_ids: list[int] = []
    for text in texts:
        # embed() は GGUF のトークナイザ設定（BOS の要否を含む）に従う。
        # pooling_type=NONE のため shape は (token_count, hidden_size) になる。
        token_features = np.asarray(llm.embed(text, normalize=False), dtype=np.float32)

        if token_features.ndim != 2 or token_features.shape[0] == 0:
            raise RuntimeError(
                "トークン単位の特徴量を取得できませんでした。"
                " llama-cpp-python / llama.cpp を更新し、GGUF が Mamba 対応か確認してください。"
            )
        last_token_features.append(token_features[-1])

        # embed() と同じ GGUF トークナイザで ID を取得する。このスクリプトは
        # 文ごとに末尾 feature を一行保存するため、その末尾 ID も一つ保存する。
        # add_bos=True は llama.cpp の標準的な embedding 入力と合わせる設定。
        token_ids = llm.tokenize(text.encode("utf-8"), add_bos=True)
        if len(token_ids) != token_features.shape[0]:
            raise RuntimeError(
                "tokenize() と embed() のトークン数が一致しませんでした "
                f"({len(token_ids)} vs {token_features.shape[0]})。"
                " llama-cpp-python / llama.cpp のバージョンを確認してください。"
            )
        last_token_ids.append(int(token_ids[-1]))

    return np.stack(last_token_features), np.asarray(last_token_ids, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="")
    parser.add_argument("--texts", nargs="+", required=True, help="特徴量を抽出するテキスト")
    parser.add_argument("--output", default="last_token_features.npy", help="保存先 (.npy)")
    parser.add_argument(
        "--token-ids-output",
        default=None,
        help="対応する末尾 token ID の保存先（既定: <output名>_token_ids.npy）",
    )
    parser.add_argument("--n-metal-layers", type=int, default=-1, help="-1 なら全層をMetalへ")
    parser.add_argument("--n-ctx", type=int, default=2048)
    args = parser.parse_args()

    features, token_ids = extract_last_token_features(
        args.texts, args.model, n_metal_layers=args.n_metal_layers, n_ctx=args.n_ctx
    )
    np.save(args.output, features)
    output_path = Path(args.output)
    token_ids_output = (
        Path(args.token_ids_output)
        if args.token_ids_output
        else output_path.with_name(f"{output_path.stem}_token_ids.npy")
    )
    np.save(token_ids_output, token_ids)
    print(f"saved {features.shape} float32 features to {args.output}")
    print(f"saved {token_ids.shape} token IDs to {token_ids_output}")


if __name__ == "__main__":
    main()
