#!/usr/bin/env python3
"""日本語プロンプトの形態素解析特徴量から、Mamba の隠れ状態を XGBoost で推定する。

既存スクリプトとの関係
---------------------
- granite.py                        : テキスト -> Mamba の末尾トークン隠れ状態 (.npy) を抽出
                                      （本スクリプト用の訓練データを生成する側）
- visualize_mamba_hidden_states.py  : .npy を読み、PCA で可視化 / 表形式表示
- inspect_npy.py                    : .npy の形状・統計情報の確認

本スクリプトは「隠れ状態を XGBoost で推定する」ための学習 / 推定 CLI。

- GiNZA (ja_ginza) で日本語プロンプトを形態素解析し、
  形態素（表層形 / 品詞 / 基本形 / 読み）を CountVectorizer でカウント特徴量にする。
- 文字 n-gram 特徴量 (1〜3) と結合する（未知語・漢字の変形に強い）。
- XGBoost の MultiOutputRegressor によって、これらの特徴量から
  隠れ状態ベクトル (例: hidden_size=2560) を数値推定する。
- 推定結果 (.npy) は visualize_mamba_hidden_states.py にそのまま渡せる。

セットアップ（macOS で初回のみ）
-------------------------------
    pip install numpy scikit-learn joblib xgboost
    pip install "ginza[ja_ginza]"            # GiNZA + 日本語モデル

実行例（Samber フォルダーからのコピペ用）
-----------------------------------------
    # 1) 訓練データ作成（既存: granite.py。日本語文ごとに末尾トークンの隠れ状態を保存）
    #    python3 granite.py --model ./mamba-2.8b-q4_k_m.gguf \\
    #        --texts "こんにちは世界" "Mamba は状態空間モデルです" \\
    #        --output train_hidden.npy
    #    各行と対応する文を train_texts.txt に 1 行ずつ記述する。

    # 2) 学習
    python3 xgboost_hidden_state_estimator.py train \\
        --latents ./train_hidden.npy --texts-file ./train_texts.txt \\
        --output ./hidden_state_model.joblib --test-size 0.0

    # 3) 推定（日本語プロンプトを受け取り隠れ状態を予測）
    python3 xgboost_hidden_state_estimator.py predict \\
        --model ./hidden_state_model.joblib \\
        --prompt "Mamba は状態空間モデルである。" --output ./predicted_hidden.npy

    # 4) 可視化（既存: visualize_mamba_hidden_states.py に結果を渡す）
    #    python3 visualize_mamba_hidden_states.py --hidden_npy ./predicted_hidden.npy \\
    #        --tokens-npy ./last_token_features_token_ids.npy --print-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _require(package: str, pip: str) -> None:
    import importlib.util

    if importlib.util.find_spec(package) is None:
        raise SystemExit(f"{package} がありません。インストールしてください:\n    {pip}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="xgboost_hidden_state_estimator",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="隠れ状態ラベル付きデータで XGBoost を学習")
    train.add_argument("--latents", type=Path, required=True,
                       help="granite.py が保存した隠れ状態 (.npy)。shape [n_texts, hidden_size]")
    train.add_argument("--texts-file", type=Path, required=True,
                       help="UTF-8 テキストファイル。1 行 = プロンプト 1 文で --latents の行と対応")
    train.add_argument("--output", type=Path, default=Path("hidden_state_model.joblib"),
                       help="学習済みパイプラインの保存先 (joblib)")
    train.add_argument("--test-size", type=float, default=0.2,
                       help="検証用に分離する割合 (0.0 なら全データで学習し検証なし)")
    train.add_argument("--n-estimators", type=int, default=300)
    train.add_argument("--max-depth", type=int, default=6)
    train.add_argument("--learning-rate", type=float, default=0.1)
    train.add_argument("--subsample", type=float, default=0.9)
    train.add_argument("--colsample", type=float, default=0.8,
                       help="colsample_bytree")
    train.add_argument("--min-child-weight", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--n-gram-max", type=int, default=3,
                       help="結合する文字 n-gram の最大次数 (1..n)")

    predict = sub.add_parser("predict", help="日本語プロンプトの隠れ状態を XGBoost で推定")
    predict.add_argument("--model", type=Path, required=True, help="train で作った .joblib")
    prompt_group = predict.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", nargs="+", help="推定する日本語プロンプト（複数可）")
    prompt_group.add_argument("--prompt-file", type=Path,
                              help="推定するプロンプト群。1 行 = 1 文")
    predict.add_argument("--output", type=Path, default=Path("predicted_hidden.npy"),
                         help="推定結果の保存先 (.npy)。shape [n_prompts, hidden_size]")
    predict.add_argument("--png", type=Path, default=None,
                         help="指定時、推定結果を PCA で 2 次元に圧縮して散布図として保存")
    predict.add_argument("--show-top", type=int, default=0,
                         help="推定結果に寄与した上位トークン語彙を文ごとに表示する数 (0 = 非表示)")
    return parser.parse_args()


# ---------------------------------------------------------------- GiNZA
def load_ginza():
    _require("spacy", 'pip install "ginza[ja_ginza]"')
    import spacy

    for name in ("ja_ginza", "ja_ginza_electra"):
        try:
            return spacy.load(name)
        except Exception:
            continue
    raise SystemExit("ja_ginza モデルが見つかりません。\n    pip install \"ginza[ja_ginza]\"")


def _reading(token) -> str:
    """GiNZA 5.x の API 差異に対応。token._.reading が無ければ ginza.reading_form を使う。"""
    try:
        return token._.reading
    except Exception:
        pass
    try:
        import ginza

        return ginza.reading_form(token, True)
    except Exception:
        return token.orth_


def _lemma(token) -> str:
    try:
        import ginza

        return ginza.lemma_(token) or token.lemma_
    except Exception:
        return token.lemma_ or token.orth_


def ginza_analyzer(nlp):
    """GiNZA で形態素解析し、カウント対象の語彙トークンを返す関数を生成する。"""

    def analyze(text: str) -> list[str]:
        tokens: list[str] = []
        for token in nlp(text):
            tokens.append(f"surf:{token.orth_}")
            tokens.append(f"pos:{token.pos_}")
            lemma = _lemma(token)
            if lemma:
                tokens.append(f"lemma:{lemma}")
            reading = _reading(token)
            if reading:
                tokens.append(f"read:{reading}")
        return tokens

    return analyze


def extract_morphological_features(nlp, texts: list[str]) -> list:
    """文ごとに形態素出現回数の辞書を返す（寄与確認・デバッグ用）。"""
    import collections

    rows = []
    for text in texts:
        counter: collections.Counter[str] = collections.Counter()
        for token in nlp(text):
            counter[f"surf:{token.orth_}"] += 1
            counter[f"pos:{token.pos_}"] += 1
            lemma = _lemma(token)
            if lemma:
                counter[f"lemma:{lemma}"] += 1
            reading = _reading(token)
            if reading:
                counter[f"read:{reading}"] += 1
        rows.append(counter)
    return rows


# ---------------------------------------------------------------- データ読み込み
def load_texts(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"テキストファイルが見つかりません: {path}")
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise SystemExit(f"テキストファイルが空です: {path}")
    return lines


def load_latents(path: Path) -> np.ndarray:
    if not path.is_file():
        raise SystemExit(f"隠れ状態ファイルが見つかりません: {path}")
    latents = np.load(path)
    if latents.ndim == 3:
        latents = latents[-1]
    if latents.ndim != 2 or latents.shape[1] == 0:
        raise SystemExit(
            f"隠れ状態は [n_texts, hidden_size] を想定していますが {latents.shape} でした。"
            " granite.py の出力（末尾トークン特徴量）を渡してください。"
        )
    return np.asarray(latents, dtype=np.float32)


# ---------------------------------------------------------------- 学習
def build_pipeline_text(analyzer, n_gram_max: int):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        transformer_list=[
            (
                "morph",
                CountVectorizer(
                    tokenizer=analyzer,
                    lowercase=False,
                    token_pattern=None,
                ),
            ),
            (
                "char",
                CountVectorizer(
                    analyzer="char_wb",
                    ngram_range=(1, n_gram_max),
                    lowercase=False,
                    token_pattern=None,
                ),
            ),
        ]
    )


def cmd_train(args: argparse.Namespace) -> None:
    _require("xgboost", "pip install xgboost")
    _require("sklearn", "pip install scikit-learn")
    _require("joblib", "pip install joblib")

    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import r2_score
    import joblib
    import xgboost as xgb

    nlp = load_ginza()
    texts = load_texts(args.texts_file)
    latents = load_latents(args.latents)

    if len(texts) < 2:
        raise SystemExit("学習には最低 2 文必要です。granite.py で隠れ状態データを増やしてください。")
    if args.test_size < 0 or args.test_size >= 1:
        raise SystemExit("--test-size は 0 以上 1 未満の値にしてください（0.0 で検証なし）")

    print(f"学習データ: {len(texts)} 文 / 隠れ状態 次元 {latents.shape[1]}")
    print("GiNZA で形態素解析し、特徴量を構築しています…")

    pipeline = build_pipeline_text(ginza_analyzer(nlp), args.n_gram_max)
    X_raw = pipeline.fit_transform(texts)
    print(f"特徴量行列: {X_raw.shape} (文数, 語彙数)  | 疎行列スパース率 "
          f"{(1 - X_raw.count_nonzero() / (X_raw.shape[0] * X_raw.shape[1])):.2%}")

    if args.test_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_raw, latents, test_size=args.test_size, random_state=args.seed
        )
    else:
        X_train, X_val, y_train, y_val = X_raw, None, latents, None

    model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample,
            min_child_weight=args.min_child_weight,
            n_jobs=-1,
            random_state=args.seed,
        ),
        n_jobs=-1,
    )

    print(f"XGBoost を学習中（{len(texts)} 文 → {latents.shape[1]} 次元を多出力回帰）…")
    model.fit(X_train, y_train)

    if X_val is not None and len(X_val):
        pred = model.predict(X_val)
        r2 = float(np.mean(r2_score(y_val, pred, multioutput="raw_values")))
        rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
        print(f"検証 R^2 = {r2:.4f}  / RMSE = {rmse:.4f}")

    artifact = {
        "vectorizer": pipeline,
        "model": model,
        "hidden_size": int(latents.shape[1]),
        "n_train": len(texts),
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "n_gram_max": args.n_gram_max,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, str(args.output), compress=3)
    print(f"学習済みパイプラインを保存: {args.output}")


# ---------------------------------------------------------------- 推定
def cmd_predict(args: argparse.Namespace) -> None:
    _require("joblib", "pip install joblib")
    _require("xgboost", "pip install xgboost")
    import joblib

    if not args.model.is_file():
        raise SystemExit(f"モデルが見つかりません: {args.model}")

    artifact = joblib.load(str(args.model))
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    nlp = load_ginza()
    if args.prompt:
        texts = list(args.prompt)
    else:
        texts = load_texts(args.prompt_file)
    if not texts:
        raise SystemExit("推定するプロンプトがありません。")

    print(f"GiNZA で {len(texts)} 文を解析し、XGBoost で隠れ状態を推定しています…")
    X = vectorizer.transform(texts)
    predicted = np.asarray(model.predict(X), dtype=np.float32)

    if predicted.ndim == 1:
        predicted = predicted.reshape(1, -1)
    if predicted.shape[1] != artifact["hidden_size"]:
        raise SystemExit(
            f"モデルの隠れ状態次元が一致しません: モデル={artifact['hidden_size']}, 予測={predicted.shape[1]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, predicted)
    print(f"推定結果を保存: {args.output}  (shape {predicted.shape}, dtype {predicted.dtype})")
    print(f"mean={predicted.mean():.6g} std={predicted.std():.6g} "
          f"min={predicted.min():.6g} max={predicted.max():.6g}")

    if args.show_top > 0 and hasattr(vectorizer, "get_feature_names_out"):
        import collections

        names = list(vectorizer.get_feature_names_out())
        morph_idx = [k for k, n in enumerate(names) if n.startswith("morph__")]
        if morph_idx:
            morph_block = X[:, morph_idx]
            if hasattr(morph_block, "toarray"):
                morph_dense = morph_block.toarray()
            else:
                morph_dense = np.asarray(morph_block)
            for i, text in enumerate(texts):
                counts: collections.Counter[str] = collections.Counter()
                row = morph_dense[i]
                for k, value in zip(morph_idx, row):
                    if value:
                        counts[names[k][len("morph__"):]] = int(value)
                print(f"\n[{i}] {text!r}")
                for name, count in counts.most_common(args.show_top):
                    print(f"    {count:>2}  {name}")

    if args.png:
        _require("matplotlib", "pip install matplotlib")
        _require("sklearn", "pip install scikit-learn")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=0).fit_transform(predicted)
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        ax.scatter(xy[:, 0], xy[:, 1], s=60, c=np.arange(len(texts)), cmap="viridis",
                   alpha=0.9, edgecolors="white", linewidths=0.35)
        for i, (x, y) in enumerate(xy):
            ax.annotate(f"{i}: {texts[i]!r}", (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.set_title(f"XGBoost で推定した隠れ状態 ({len(texts)} 文, {artifact['hidden_size']} dim)")
        ax.grid(alpha=0.2)
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=220, bbox_inches="tight")
        print(f"散布図を保存: {args.png}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        raise SystemExit(f"不明なコマンド: {args.command}")


if __name__ == "__main__":
    main()