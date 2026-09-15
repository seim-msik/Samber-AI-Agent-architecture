# 日本語物語文の難易度予測パイプライン

約1000文字の日本語物語文を生成し、GGUF 形式の Mamba モデルでトークン単位の
隠れ状態を抽出、そこから特徴量を集約して決定木で難易度（1〜100）を予測します。

Apple Silicon (arm64) と Metal GPU のみを対象としています。

## 構成

| ファイル | 役割 |
| --- | --- |
| `generate_stories.py` | 日本語の物語文（約1000文字）を生成し `data/story_*.txt` に保存 |
| `extract_hidden.py` | llama-cpp-python (Metal) で隠れ状態を抽出し `data/hidden_states/story_*.npy` に保存 |
| `build_features.py` | 隠れ状態から特徴量を集約し、難易度ラベル付き `data/features.csv` を出力 |
| `train_tree.py` | 決定木回帰で難易度を学習し R2 スコアを表示 |
| `requirements.txt` | 依存パッケージ一覧 |
| `setup_metal.sh` | llama-cpp-python の Metal ビルドと GGUF モデル取得 |

中間ファイルはすべて `data/` に保存されます。

## セットアップ（Metal）

```bash
# 1) 環境確認
python -c "import platform; print(platform.machine())"   # arm64 であること

# 2) llama-cpp-python の Metal ビルド + モデル取得（一度だけ）
bash setup_metal.sh
```

`setup_metal.sh` は以下の処理を行います。

- `CMAKE_ARGS="-DGGML_METAL=on"` で llama-cpp-python をビルド（GPU は Metal）
- numpy / pandas / scikit-learn / ginza（ja_ginza モデル込み）をインストール
- `models/mamba-130m-hf-Q3_K_M.gguf` を取得

## 実行手順

```bash
# 1) 物語文を生成（例: 10 話）
python generate_stories.py --n 10

# 2) 隠れ状態を抽出（Metal GPU で実行。ログに Metal の文字列が表示される）
python extract_hidden.py

# 3) 特徴量集約 → features.csv
python build_features.py

# 4) 決定木で学習・評価 → R2 スコア表示
python train_tree.py
```

## 特徴量

各物語の隠れ状態 `[tokens, hidden_size]` から以下の特徴量を算出します。

- `n_tokens`: トークン数
- `mean_norm` / `norm_std`: トークンごとの L2 ノルムの平均・標準偏差
- `mean_mean` / `mean_var`: トークンごとの平均・分散
- `mean_max` / `mean_min`: トークンごとの最大値・最小値
- `sim_mean` / `sim_std` / `sim_min`: 隣接トークン間コサイン類似度の平均・標準偏差・最小値

難易度ラベルは GiNZA による平均文長と漢字比率から簡易的に 1〜100 の整数で算出します
（`build_features.py` 内 `difficulty_label()`）。

## 検証手順

```bash
bash setup_metal.sh                                # Metal ビルド成功
python generate_stories.py --n 10                  # data/story_0000..0009.txt
python extract_hidden.py                           # data/hidden_states/*.npy (10個)
python build_features.py                           # data/features.csv
python train_tree.py                               # R2 スコア表示
```

## 注意

- 依存パッケージは llama-cpp-python / numpy / pandas / scikit-learn / ginza のみです。
- モデルは GGUF 形式を使用し、GPU は Metal のみを使用します。
