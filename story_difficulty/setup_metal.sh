#!/usr/bin/env bash
# llama-cpp-python を Metal ビルドでインストールし、GGUF モデルを取得する。
# 前提: Apple Silicon (arm64) の macOS
set -euo pipefail

MODEL_URL="https://huggingface.co/tensorblock/mamba-130m-hf-GGUF/resolve/main/mamba-130m-hf-Q3_K_M.gguf"
MODEL_PATH="models/mamba-130m-hf-Q3_K_M.gguf"

arch="$(uname -m)"
if [[ "$arch" != "arm64" ]]; then
  echo "Apple Silicon (arm64) のみ対応しています: $arch" >&2
  exit 1
fi

# コンパイラ環境の確認（Metal ビルドには Xcode Command Line Tools と CMake が必要）
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools が見つかりません。'xcode-select --install' を実行してください。" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake が見つかりません。'brew install cmake' を実行してください。" >&2
  exit 1
fi

echo "== Python 依存パッケージ =="
python3 -m pip install --upgrade pip
# spacy/thinc は numpy<2 が必要なため固定する
python3 -m pip install "numpy<2" pandas "ginza[ja-ginza]"
python3 -m pip install scikit-learn

echo "== llama-cpp-python を Metal ビルド =="
CMAKE_ARGS="-DGGML_METAL=on" python3 -m pip install --upgrade --force-reinstall llama-cpp-python

echo "== GGUF モデル取得 =="
mkdir -p models
if [[ ! -f "$MODEL_PATH" ]]; then
  curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
else
  echo "モデルはすでにあります: $MODEL_PATH"
fi

echo "== 完了: Metal ビルド成功 =="
python3 -c "import platform; print('arch:', platform.machine())"