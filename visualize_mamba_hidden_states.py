#!/usr/bin/env python3
"""Print and optionally plot final token positions in a Mamba hidden-state .npy file.

The expected array shape is [tokens, hidden_size].  Arrays with shape
[layers, tokens, hidden_size] are also accepted; choose a layer with --layer.

For an exact token/row correspondence, pass the token IDs produced during
hidden-state extraction through --tokens-npy.  --prompt + --gguf is convenient,
but its BOS/special-token settings must match the extractor exactly.

Samber フォルダーから末尾 100 行をターミナル表示するコピペ用コマンド:
    cd ~/Desktop/samber
    python3 visualize_mamba_hidden_states.py --hidden_npy ./last_token_features.npy --tokens-npy ./last_token_features_token_ids.npy --gguf ./mamba-2.8b-q4_k_m.gguf --print-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden_npy", type=Path, required=True, help="hidden states (.npy)")
    token_source = parser.add_mutually_exclusive_group(required=True)
    token_source.add_argument("--tokens-npy", type=Path,
                              help="1-D token IDs saved by the extraction script (recommended)")
    token_source.add_argument("--prompt", type=Path,
                              help="UTF-8 prompt text to tokenize with --gguf")
    parser.add_argument("--gguf", type=Path,
                        help="GGUF model, required when --prompt is used; optional for token decoding")
    parser.add_argument("--layer", type=int, default=-1,
                        help="layer index for a 3-D array (default: -1, final layer)")
    parser.add_argument("--last", type=int, default=100,
                        help="number of final token rows to print/draw (default: 100)")
    parser.add_argument("--output", type=Path, default=Path("hidden_tokens_last100.png"))
    parser.add_argument("--title", default=None)
    parser.add_argument("--annotate", action="store_true",
                        help="write a compact token label next to every point")
    parser.add_argument("--print-only", action="store_true",
                        help="only print the final tokens; do not create a PCA plot")
    parser.add_argument("--no-bos", action="store_true",
                        help="when using --prompt, do not prepend the BOS token")
    return parser.parse_args()


def select_states(path: Path, layer: int) -> np.ndarray:
    states = np.load(path)
    if states.ndim == 3:
        states = states[layer]
    if states.ndim != 2:
        raise ValueError(f"Expected [tokens, hidden] or [layers, tokens, hidden], got {states.shape}")
    return np.asarray(states, dtype=np.float32)


def load_tokenizer(gguf: Path):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit("Install llama-cpp-python: pip install llama-cpp-python") from exc
    # We only use the tokenizer.  n_ctx=1 keeps the otherwise unused context small.
    return Llama(model_path=str(gguf), n_ctx=1, logits_all=False, verbose=False)


def get_ids_and_labels(args: argparse.Namespace) -> tuple[np.ndarray, list[str]]:
    tokenizer = load_tokenizer(args.gguf) if args.gguf else None
    if args.tokens_npy:
        ids = np.load(args.tokens_npy).reshape(-1).astype(np.int64)
    else:
        if not args.gguf:
            raise SystemExit("--gguf is required together with --prompt")
        ids = np.asarray(tokenizer.tokenize(args.prompt.read_bytes(), add_bos=not args.no_bos), dtype=np.int64)

    if tokenizer is None:
        labels = [str(token_id) for token_id in ids]
    else:
        labels = []
        for token_id in ids:
            piece = tokenizer.detokenize([int(token_id)]).decode("utf-8", errors="replace")
            labels.append(piece.replace("\n", "\\n").replace("\t", "\\t"))
    return ids, labels


def print_tail(ids: np.ndarray, labels: list[str], start: int) -> None:
    """Render a terminal-friendly view of the final token rows."""
    print(f"\nFinal {len(ids) - start} tokens (position | token ID | token text)")
    print("-" * 78)
    for pos in range(start, len(ids)):
        print(f"{pos:>6} | {ids[pos]:>8} | {labels[pos]!r}")


def main() -> None:
    args = parse_args()
    states = select_states(args.hidden_npy, args.layer)
    ids, labels = get_ids_and_labels(args)
    if len(ids) != len(states):
        raise SystemExit(
            f"Token/state count differs: {len(ids)} token IDs vs {len(states)} hidden-state rows. "
            "Pass the exact IDs used by your extraction script via --tokens-npy, or match its BOS settings."
        )
    if args.last < 1:
        raise SystemExit("--last must be at least 1")

    start = max(0, len(states) - args.last)
    print_tail(ids, labels, start)
    if args.print_only:
        return

    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise SystemExit(
            "For a PNG plot install: pip install matplotlib scikit-learn\n"
            "Or use --print-only to display tokens in the terminal."
        ) from exc

    # Fit PCA over the complete sequence so the selected tail preserves its
    # position relative to the whole prompt; only the tail is displayed.
    xy = PCA(n_components=2, random_state=0).fit_transform(states)
    positions = np.arange(start, len(states))
    tail_xy = xy[start:]

    fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
    points = ax.scatter(tail_xy[:, 0], tail_xy[:, 1], c=positions, cmap="viridis",
                        s=50, alpha=0.9, edgecolors="white", linewidths=0.35)
    colorbar = fig.colorbar(points, ax=ax, pad=0.01)
    colorbar.set_label("Token position")
    for index, (x, y) in enumerate(tail_xy, start=start):
        label = f"{index}: {labels[index]!r}"
        ax.annotate(label if args.annotate else str(index), (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(args.title or f"Mamba hidden states: final {len(tail_xy)} token rows")
    ax.grid(alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"\nSaved PCA plot: {args.output}")


if __name__ == "__main__":
    main()
