#!/usr/bin/env python3
"""日本語の物語文（約1000文字）をランダム生成して data/ に保存する。

実行例:
    python generate_stories.py --n 10
    python generate_stories.py --n 20 --seed 42 --target 1200

出力:
    data/story_0000.txt, data/story_0001.txt, ...
    （1行目のコメント行は含めず、本文のみ UTF-8 で保存）
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

DEFAULT_TARGET_CHARS = 1000

NAMES = ["太郎", "花子", "健太", "美咲", "雄介", "さくら", "蓮", "桃花", "大和", "凛"]
PLACES = ["森", "山", "海辺", "古い町", "学校", "雪原", "田んぼ", "神社", "川辺", "桜の丘"]
THINGS = ["古い手紙", "青い小鳥", "土の中の宝箱", "錆びた時計", "赤い自転車",
          "迷い猫", "落ち葉", "雨傘", "星", "古びた本"]

EASY_SENTENCES = [
    "{name}は{place}に行った。",
    "{name}は{thing}を見つけた。",
    "{name}はとても嬉しかった。",
    "{name}は{thing}をゆっくり手に取った。",
    "空はとても青くて、{name}は気持ちよかった。",
    "「すごいね」と{name}は言った。",
    "{thing}は古いものだったが、{name}には大切なものだった。",
    "それから、{name}は家に帰った。",
    "{place}はとても静かだった。",
    "{name}は明日もそこへ行こうと思った。",
    "{name}は小さい声で「ありがとう」と言った。",
    "その日の夕方、{name}は{place}で休んだ。",
    "{thing}は{name}の横に置いてあった。",
    "風が吹いて、{thing}が動いた。",
    "{name}は笑って、{place}をあとにした。",
]

HARD_SENTENCES = [
    "{place}には言葉にならない静けさが満ちていた。",
    "{name}は深いため息をつきながら、窓の外に広がる{place}を眺めた。",
    "記憶の奥底に眠っていた{thing}が、ふいにその姿を現した。",
    "{name}の心に去来する想いは、容易に言葉へと置き換えられなかった。",
    "夜の帳が下りる頃、{place}の木立の影が長く伸びた。",
    "{thing}は時代の移ろいに耐え、なおその気配を保っていた。",
    "天の一角を裂くような雷鳴が、山あいの村へと響き渡った。",
    "{name}は長い沈黙の果てに、ようやく口を開いた。",
    "運命の歯車が静かに、しかし確実に回り始めたことを、誰もまだ知らない。",
    "古の伝承によれば、{place}の奥には封印された力が眠るという。",
    "{name}の足音だけが、白い雪の上に規則正しく刻まれていた。",
    "過ぎ去った季節の面影が、{place}の空気にひそかに残っていた。",
    "{name}の指先は{thing}の冷たい感触に、微かに震えた。",
    "海の彼方に沈む夕日が、波間を真紅に染め上げていた。",
    "{thing}をめぐる謎は深まるばかりで、{name}は一晩中考え込んだ。",
]


def make_sentence(rng: random.Random, style: int) -> str:
    """style 0=易, 1=やや易, 2=難 に応じて文テンプレートを選び、語彙を埋める。"""
    if style == 0:
        template = rng.choice(EASY_SENTENCES)
    elif style == 1:
        template = rng.choice(EASY_SENTENCES + HARD_SENTENCES[:6])
    else:
        template = rng.choice(HARD_SENTENCES)

    kwargs = {
        "name": rng.choice(NAMES),
        "place": rng.choice(PLACES),
        "thing": rng.choice(THINGS),
    }
    return template.format(**kwargs)


# 物語の難易度レベルごとのスタイルの重み (易, 中, 難)
LEVEL_STYLE_WEIGHTS = {
    0: (8, 1, 1),  # 易しい物語: ほとんど易文
    1: (1, 2, 1),  # 標準的な物語
    2: (1, 1, 8),  # 難しい物語: ほとんど難文
}
STORY_LEVEL_WEIGHTS = (3, 4, 3)


def make_story(rng: random.Random, target_chars: int) -> str:
    """目標文字数になるまで文を追加して物語文を作る。

    物語ごとに難易度レベルを選び、それに応じて文体を偏らせることで
    平均文長・漢字比率に差をつけ、難易度ラベル (1〜100) の幅を持たせる。
    """
    level = rng.choices((0, 1, 2), weights=STORY_LEVEL_WEIGHTS)[0]
    style_weights = LEVEL_STYLE_WEIGHTS[level]

    sentences: list[str] = []
    total = 0
    while total < target_chars:
        style = rng.choices((0, 1, 2), weights=style_weights)[0]
        sentence = make_sentence(rng, style)
        sentences.append(sentence)
        total += len(sentence)
    return "".join(sentences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="生成する物語の数 (default: 10)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_CHARS,
                        help="目標文字数 (default: 1000)")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード (default: 0)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="保存先ディレクトリ (default: data)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 1:
        raise SystemExit("--n は 1 以上にしてください。")

    out_dir = args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for i in range(args.n):
        text = make_story(rng, args.target)
        path = out_dir / f"story_{i:04d}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"generated {path.name}: {len(text)} 文字")


if __name__ == "__main__":
    main()
