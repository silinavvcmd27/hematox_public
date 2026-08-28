# Вписать сгенерированные картинки в заглушки схемы figure_pipeline1.svg.
# Заглушки — верхнеуровневые <rect class="ph" id="rectNN"> с абсолютными x/y/w/h;
# на каждую кладём <image> тех же координат поверх всего (заглушка и её подпись
# оказываются перекрыты). Исходный SVG не трогаем, пишем *_final.svg.
#
# Запускать там, где рядом лежат и SVG, и три картинки (проще всего на сервере
# в outputs/figure/):
#   python embed_panels.py
#   python embed_panels.py --svg outputs/figure/figure_pipeline1.svg --dir outputs/figure

import argparse
import base64
import re
from pathlib import Path

# id заглушки -> файл картинки
MAP = {
    "rect99":  "c3_layer2_pred.png",   # результат слоя 2 (панель c)
    "rect115": "d1_tcga_he.png",       # H&E среза TCGA (панель d)
    "rect136": "d2_zonemap.png",       # карта зон (панель d)
}


def geom(svg, rid):
    """x/y/width/height рамки по её id (атрибуты у Inkscape на отдельных строках)."""
    m = re.search(r'<rect\b[^>]*\bid="%s"[^>]*?/>' % re.escape(rid), svg)
    if not m:
        return None
    el = m.group(0)
    vals = {}
    for k in ("x", "y", "width", "height"):
        mm = re.search(r'\b%s="([-\d.]+)"' % k, el)
        if not mm:
            return None
        vals[k] = mm.group(1)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", default="outputs/figure/figure_pipeline1.svg")
    ap.add_argument("--dir", default="outputs/figure", help="папка с картинками")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    svg = Path(args.svg).read_text(encoding="utf-8")
    imgs = []
    for rid, png in MAP.items():
        g = geom(svg, rid)
        if g is None:
            print(f"!! рамка {rid} не найдена — пропуск")
            continue
        p = Path(args.dir) / png
        if not p.exists():
            print(f"!! нет картинки {p} — пропуск {rid}")
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        imgs.append(
            '<image x="{x}" y="{y}" width="{width}" height="{height}" '
            'preserveAspectRatio="xMidYMid slice" '
            'xlink:href="data:image/png;base64,{b}"/>'.format(b=b64, **g))
        print(f"вписал {png} -> {rid} ({g['width']}x{g['height']} @ {g['x']},{g['y']})")

    if not imgs:
        raise SystemExit("нечего вставлять — проверь, что картинки на месте")

    svg = svg.replace("</svg>", "\n".join(imgs) + "\n</svg>", 1)
    out = args.out or str(Path(args.svg).with_name("figure_pipeline_final.svg"))
    Path(out).write_text(svg, encoding="utf-8")
    print("готово:", out, f"| вставлено {len(imgs)}/3")


if __name__ == "__main__":
    main()
