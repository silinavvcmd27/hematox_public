#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure_model.py — схема архитектуры hematox_spatial в SVG.

Запуск:      python make_figure_model.py --out figure_model.svg
С картинками: python make_figure_model.py --out figure_model.svg --img-dir outputs/figure/panels

Файл на выходе редактируется в Inkscape: текст — настоящие <text>, а не кривые,
блоки сгруппированы и подписаны (панель видна в диалоге "Объекты"),
картинки вставляются как ссылки на PNG (Inkscape: "Внедрить изображения",
если нужен самодостаточный файл).
"""
import argparse, os, html

# ----------------------------------------------------------------------------- палитра
C = dict(
    a_fill="#EAF3FB", a_line="#2E75B6",     # панель a — разметка
    b_fill="#FDF6E7", b_line="#BF8F00",     # панель b — признаки и граф
    c_fill="#FBECEC", c_line="#C0392B",     # панель c — обучение
    d_fill="#EDF7EE", d_line="#2E8B57",     # панель d — применение
    box="#FFFFFF", boxline="#8A8A8A",
    txt="#1A1A1A", sub="#4D4D4D",
    tumor="#C0392B", horm="#E8873A", matr="#F0A030",
    imm="#2E86C1", vasc="#7D3C98", other="#909090",
    frozen="#2E86C1", trained="#C0392B",
    grey="#B0B0B0",
)
FONT = "Arial, Helvetica, sans-serif"

# ----------------------------------------------------------------------------- примитивы
class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []
        self.depth = 0
    def add(self, s): self.parts.append("  " * self.depth + s)
    def group(self, label):
        label = html.escape(label, quote=True)
        self.add(f'<g inkscape:groupmode="layer" inkscape:label="{label}">' if self.depth == 0
                 else f'<g inkscape:label="{label}">')
        self.depth += 1
    def endgroup(self):
        self.depth -= 1; self.add("</g>")
    def rect(self, x, y, w, h, fill, stroke=None, sw=1.4, rx=8, dash=None, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ' stroke="none"'
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" fill-opacity="{op}"{st}{d}/>')
    def line(self, x1, y1, x2, y2, stroke, sw=1.4, dash=None, marker=True):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arrow)"' if marker else ""
        self.add(f'<path d="M {x1:.1f},{y1:.1f} L {x2:.1f},{y2:.1f}" fill="none" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')
    def elbow(self, pts, stroke, sw=1.4, marker=True):
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        m = ' marker-end="url(#arrow)"' if marker else ""
        self.add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
                 f'stroke-linejoin="round"{m}/>')
    def text(self, x, y, t, size=12, weight="normal", fill=None, anchor="start",
             style="", ls=1.35):
        fill = fill or C["txt"]
        lines = t.split("\n")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"'
                 + (f' style="{style}"' if style else "") + '>')
        self.depth += 1
        for i, ln in enumerate(lines):
            dy = 0 if i == 0 else size * ls
            self.add(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{html.escape(ln)}</tspan>')
        self.depth -= 1
        self.add("</text>")
    def circle(self, cx, cy, r, fill, stroke=None, sw=1.2):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ' stroke="none"'
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{st}/>')
    def image(self, x, y, w, h, href):
        self.add(f'<image x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'xlink:href="{html.escape(href)}" preserveAspectRatio="xMidYMid slice"/>')
    def dump(self):
        head = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
                f'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
                f'width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
                f'version="1.1">\n'
                f'<defs>\n'
                f'  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">\n'
                f'    <path d="M 0,0 L 10,5 L 0,10 z" fill="context-stroke"/>\n'
                f'  </marker>\n'
                f'  <marker id="arrowg" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">\n'
                f'    <path d="M 0,0 L 10,5 L 0,10 z" fill="#6E6E6E"/>\n'
                f'  </marker>\n'
                f'</defs>\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"

# --------------------------------------------------- измерение ширины текста и перенос
# Если есть matplotlib — меряем настоящей метрикой шрифта; иначе консервативная оценка.
try:
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    _FPCACHE = {}

    def tw(t, size, weight="normal"):
        """ширина строки в px при данном кегле"""
        if not t:
            return 0.0
        key = (round(size, 2), weight)
        if key not in _FPCACHE:
            _FPCACHE[key] = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        return TextPath((0, 0), t, prop=_FPCACHE[key]).get_extents().width
except Exception:                                                    # без matplotlib
    def tw(t, size, weight="normal"):
        return len(t) * size * (0.63 if weight == "bold" else 0.60)


def wrap(t, size, width, weight="normal"):
    out, cur = [], ""
    for w_ in t.split():
        cand = (cur + " " + w_).strip()
        if tw(cand, size, weight) <= width or not cur:
            cur = cand
        else:
            out.append(cur); cur = w_
    if cur: out.append(cur)
    return "\n".join(out)


def shrink(t, size, width, weight="normal", floor=8.5):
    """уменьшает кегль, пока строка не влезет в ширину"""
    while size > floor and tw(t, size, weight) > width:
        size -= 0.25
    return size

# ------------------------------------------------------------------------------ иконки
def snowflake(g, cx, cy, r=9, col=None):
    col = col or C["frozen"]
    g.group("значок заморожен")
    for a in (0, 60, 120):
        import math
        dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
        g.line(cx - dx, cy - dy, cx + dx, cy + dy, col, 1.8, marker=False)
        for s in (-1, 1):
            ex, ey = cx + s * dx, cy + s * dy
            for b in (a + 140, a - 140):
                g.line(ex, ey, ex + 0.42 * r * math.cos(math.radians(b)),
                       ey + 0.42 * r * math.sin(math.radians(b)), col, 1.6, marker=False)
    g.endgroup()

def flame(g, cx, cy, r=9, col=None):
    col = col or C["trained"]
    g.group("значок обучается")
    g.add(f'<path d="M {cx:.1f},{cy-r:.1f} '
          f'C {cx+r*0.9:.1f},{cy-r*0.25:.1f} {cx+r*0.55:.1f},{cy+r*0.15:.1f} {cx+r*0.5:.1f},{cy+r*0.5:.1f} '
          f'C {cx+r*0.45:.1f},{cy+r:.1f} {cx-r*0.45:.1f},{cy+r:.1f} {cx-r*0.5:.1f},{cy+r*0.5:.1f} '
          f'C {cx-r*0.6:.1f},{cy:.1f} {cx-r*0.2:.1f},{cy-r*0.2:.1f} {cx:.1f},{cy-r:.1f} Z" '
          f'fill="{col}" stroke="none"/>')
    g.add(f'<path d="M {cx:.1f},{cy-r*0.15:.1f} '
          f'C {cx+r*0.4:.1f},{cy+r*0.2:.1f} {cx+r*0.2:.1f},{cy+r*0.75:.1f} {cx:.1f},{cy+r*0.8:.1f} '
          f'C {cx-r*0.25:.1f},{cy+r*0.75:.1f} {cx-r*0.35:.1f},{cy+r*0.3:.1f} {cx:.1f},{cy-r*0.15:.1f} Z" '
          f'fill="#FFFFFF" fill-opacity="0.75" stroke="none"/>')
    g.endgroup()

def stepnum(g, x, y, n, col):
    g.circle(x, y, 12.5, col, "#FFFFFF", 1.6)
    g.text(x, y + 4.6, str(n), 13, "bold", "#FFFFFF", "middle")

def placeholder(g, x, y, w, h, label, fname, img_dir=None, tone="#F2F2F2"):
    """Заглушка под картинку; если файл найден — вставляется он."""
    path = os.path.join(img_dir, fname) if img_dir else None
    if path and os.path.exists(path):
        g.group(f"картинка {fname}")
        g.image(x, y, w, h, path)
        g.rect(x, y, w, h, "none", C["boxline"], 1.2, rx=4)
        g.endgroup()
        return True
    g.group(f"ЗАГЛУШКА {fname}")
    g.rect(x, y, w, h, tone, C["boxline"], 1.3, rx=4, dash="5,4")
    g.line(x, y, x + w, y + h, "#D2D2D2", 1.0, marker=False)
    g.line(x + w, y, x, y + h, "#D2D2D2", 1.0, marker=False)
    lab = wrap(label, 11.5, w - 14)
    nl = len(lab.split("\n"))
    g.text(x + w / 2, y + h / 2 - (nl - 1) * 7.8 + 4, lab, 11.5, "bold", C["sub"], "middle")
    g.text(x + w / 2, y + h - 8, fname, 9.5, "normal", "#9A9A9A", "middle")
    g.endgroup()
    return False


# ============================================================================= схема
def build(img_dir=None):
    W = 1500
    M = 22
    g = SVG(W, 1660)
    PX, PW = M, W - 2 * M

    # ------------------------------------------------------------------ шапка
    g.group("шапка")
    title = ("Клеточный граф на признаках UNI: сегментация пяти зон микроокружения "
             "опухоли яичника по H&E")
    ts = shrink(title, 20, PW - 300, "bold")
    g.text(PX + 4, 30, title, ts, "bold")
    g.text(PX + 4, 52, "обучение по меткам пространственной транскриптомики Xenium · "
                       "применение к обычному H&E без транскриптомики", 13, "normal", C["sub"])
    # значки — справа, на второй строке, чтобы не налезать на заголовок
    lx0 = W - M - 4 - (tw("обучается", 12) + 22)
    flame(g, lx0 - 9, 48); g.text(lx0 + 6, 53, "обучается", 12, "normal", C["sub"])
    lx1 = lx0 - 24 - (tw("заморожено", 12) + 22)
    snowflake(g, lx1 - 9, 48); g.text(lx1 + 6, 53, "заморожено", 12, "normal", C["sub"])
    g.endgroup()

    # ================================================================== ПАНЕЛЬ a
    ay, ah = 66, 352
    g.group("панель a — разметка")
    g.rect(PX, ay, PW, ah, C["a_fill"], C["a_line"], 1.6, rx=10)
    g.text(PX + 16, ay + 27, "a", 19, "bold", C["a_line"])
    g.text(PX + 44, ay + 27, "Разметка-истина из пространственной транскриптомики "
                             "(выполняется один раз, на обучающих срезах)", 15, "bold")

    cw, chh, cy = 300, 152, ay + 44
    xs = [40, 400, 760, 1120]
    cards = [
        (1, "H&E-срез", "0.27 мкм/px, 3 среза яичника", "a1_he_thumb.png", "H&E обучающего среза"),
        (2, "Метки клеток Xenium", "332 000 клеток, тип по экспрессии", "a2_cells_thumb.png",
         "точки клеток, цвет = тип"),
        (3, "Совмещение координат", "аффинное + нежёсткое, Dice 0.99", "a4_zoom_cells.png",
         "наложение ST на H&E"),
        (4, "Маска-истина", "Вороной: пиксель = класс ближайшей клетки", "a5_voronoi.png",
         "маска пяти зон"),
    ]
    for (n, t, sub, fn, lab), x in zip(cards, xs):
        g.group(f"a{n} {t}")
        placeholder(g, x, cy, cw, chh, lab, fn, img_dir)
        stepnum(g, x + 15, cy - 8, n, C["a_line"])
        g.text(x, cy + chh + 20, t, 13.5, "bold")
        g.text(x, cy + chh + 37, wrap(sub, 11.5, cw), 11.5, "normal", C["sub"])
        g.endgroup()
    for x in xs[:-1]:
        g.line(x + cw + 10, cy + chh / 2, x + cw + 50, cy + chh / 2, "#6E6E6E", 2.0)

    # легенда классов
    ly = ay + 292
    g.group("легенда зон")
    g.text(40, ly + 4, "Пять зон:", 12.5, "bold")
    leg = [("опухоль", C["tumor"]), ("строма гормональная", C["horm"]),
           ("строма матриксная", C["matr"]), ("иммунные клетки", C["imm"]),
           ("сосуды и прочее", C["vasc"])]
    lx = 48 + tw("Пять зон:", 12.5, "bold") + 22
    for name, col in leg:
        g.rect(lx, ly - 8, 15, 15, col, "#FFFFFF", 1.0, rx=3)
        g.text(lx + 21, ly + 4, name, 12, "normal")
        lx += 21 + tw(name, 12) + 28
    g.endgroup()
    g.text(40, ay + 330, "Тип клетки известен из экспрессии генов, а не размечен патологом: "
                         "подтипы стромы на H&E глазом неразличимы, поэтому экспертной разметки "
                         "для этой задачи не существует.", 12, "normal", C["sub"])
    g.endgroup()

    # ================================================================== ПАНЕЛЬ b
    by, bh = ay + ah + 14, 322
    g.group("панель b — признаки и граф")
    g.rect(PX, by, PW, bh, C["b_fill"], C["b_line"], 1.6, rx=10)
    g.text(PX + 16, by + 27, "b", 19, "bold", C["b_line"])
    g.text(PX + 44, by + 27, "Признаки клеток и построение графа", 15, "bold")

    ry, rh = by + 46, 150
    # 5 — патчи
    g.group("b5 нарезка на патчи")
    placeholder(g, 40, ry, 240, rh, "патчи на срезе", "b1_patch_grid.png", img_dir)
    stepnum(g, 55, ry - 8, 5, C["b_line"])
    g.text(40, ry + rh + 20, "Нарезка на патчи", 13.5, "bold")
    g.text(40, ry + rh + 37, "256 px = 70 мкм, сетка 14 × 14 токенов", 11.5, "normal", C["sub"])
    g.endgroup()
    g.line(292, ry + rh / 2, 332, ry + rh / 2, "#6E6E6E", 2.0)

    # 6 — UNI
    g.group("b6 UNI")
    g.rect(344, ry, 240, rh, C["box"], C["boxline"], 1.4)
    g.text(464, ry + 34, "UNI", 22, "bold", "#2E86C1", "middle")
    g.text(464, ry + 56, "ViT-L/16", 13, "normal", C["sub"], "middle")
    for i in range(6):
        g.rect(374 + i * 30, ry + 72, 20, 46, "#D6E6F5", "#8FB8DC", 1.0, rx=3)
    g.text(464, ry + 136, "303 млн весов · заморожен", 11.5, "bold", C["sub"], "middle")
    snowflake(g, 566, ry + 16)
    stepnum(g, 359, ry - 8, 6, C["b_line"])
    g.text(344, ry + rh + 20, "Кодирование патча", 13.5, "bold")
    g.text(344, ry + rh + 37, "обучен на 100 000+ гистологических слайдов", 11.5, "normal", C["sub"])
    g.endgroup()
    g.line(596, ry + rh / 2, 636, ry + rh / 2, "#6E6E6E", 2.0)

    # 7 — клетка наследует токен
    g.group("b7 клетка наследует токен")
    g.rect(648, ry, 240, rh, C["box"], C["boxline"], 1.4)
    gx, gy, cell = 668, ry + 22, 20
    for i in range(6):
        for j in range(5):
            g.rect(gx + i * cell, gy + j * cell, cell, cell, "#F7F7F7", "#DADADA", 0.8, rx=0)
    pts = [(0.5, 0.4), (1.6, 1.2), (2.4, 0.7), (3.7, 2.3), (4.4, 1.5), (1.2, 3.4),
           (2.8, 3.8), (4.9, 3.2), (3.1, 1.9), (5.4, 4.3)]
    for px_, py_ in pts:
        g.circle(gx + px_ * cell + cell / 2, gy + py_ * cell + cell / 2, 4.2, "#C0392B",
                 "#FFFFFF", 1.0)
    g.text(768, ry + 140, "1 токен ≈ 5 мкм < ядра", 11.5, "bold", C["sub"], "middle")
    stepnum(g, 663, ry - 8, 7, C["b_line"])
    g.text(648, ry + rh + 20, "Клетка наследует токен", 13.5, "bold")
    g.text(648, ry + rh + 37, "UNI считается один раз на патч, не на клетку", 11.5, "normal", C["sub"])
    g.endgroup()
    g.line(900, ry + rh / 2, 940, ry + rh / 2, "#6E6E6E", 2.0)

    # 8 — граф
    g.group("b8 построение графа")
    placeholder(g, 952, ry, 240, rh, "граф клеток", "b2_graph_zoom.png", img_dir)
    stepnum(g, 967, ry - 8, 8, C["b_line"])
    g.text(952, ry + rh + 20, "Граф клеток", 13.5, "bold")
    g.text(952, ry + rh + 37, "ребро ≤ 50 мкм, k = 8 соседей", 11.5, "normal", C["sub"])
    g.endgroup()

    # блок с числами справа
    g.group("b числа")
    g.rect(1216, ry, 240, rh, "#FFFFFF", C["b_line"], 1.3, rx=6)
    g.text(1336, ry + 26, "На срез", 13, "bold", C["b_line"], "middle")
    for i, (k, v) in enumerate([("узлов", "288 000"), ("рёбер", "4.6 млн"),
                                ("признак узла", "1024"), ("контекст", "≈ 100 мкм")]):
        g.text(1236, ry + 52 + i * 24, k, 12, "normal", C["sub"])
        g.text(1436, ry + 52 + i * 24, v, 12, "bold", C["txt"], "end")
    g.endgroup()
    g.text(40, by + bh - 14, "Узел графа — клетка, а не патч: решение о клетке принимается "
                             "с учётом соседей, а не по текстуре в одной точке.",
           12, "normal", C["sub"])
    g.endgroup()


    # ================================================================== ПАНЕЛЬ c
    cy0, ch = by + bh + 14, 466
    g.group("панель c — две графовые сети")
    g.rect(PX, cy0, PW, ch, C["c_fill"], C["c_line"], 1.6, rx=10)
    g.text(PX + 16, cy0 + 27, "c", 19, "bold", C["c_line"])
    g.text(PX + 44, cy0 + 27, "Две графовые сети поверх замороженных признаков", 15, "bold")
    _t = "обучается 1.06 млн параметров — 0.35 % от 303 млн UNI"
    _x = W - M - 16 - tw(_t, 12.5, "bold")
    flame(g, _x - 16, cy0 + 20)
    g.text(_x, cy0 + 25, _t, 12.5, "bold", C["c_line"])

    def chain(y, n, title, sub, out_dim, out_names, param, imgfile, imglab, metrics):
        g.group(f"c{n} {title}")
        stepnum(g, 55, y + 6, n, C["c_line"])
        g.text(76, y + 11, title, 14, "bold")
        g.text(76 + tw(title, 14, "bold") + 16, y + 11, "· " + sub, 12, "normal", C["sub"])
        # цепочка блоков
        bx, byy, bhh = 40, y + 40, 54
        blocks = [("вход узла", "1024"), ("Linear", "1024 → 256"),
                  ("GraphConv", "× 2, скрытый 256"), ("Linear", f"256 → {out_dim}")]
        ws = [104, 112, 150, 112]
        for (t, st), w_ in zip(blocks, ws):
            hl = t == "GraphConv"
            g.rect(bx, byy, w_, bhh, "#FBE3E1" if hl else C["box"],
                   C["c_line"] if hl else C["boxline"], 1.6 if hl else 1.3, rx=6)
            g.text(bx + w_ / 2, byy + 22, t, 12.5, "bold" if hl else "normal",
                   C["c_line"] if hl else C["txt"], "middle")
            g.text(bx + w_ / 2, byy + 39, st, 11, "normal", C["sub"], "middle")
            if t != blocks[-1][0]:
                g.line(bx + w_ + 3, byy + bhh / 2, bx + w_ + 21, byy + bhh / 2, "#6E6E6E", 1.8)
            bx += w_ + 24
        g.text(40, byy + bhh + 18, param, 11.5, "normal", C["sub"])
        # выход — цветные метки классов, в столбик справа от цепочки
        ox = bx + 2
        g.text(ox, byy - 4, "выход", 11.5, "bold", C["sub"])
        for i, (nm, col) in enumerate(out_names):
            g.rect(ox, byy + 6 + i * 18, 12, 12, col, "#FFFFFF", 1.0, rx=3)
            g.text(ox + 18, byy + 16 + i * 18, nm, 11.5, "normal")
        # картинка результата
        ix, iw, ih = 748, 214, 128
        placeholder(g, ix, y + 20, iw, ih, imglab, imgfile, img_dir)
        g.text(ix, y + 164, "предсказание на валидационном срезе", 10.5, "normal", C["sub"])
        # таблица метрик: подпись строки сверху, пояснение — отдельной строкой ниже
        mx, mw = ix + iw + 24, W - M - 16 - (ix + iw + 24)
        rows = metrics[1]
        rh_ = 30 if all(r[2] for r in rows) else 24
        mh = 30 + len(rows) * rh_ + 6
        g.rect(mx, y + 20, mw, mh, "#FFFFFF", C["c_line"], 1.3, rx=6)
        g.text(mx + 14, y + 40, metrics[0], 12.5, "bold", C["c_line"])
        for i, (lab_, val, det) in enumerate(rows):
            yy = y + 60 + i * rh_
            vs = 11.8
            g.text(mx + mw - 14, yy, val, vs, "bold", C["txt"], "end")
            avail = mw - 28 - tw(val, vs, "bold") - 16
            ls = shrink(lab_, 11.8, avail)
            g.text(mx + 14, yy, lab_, ls, "normal", C["sub"])
            if det:
                ds = shrink(det, 10.5, mw - 28)
                g.text(mx + 14, yy + 13, det, ds, "normal", "#8A8A8A")
        g.endgroup()
        return y + 20 + mh

    chain(cy0 + 44, 9, "Слой 1 · опухоль или строма",
          "все узлы графа",
          2, [("опухоль", C["tumor"]), ("строма", "#9A9A9A")],
          "526 594 обучаемых параметра",
          "c1_layer1_pred.png", "карта опухоль/строма",
          ("Кросс-валидация по срезам (leave-one-slide-out)",
           [("граф клеток", "mIoU 0.836", "Dice: опухоль 0.92 · строма 0.90"),
            ("свёрточная голова U-Net", "mIoU 0.754", "те же признаки UNI, столько же параметров"),
            ("логистическая регрессия по одной клетке", "—", "базовый уровень, посчитать"),
            ("граф без рёбер (r = 0)", "—", "проверка вклада контекста")]))

    # стрелка между слоями
    g.line(300, cy0 + 224, 300, cy0 + 248, C["c_line"], 2.0)
    g.text(316, cy0 + 244, "на вход слоя 2 идут только стромальные узлы; "
                           "опухолевые исключены из обучения", 11.5, "normal", C["c_line"])

    chain(cy0 + 262, 10, "Слой 2 · подтип стромы",
          "подграф стромы",
          4, [("гормональная", C["horm"]), ("матриксная", C["matr"]),
              ("иммунные", C["imm"]), ("сосуды", C["vasc"])],
          "527 106 параметров · рецептивное поле 2 × 50 мкм ≈ 100 мкм",
          "c3_layer2_pred.png", "карта подтипов стромы",
          ("Dice по подтипам",
           [("внутри среза (пространственный сплит)", "0.31 – 0.71",
             "сосуды 0.71 · иммунные 0.68 · матриксная 0.65 · гормональная 0.31"),
            ("между пациентами, компартменты", "0.54 – 0.59", "CAF 0.59 · иммунный 0.55 · сосудистый 0.54"),
            ("между пациентами, деление CAF", "0.26", "граница текущего объёма разметки"),
            ("STHELAR, независимый датасет", "0.53 – 0.81", "сосудистый 0.81 · иммунный 0.76 · CAF 0.53")]))
    g.endgroup()


    # ================================================================== ПАНЕЛЬ d
    dy, dh = cy0 + ch + 14, 372
    g.group("панель d — применение")
    g.rect(PX, dy, PW, dh, C["d_fill"], C["d_line"], 1.6, rx=10)
    g.text(PX + 16, dy + 27, "d", 19, "bold", C["d_line"])
    g.text(PX + 44, dy + 27, "Применение к новому срезу — транскриптомика больше не нужна",
           15, "bold")

    sy, sh_ = dy + 46, 128
    steps = [
        (11, "Срез TCGA", "чтение .svs, нарезка", "d1_tcga_he.png", "H&E TCGA-OV", True),
        (12, "Нормализация окраски", "Macenko", None, None, False),
        (13, "Детекция ядер", "StarDist → координаты", "d2_nuclei.png", "ядра StarDist", True),
        (14, "Признаки UNI", "1 проход на патч", None, None, False),
        (15, "Слой 1 → слой 2", "класс каждой клетки", None, None, False),
        (16, "Карта пяти зон", "весь срез", "d2_zonemap.png", "карта зон TCGA", True),
    ]
    sx, sw_, gap = 40, 208, 30
    for i, (n, t, sub, fn, lab, isimg) in enumerate(steps):
        x = sx + i * (sw_ + gap)
        g.group(f"d{n} {t}")
        if isimg:
            placeholder(g, x, sy, sw_, sh_, lab, fn, img_dir)
        else:
            g.rect(x, sy, sw_, sh_, C["box"], C["boxline"], 1.4)
            if n == 12:
                for k, col in enumerate(["#E7B9C8", "#D9A6D0", "#C58FB8"]):
                    g.rect(x + 18 + k * 30, sy + 26, 26, 34, col, "#FFFFFF", 1.0, rx=3)
                g.line(x + 112, sy + 43, x + 132, sy + 43, "#6E6E6E", 1.6)
                for k in range(2):
                    g.rect(x + 140 + k * 30, sy + 26, 26, 34, "#D8A2BE", "#FFFFFF", 1.0, rx=3)
                g.text(x + sw_ / 2, sy + 84, "единый вид окраски", 11.5, "normal", C["sub"], "middle")
                g.text(x + sw_ / 2, sy + 102, "не зависит от лаборатории", 11.5, "normal",
                       C["sub"], "middle")
            elif n == 14:
                g.text(x + sw_ / 2, sy + 40, "UNI", 20, "bold", "#2E86C1", "middle")
                snowflake(g, x + sw_ / 2 + 46, sy + 34)
                for k in range(5):
                    g.rect(x + 40 + k * 26, sy + 56, 18, 38, "#D6E6F5", "#8FB8DC", 1.0, rx=3)
                g.text(x + sw_ / 2, sy + 112, "признак 1024 на клетку", 11.5, "normal",
                       C["sub"], "middle")
            elif n == 15:
                for k, (nm, col) in enumerate([("Слой 1", C["c_line"]), ("Слой 2", C["c_line"])]):
                    g.rect(x + 24, sy + 20 + k * 46, 160, 36, "#FBE3E1", col, 1.4, rx=5)
                    g.text(x + 104, sy + 43 + k * 46, nm, 13, "bold", col, "middle")
                flame(g, x + 190, sy + 16)
                g.line(x + 104, sy + 58, x + 104, sy + 64, C["c_line"], 1.6)
        stepnum(g, x + 15, sy - 8, n, C["d_line"])
        g.text(x, sy + sh_ + 20, t, 13, "bold")
        g.text(x, sy + sh_ + 37, wrap(sub, 11.5, sw_), 11.5, "normal", C["sub"])
        g.endgroup()
        if i < len(steps) - 1:
            g.line(x + sw_ + 4, sy + sh_ / 2, x + sw_ + gap - 4, sy + sh_ / 2, "#6E6E6E", 2.0)

    # 17 — показатели
    oy = sy + sh_ + 62
    g.group("d17 показатели среза")
    stepnum(g, 55, oy + 6, 17, C["d_line"])
    g.text(76, oy + 11, "Показатели среза", 14, "bold")
    g.rect(40, oy + 24, 700, 76, "#FFFFFF", C["d_line"], 1.3, rx=6)
    g.text(58, oy + 48, "TSR = строма / (опухоль + строма)", 13.5, "bold")
    g.text(58, oy + 68, "связан с химиорезистентностью при HGSC", 11.5, "normal", C["sub"])
    g.text(58, oy + 88, "согласие с ручной оценкой: ICC(2,1) и анализ Бланда—Альтмана",
           11.5, "normal", C["sub"])
    g.rect(760, oy + 24, 696, 76, "#FFFFFF", C["d_line"], 1.3, rx=6)
    g.text(778, oy + 48, "Состав стромы: доли четырёх подтипов", 13.5, "bold")
    comp = [(0.38, C["matr"], "матриксная"), (0.24, C["horm"], "гормональная"),
            (0.22, C["imm"], "иммунные"), (0.16, C["vasc"], "сосуды")]
    bx2, bw_ = 778, 500
    for frac, col, nm in comp:
        w_ = bw_ * frac
        g.rect(bx2, oy + 60, w_, 20, col, "#FFFFFF", 1.0, rx=0)
        bx2 += w_
    g.text(778, oy + 94, "показано схематично; реальные доли считаются по срезу",
           11, "normal", C["sub"])
    g.endgroup()
    g.endgroup()

    # ------------------------------------------------------------------ подвал
    g.group("подвал")
    g.text(PX + 4, dy + dh + 26,
           "Разделение стромы на гормонпродуцирующую и матриксную патолог по обычному H&E "
           "провести не может: модель восстанавливает его по морфологии и взаимному "
           "расположению клеток.", 12.5, "normal", C["sub"])
    g.endgroup()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figure_model.svg")
    ap.add_argument("--img-dir", default=None,
                    help="каталог с PNG-панелями; отсутствующие остаются заглушками")
    a = ap.parse_args()
    svg = build(a.img_dir)
    open(a.out, "w", encoding="utf-8").write(svg.dump())
    print("записано:", a.out)


if __name__ == "__main__":
    main()
