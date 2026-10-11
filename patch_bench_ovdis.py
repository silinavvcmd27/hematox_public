import pathlib

p = pathlib.Path("sthelar_bench.py")
s = p.read_text(encoding="utf-8")

lines = s.split("\n")
i = next(i for i, l in enumerate(lines)
         if l.strip() == "if nov[s] < args.save_overlays:")
indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
lines[i] = (indent + "disagree = float(((p1 == 1) != (g1 == STR)).mean())\n"
            + indent + "if nov[s] < args.save_overlays "
            + "and disagree >= args.overlay_disagree:")

j = next(j for j, l in enumerate(lines) if l.strip() == "args = ap.parse_args()")
lines.insert(j, '    ap.add_argument("--overlay-disagree", type=float, default=0.0,\n'
                '                    help="сохранять оверлеи только там, где доля '
                'расхождений не ниже этой")')

pathlib.Path("sthelar_bench.py").write_text("\n".join(lines), encoding="utf-8")
print("правка внесена")
