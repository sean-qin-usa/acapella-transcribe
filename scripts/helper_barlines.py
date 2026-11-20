"""Give single-measure systems a barline so the score reader can group their staves.

The reader works out which staves belong together from barlines that cross them. A system
that is one long measure has only its left edge and right end, and the reader then treats
every staff as unrelated. This draws one extra barline in such systems, at the emptiest
column near the middle, shaped like the page's own barlines (same width, joining the same
staves). The stitcher later merges the two halves back into one measure.

usage: python -I scripts/helper_barlines.py <page.png> <out.png> [--per-system 4]
Writes <out>.json listing the systems that were given a helper barline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def find_staves(dark, min_cover=0.5):
    """Staves as (top_row, bottom_row), plus line spacing and the rows that are staff lines.
    A staff is a run of long horizontal lines about four line-spacings tall; a long beam
    that happens to look like a sixth line does not disqualify it."""
    cover = dark.mean(axis=1)
    rows = np.where(cover > min_cover)[0]
    lines = [[rows[0], rows[0]]]
    for r in rows[1:]:
        if r - lines[-1][1] <= 2:
            lines[-1][1] = r
        else:
            lines.append([r, r])
    centres = [(a + b) / 2 for a, b in lines]
    gaps = np.diff(centres)
    space = float(np.median(gaps))
    groups, cur = [], [0]
    for i, g in enumerate(gaps):
        if g < 1.8 * space:
            cur.append(i + 1)
        else:
            groups.append(cur)
            cur = [i + 1]
    groups.append(cur)
    staves = []
    for g in groups:
        if len(g) >= 5 and abs((centres[g[-1]] - centres[g[0]]) - 4 * space) < 0.6 * space:
            staves.append((int(round(centres[g[0]])), int(round(centres[g[-1]]))))  # right height; extra rows are beams
            continue
        for k in range(len(g) - 4):  # any five consecutive lines spanning four spaces
            top, bot = centres[g[k]], centres[g[k + 4]]
            if abs((bot - top) - 4 * space) < 0.6 * space:
                staves.append((int(round(top)), int(round(bot))))
                break
    return staves, space, rows


def dark_runs(mask):
    xs = np.where(mask)[0]
    runs = []
    for x in xs:
        if runs and x - runs[-1][1] <= 1:
            runs[-1][1] = x
        else:
            runs.append([int(x), int(x)])
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("out")
    ap.add_argument("--per-system", type=int, default=4)
    args = ap.parse_args()
    img = Image.open(args.page).convert("L")
    a = np.asarray(img).copy()
    dark = a < 128
    staves, space, line_rows = find_staves(dark)
    n = args.per_system
    info = {"page": Path(args.page).name, "staves_found": len(staves), "systems": []}
    is_line = np.zeros(a.shape[0], bool)
    is_line[line_rows] = True
    for s in range(len(staves) // n):
        sys_staves = staves[s * n : (s + 1) * n]
        top, bot = sys_staves[0][0], sys_staves[0][1]
        # barlines: columns dark over the whole height of the first staff
        runs = dark_runs(dark[top : bot + 1].mean(axis=0) > 0.97)
        if len(runs) < 2:
            info["systems"].append({"system": s + 1, "helper": False, "note": "no system edges found"})
            continue
        left, right = runs[0], runs[-1]
        # a real barline crosses every staff of the system; note stems that happen to fill
        # one staff's height do not
        everywhere = np.ones(a.shape[1], bool)
        for t, b in sys_staves:
            everywhere &= dark[t : b + 1].mean(axis=0) > 0.97
        interior = [r for r in dark_runs(everywhere) if r[0] > left[1] + 3 and r[1] < right[0] - 12]
        width = max(right[1] - right[0] + 1, 2) if right[1] - right[0] < 4 else 2
        # which neighbouring staves does the page's own right-end barline join?
        joined = [bool(dark[sys_staves[i][1] + 2 : sys_staves[i + 1][0] - 1, right[0] : right[1] + 1].mean() > 0.9) for i in range(n - 1)]
        entry = {"system": s + 1, "interior_barlines": len(interior), "helper": False}
        if not interior:
            lo = left[1] + int(0.30 * (right[0] - left[1]))
            hi = left[1] + int(0.70 * (right[0] - left[1]))
            r0 = max(sys_staves[0][0] - int(4 * space), 0)
            r1 = min(sys_staves[-1][1] + int(4 * space), a.shape[0])
            band = dark[r0:r1].copy()
            band[is_line[r0:r1]] = False  # staff lines are everywhere; ignore them
            ink = band.sum(axis=0).astype(float)
            window = np.convolve(ink, np.ones(17), mode="same")  # room on both sides
            x = lo + int(np.argmin(window[lo:hi]))
            start = 0
            for i in range(n):
                if i == n - 1 or not joined[i]:
                    a[sys_staves[start][0] : sys_staves[i][1] + 1, x : x + width] = 0
                    start = i + 1
            entry.update(helper=True, x=int(x), fraction=round((x - left[1]) / (right[0] - left[1]), 3), ink_at_x=float(window[x]))
        entry["staves_joined_by_barlines"] = joined
        info["systems"].append(entry)
    Image.fromarray(a).save(args.out)
    Path(str(args.out) + ".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({"page": info["page"], "staves": info["staves_found"], "systems": [(e["system"], e.get("interior_barlines"), "helper at x=%s" % e["x"] if e["helper"] else "-") for e in info["systems"]]}))


if __name__ == "__main__":
    main()
