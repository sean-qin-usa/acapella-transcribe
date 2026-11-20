"""Reassemble per-page MusicXML from the oemer score reader into one score with named staves.

The reader writes a system of N staves as blocks, one after the other, each block starting
with a <print> marker. Two shapes occur:

  paired      the reader grouped the staves in twos (it found barlines joining them): each
              block holds two staves, and a system is N/2 blocks.
  one-by-one  the reader could not group them (systems that are one long measure): each block
              is a single staff. Such a page is used only if it was read from an image with
              helper barlines (scripts/helper_barlines.py) and the number of blocks equals
              the number of staves found on that image, so each block can be placed by its
              position on the page. Anything else is skipped and listed, never guessed.

usage: python -I scripts/omr_stitch.py <dir with page-NN.musicxml> <out_dir>
           [--layout "Violin 1,Violin 2,Piano,Piano"] [--beats 4] [--last-page N]
           [--helped <dir of reads from helper-barline images>] [--page-info <dir of page-NN.png.json>]
--layout lists the staves of one system from top to bottom; staves with the same name are
merged into one part (a piano's two staves).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pretty_midi
from music21 import converter

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def load_page(path):
    """Staves (each a list of measures in document order) and blocks of measure positions."""
    text = Path(path).read_text()
    part = re.split(r"<part id=", text)[1]
    bodies = re.findall(r'<measure number="\d+"[^>]*>(.*?)</measure>', part, flags=re.S)
    blocks = []
    for i, body in enumerate(bodies):
        if "<print" in body or not blocks:
            blocks.append([])
        blocks[-1].append(i)
    staves = [list(p.getElementsByClass("Measure")) for p in converter.parse(str(path)).parts]
    return staves, blocks


def contents(measure):
    """Notes of one measure as (offset, length, midi pitch), and the measure's filled length."""
    notes, filled = [], 0.0
    if measure is None:
        return notes, filled
    for el in measure.flatten().notesAndRests:
        if el.duration.isGrace or el.quarterLength == 0:
            continue
        filled = max(filled, float(el.offset + el.quarterLength))
        if el.isRest:
            continue
        for p in el.pitches:
            notes.append((float(el.offset), float(el.quarterLength), int(p.midi)))
    return notes, filled


def systems_paired(staves, blocks, n_staves, beats, helper=None):
    """Each block is two staves; a system is n_staves/2 blocks with equal measure counts.
    `helper` lists, per system, whether a helper barline split its single measure in two; those
    two halves are laid end to end and squeezed back into one measure."""
    per = -(-n_staves // 2)
    if len(staves) != 2 or len(blocks) % per:
        return None
    out = []
    for i in range(0, len(blocks), per):
        system = blocks[i : i + per]
        if len({len(b) for b in system}) != 1:
            return None
        notes = []
        merge = bool(helper and len(helper) == len(blocks) // per and helper[i // per] and len(system[0]) == 2)
        for b, block in enumerate(system):
            for s in range(2):
                k = 2 * b + s
                if k >= n_staves:
                    continue
                got = [contents(staves[s][idx] if idx < len(staves[s]) else None) for idx in block]
                if merge:
                    (n1, f1), (n2, f2) = got
                    total = (f1 + f2) or beats
                    for base, ns in ((0.0, n1), (f1, n2)):
                        notes += [(k, 0, min((base + off) * beats / total, beats - 1e-6), dur * beats / total, pitch) for off, dur, pitch in ns]
                else:
                    for j, (ns, _) in enumerate(got):
                        notes += [(k, j, min(off, beats), dur, pitch) for off, dur, pitch in ns]  # an over-full bar keeps its notes
        out.append({"measures": 1 if merge else len(system[0]), "notes": notes, "how": "paired"})
    return out


def systems_one_by_one(staves, blocks, n_staves, beats, real_measures):
    """Each block is one staff, in page order. A staff's measures are laid end to end and
    stretched to the system's true length, so a misread rhythm in one staff cannot push it
    out of step with the others beyond the end of the system."""
    if len(staves) != 1 or len(blocks) % n_staves or len(blocks) // n_staves != len(real_measures):
        return None
    out = []
    for i in range(0, len(blocks), n_staves):
        target = real_measures[i // n_staves] * beats
        notes = []
        for k, block in enumerate(blocks[i : i + n_staves]):
            pos, got = 0.0, []
            for idx in block:
                ns, filled = contents(staves[0][idx])
                got += [(pos + off, dur, pitch) for off, dur, pitch in ns]
                pos += filled
            scale = target / pos if pos > 0 else 1.0
            for off, dur, pitch in got:
                pos = min(off * scale, target - 1e-6)
                notes.append((k, int(pos // beats), pos % beats, dur * scale, pitch))
        out.append({"measures": real_measures[i // n_staves], "notes": notes, "how": "one-by-one"})
    return out


def stitch(pages_dir, layout, beats, last_page=None, helped_dir=None, info_dir=None):
    n = len(layout)
    rows, skipped, page_info, fill = [], [], [], []
    measure_index = 0
    for f in sorted(Path(pages_dir).glob("page-*.musicxml")):
        if last_page is not None and int(re.search(r"(\d+)", f.stem).group(1)) > last_page:
            continue
        candidates = [("as read", f)]
        if helped_dir and (Path(helped_dir) / f.name).exists():
            candidates.append(("with helper barlines", Path(helped_dir) / f.name))
        info_path = Path(info_dir) / (f.stem + ".png.json") if info_dir else None
        info = json.loads(info_path.read_text()) if info_path and info_path.exists() else None
        if info and len(candidates) == 2 and any(s.get("helper") for s in info["systems"]):
            candidates.reverse()  # the page needed a helper barline, so trust that read first
        systems, used, tried = None, None, []
        for label, path in candidates:
            staves, blocks = load_page(path)
            helper = [bool(x.get("helper")) for x in info["systems"]] if info and label == "with helper barlines" else None
            systems = systems_paired(staves, blocks, n, beats, helper)
            if systems is None and info and label == "with helper barlines" and len(blocks) == info["staves_found"]:
                real = [s.get("interior_barlines", 0) + 1 for s in info["systems"]]
                systems = systems_one_by_one(staves, blocks, n, beats, real)
            tried.append({"read": label, "staves_in_file": len(staves), "blocks": len(blocks)})
            if systems is not None:
                used = label
                break
        if systems is None:
            skipped.append({"page": f.name, "tried": tried, "staves_on_page": info["staves_found"] if info else None})
            continue
        for system in systems:
            for k, j, off, dur, pitch in system["notes"]:
                rows.append(dict(part=layout[k], staff=k, measure=measure_index + j + 1, beat=round(off + 1, 4), duration_beats=round(dur, 4), pitch_midi=pitch, pitch_name="%s%d" % (NAMES[pitch % 12], pitch // 12 - 1), page=f.name))
            per_measure = {}
            for k, j, off, dur, _ in system["notes"]:
                per_measure[(k, j)] = max(per_measure.get((k, j), 0.0), off + dur)
            fill += [v / beats for v in per_measure.values()]
            measure_index += system["measures"]
        page_info.append({"page": f.name, "read": used, "how": systems[0]["how"], "systems": len(systems), "measures_per_system": [s["measures"] for s in systems]})
    f = np.array(fill) if fill else np.array([0.0])
    check = {"measures": measure_index, "staff_measures_filling_the_bar_within_10pct": float(np.mean(np.abs(f - 1) < 0.1)), "pages_skipped": skipped}
    return rows, page_info, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("out")
    ap.add_argument("--layout", default="Violin 1,Violin 2,Piano,Piano")
    ap.add_argument("--beats", type=float, default=4.0, help="quarter notes per measure")
    ap.add_argument("--last-page", type=int, default=None, help="join pages 1..N only")
    ap.add_argument("--helped", default=None, help="folder of reads made from helper-barline images")
    ap.add_argument("--page-info", default=None, help="folder of page-NN.png.json files written by helper_barlines.py")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = [x.strip() for x in args.layout.split(",")]
    rows, page_info, check = stitch(args.pages, layout, args.beats, args.last_page, args.helped, args.page_info)
    check["covers_pages"] = [1, args.last_page] if args.last_page else "all that could be joined"

    with open(out / "notes_score.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["part", "staff", "measure", "beat", "duration_beats", "pitch_midi", "pitch_name", "page"])
        w.writeheader()
        w.writerows(rows)
    pm = pretty_midi.PrettyMIDI(initial_tempo=60)  # one quarter note = one second; tempo is nominal
    for name in dict.fromkeys(layout):
        inst = pretty_midi.Instrument(program=0 if "piano" in name.lower() else 40, name=name)
        for r in rows:
            if r["part"] == name:
                s = (r["measure"] - 1) * args.beats + r["beat"] - 1
                inst.notes.append(pretty_midi.Note(80, r["pitch_midi"], s, s + max(r["duration_beats"], 0.05)))
        inst.notes.sort(key=lambda n: (n.start, n.pitch))
        pm.instruments.append(inst)
    pm.write(str(out / "score.mid"))
    counts = {name: sum(r["part"] == name for r in rows) for name in dict.fromkeys(layout)}
    meta = {"layout": layout, "beats_per_measure": args.beats, "notes_per_part": counts, "pages": page_info, **check}
    (out / "score_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "pages"}, indent=1))


if __name__ == "__main__":
    main()
