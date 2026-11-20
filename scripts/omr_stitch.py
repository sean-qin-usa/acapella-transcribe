"""Reassemble per-page MusicXML from the oemer score reader into one score with named staves.

oemer writes a system of N staves as blocks of two staves, one block after the other, each
block starting with a <print> marker. This puts the blocks back side by side, names each
staff from --layout, and writes the result as MIDI plus a plain table of notes.

usage: python -I scripts/omr_stitch.py <dir with page-NN.musicxml> <out_dir>
           [--layout "Violin 1,Violin 2,Piano,Piano"] [--beats 4]
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


def blocks_of(xml_text):
    """Measure numbers grouped into blocks; a new block starts at each <print> marker."""
    part = re.split(r"<part id=", xml_text)[1]
    out = []
    for num, body in re.findall(r'<measure number="(\d+)"[^>]*>(.*?)</measure>', part, flags=re.S):
        if "<print" in body or not out:
            out.append([])
        out[-1].append(int(num))
    return out


def measure_notes(staff_part, number):
    m = staff_part.measure(number)
    notes = []
    if m is None:
        return notes
    for el in m.flatten().notes:
        if el.duration.isGrace or el.quarterLength == 0:
            continue
        for p in el.pitches:
            notes.append((float(el.offset), float(el.quarterLength), int(p.midi)))
    return notes


def stitch(pages_dir, layout, beats, last_page=None):
    per_block = 2
    n_blocks = -(-len(layout) // per_block)
    rows, skipped, page_info, fill = [], [], [], []
    measure_index = 0
    for f in sorted(Path(pages_dir).glob("page-*.musicxml")):
        if last_page is not None and int(re.search(r"(\d+)", f.stem).group(1)) > last_page:
            continue
        blocks = blocks_of(f.read_text())
        staves = list(converter.parse(str(f)).parts)
        systems = [blocks[i : i + n_blocks] for i in range(0, len(blocks), n_blocks)]
        ok = len(blocks) % n_blocks == 0 and all(len({len(b) for b in s}) == 1 for s in systems) and len(staves) >= min(per_block, len(layout))
        if not ok:
            skipped.append({"page": f.name, "staves_read": len(staves), "measures_per_block": [len(b) for b in blocks]})
            continue
        for system in systems:
            for j in range(len(system[0])):
                for b, block in enumerate(system):
                    for s in range(per_block):
                        k = b * per_block + s
                        if k >= len(layout) or s >= len(staves):
                            continue
                        notes = measure_notes(staves[s], block[j])
                        if notes:
                            fill.append(max(o + d for o, d, _ in notes) / beats)
                        for off, dur, pitch in notes:
                            rows.append(dict(part=layout[k], staff=k, measure=measure_index + 1, beat=round(min(off, beats) + 1, 4), duration_beats=round(dur, 4), pitch_midi=pitch, pitch_name="%s%d" % (NAMES[pitch % 12], pitch // 12 - 1), page=f.name))
                measure_index += 1
        page_info.append({"page": f.name, "systems": len(systems), "measures_per_system": [len(s[0]) for s in systems]})
    f = np.array(fill) if fill else np.array([0.0])
    check = {"measures": measure_index, "staff_measures_filling_the_bar_within_10pct": float(np.mean(np.abs(f - 1) < 0.1)), "pages_skipped": skipped}
    return rows, page_info, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("out")
    ap.add_argument("--layout", default="Violin 1,Violin 2,Piano,Piano")
    ap.add_argument("--beats", type=float, default=4.0, help="quarter notes per measure")
    ap.add_argument("--last-page", type=int, default=None, help="join pages 1..N only (use when later pages could not be read reliably)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layout = [x.strip() for x in args.layout.split(",")]
    rows, page_info, check = stitch(args.pages, layout, args.beats, args.last_page)
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
