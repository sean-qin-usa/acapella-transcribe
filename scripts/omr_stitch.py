"""Reassemble per-page MusicXML from the oemer score reader into one score with separate staves.

oemer treats a four-staff system (two solo staves over a piano grand staff) as two two-staff
blocks written one after the other, each starting with a <print> marker. This puts the blocks
back side by side and writes a MIDI file with one track per staff group.

usage: python -I scripts/omr_stitch.py <dir with page-NN.musicxml> <out.mid> [--beats 4]
"""
import argparse
import re
import sys
from pathlib import Path

import pretty_midi
from music21 import converter


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("out")
    ap.add_argument("--beats", type=float, default=4.0, help="quarter notes per measure")
    args = ap.parse_args()
    tracks = {"Violin 1": [], "Violin 2": [], "Piano": []}
    measure_index, odd_pages, fill = 0, [], []
    for f in sorted(Path(args.pages).glob("page-*.musicxml")):
        blocks = blocks_of(f.read_text())
        sc = converter.parse(str(f))
        staves = list(sc.parts)
        if len(staves) != 2 or len(blocks) % 2 or any(len(a) != len(b) for a, b in zip(blocks[0::2], blocks[1::2])):
            odd_pages.append((f.name, len(staves), [len(b) for b in blocks]))
            continue
        for solo, piano in zip(blocks[0::2], blocks[1::2]):
            for j in range(len(solo)):
                t0 = measure_index * args.beats
                for name, staff, num in (("Violin 1", 0, solo[j]), ("Violin 2", 1, solo[j]), ("Piano", 0, piano[j]), ("Piano", 1, piano[j])):
                    notes = measure_notes(staves[staff], num)
                    if name != "Piano" and notes:
                        fill.append(max(o + d for o, d, _ in notes) / args.beats)
                    for off, dur, pitch in notes:
                        tracks[name].append((t0 + min(off, args.beats), t0 + min(off + dur, args.beats + 1), pitch))
                measure_index += 1
        print("%s: %d systems, measures per system %s" % (f.name, len(blocks) // 2, [len(b) for b in blocks[0::2]]))
    pm = pretty_midi.PrettyMIDI(initial_tempo=60)  # one quarter note = one second, tempo is nominal
    for name, notes in tracks.items():
        inst = pretty_midi.Instrument(program=0 if name == "Piano" else 40, name=name)
        inst.notes = [pretty_midi.Note(80, p, s, max(e, s + 0.05)) for s, e, p in sorted(notes)]
        pm.instruments.append(inst)
    pm.write(args.out)
    import numpy as np

    f = np.array(fill) if fill else np.array([0.0])
    print("measures: %d | notes: %s" % (measure_index, {k: len(v) for k, v in tracks.items()}))
    print("violin measures whose notes end within 10%% of the bar length: %.0f%% (a rough check of how well rhythms were read)" % (100 * np.mean(np.abs(f - 1) < 0.1)))
    if odd_pages:
        print("pages skipped because the layout was not two blocks per system:", odd_pages)


if __name__ == "__main__":
    main()
