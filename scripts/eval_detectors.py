"""Recall and precision of the two note sources on one recording, and what their agreement is worth.

The two sources are the built-in detector (which only says "a note of this pitch sounds here")
and an instrument-aware model's MIDI file (scripts/run_yourmt3.py). They are different networks
trained on different recordings, so the working assumption is that their false notes rarely
coincide while real notes are heard by both. This script measures that against a score:

  recall     share of the score's notes that the source wrote
  precision  share of the source's notes that are in the score

for each source alone, for the notes both wrote, for the notes either wrote, and for the notes
only one of them wrote.

usage: python -I scripts/eval_detectors.py <score.mid> <recording> <model.mid>
           [--family strings] [--score-parts violin] [--lowest G3] [--tol 0.3] [--agree 0.15]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import duet  # noqa: E402
import eval_ensemble  # noqa: E402
import eval_urmp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("audio")
    ap.add_argument("model_notes")
    ap.add_argument("--family", default="all", help="which instrument family of the model's file to judge (default: every pitched note)")
    ap.add_argument("--score-parts", default="", help="judge only against score parts whose name contains this (the whole score is still used to line it up)")
    ap.add_argument("--lowest", default=None, help="ignore built-in detector notes below this, e.g. G3 when only violins are judged")
    ap.add_argument("--tol", type=float, default=0.3, help="how far a note's start may be from the score's (seconds)")
    ap.add_argument("--agree", type=float, default=0.15, help="how close two sources' starts must be to count as the same note (seconds)")
    ap.add_argument("--transpose", default="auto")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args()

    parts, warp, heard, y, info = eval_ensemble.line_up(args.score, args.audio, args.transpose)
    judged = [(name, notes) for name, notes in parts if args.score_parts.lower() in name.lower()]
    if args.family == "all":
        a, b, labels = duet.notes_from_midi(args.model_notes, "strings")
        model = sorted(a + b, key=lambda n: n.start)
    else:
        model, _, labels = duet.notes_from_midi(args.model_notes, args.family)
    plain = amt.merge_held(list(heard), y)
    if args.lowest:
        floor = duet.note_number(args.lowest) + info["score_moved_by_semitones"]
        plain = [n for n in plain if n.pitch >= floor]
    eval_urmp.TOL = args.tol

    def truth(shift=0.0):
        rows = [dict(start=float(warp(s)) + shift, end=float(warp(e)) + shift, pitch=p) for _, notes in judged for s, e, p in notes]
        return sorted(rows, key=lambda g: g["start"])

    t = truth()

    def score(notes):
        hits = len(eval_urmp.match(t, notes))
        chance = float(np.mean([len(eval_urmp.match(truth(s), notes)) for s in (5.0, -5.0, 9.0)]))
        return {
            "notes": len(notes),
            "recall": round(hits / max(len(t), 1), 3),
            "precision": round(hits / max(len(notes), 1), 3),
            "recall_by_chance": round(chance / max(len(t), 1), 3),
            "precision_by_chance": round(chance / max(len(notes), 1), 3),
        }

    plain_only = duet.cross_check(model, plain, tol=args.agree)
    both = [n for n in model if n.seen >= 1.0]
    model_only = [n for n in model if n.seen < 1.0]
    report = {
        **info,
        "judged_against": [name for name, _ in judged],
        "score_notes": len(t),
        "tracks_in_model_file": labels,
        "built-in detector": score(plain),
        "instrument-aware model": score(model),
        "both wrote it": score(both),
        "either wrote it": score(sorted(model + plain_only, key=lambda n: n.start)),
        "only the model wrote it": score(model_only),
        "only the built-in detector wrote it": score(plain_only),
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
