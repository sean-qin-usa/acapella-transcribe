"""Score a transcription of several like-sounding instruments against a machine-readable score.

The score (a MIDI file with one named track per player, in score time) is stretched onto the
recording, and two things are measured:
  notes   how many of the score's notes the transcription contains, and how many of the
          transcription's notes are in the score, with the same test repeated on a
          time-shifted score to show how much is coincidence;
  parts   of the notes found, how many were given to the right player by each way of
          dividing them.

usage: python -I scripts/eval_ensemble.py <score.mid> <recording> <notes.mid> [--family strings]
           [--floors G3,G3,C3,C2] [--tol 0.3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import duet  # noqa: E402
import eval_score  # noqa: E402
import eval_urmp  # noqa: E402


def find_transposition(parts, heard):
    """The shift (semitones) at which the score's pitches best match the pitches heard, ignoring timing."""
    from collections import Counter

    counts = Counter(p for _, _, p in heard)
    total = sum(len(notes) for _, notes in parts)

    def fit(shift):
        want = Counter(p + shift for _, notes in parts for _, _, p in notes)
        return sum(min(want[p], counts[p]) for p in want) / total

    fits = {shift: fit(shift) for shift in range(-14, 15)}
    best = max(fits, key=lambda k: (fits[k], -abs(k)))
    return best, fits[best], max(v for k, v in fits.items() if k != best)


def line_up(score, audio, transpose="auto", how="notes"):
    """Read a score and stretch it onto a recording.

    Returns (parts at the recording's pitch, warp from score time to recording time, the notes
    the built-in detector heard, the recording, a dict describing what was done)."""
    parts = eval_score.read_score(score)
    y = amt.load_mono(audio)
    heard = amt.poly_notes(audio, fmin=60.0, fmax=4200.0, min_ms=58)
    triples = [(n.start, n.end, n.pitch) for n in heard]
    if transpose == "auto":
        shift, fit, runner_up = find_transposition(parts, triples)
        info = {"score_moved_by_semitones": shift, "pitch_match_at_that_shift_and_next_best": [round(fit, 2), round(runner_up, 2)]}
    else:
        shift = int(transpose)
        info = {"score_moved_by_semitones": shift, "pitch_match_at_that_shift_and_next_best": None}
    parts = [(name, [(s0, e0, p0 + shift) for s0, e0, p0 in notes]) for name, notes in parts]
    everything = [n for _, notes in parts for n in notes]
    if how == "notes":
        warp, _ = eval_score.align_to_notes(everything, triples, len(y) / amt.SR)
    else:
        warp, _ = eval_score.align(everything, y, onset_w=1.0, smooth=9)
    info.update(aligned_by=how, recording_seconds=round(len(y) / amt.SR, 1))
    return parts, warp, heard, y, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("audio")
    ap.add_argument("notes")
    ap.add_argument("--family", default="strings")
    ap.add_argument("--floors", default="G3,G3,C3,C2", help="lowest playable note of each part, highest part first")
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--transpose", default="auto", help="semitones to move the score to the recording's pitch, or 'auto' to find it (an edition in another octave, a choir at Baroque pitch)")
    ap.add_argument("--align", default="notes", choices=["notes", "harmony"], help="line the score up with the recording by individual notes (heard by the built-in detector) or by harmony")
    args = ap.parse_args()

    # the built-in detector is independent of the transcription being judged
    parts, warp, _, y, info = line_up(args.score, args.audio, args.transpose, args.align)
    names = [n for n, _ in parts]
    floors = [duet.note_number(x) + info["score_moved_by_semitones"] for x in args.floors.split(",")]
    det, other, labels = duet.notes_from_midi(args.notes, args.family)
    eval_urmp.TOL = args.tol

    def truth(shift=0.0, which=None):
        rows = [dict(start=float(warp(s)) + shift, end=float(warp(e)) + shift, pitch=p, player=i) for i, (_, notes) in enumerate(parts) if which is None or i == which for s, e, p in notes]
        return sorted(rows, key=lambda g: g["start"])

    t = truth()
    m = eval_urmp.match(t, det)
    chance = float(np.mean([len(eval_urmp.match(truth(s), det)) / len(t) for s in (5.0, -5.0, 9.0)]))
    report = {
        **info,
        "tracks_in_transcription": labels,
        "score_notes": len(t),
        "transcribed_notes": len(det),
        "found": len(m) / len(t),
        "found_by_chance": chance,
        "transcribed_notes_in_score": len(m) / max(len(det), 1),
        "found_per_part": {names[i]: len(eval_urmp.match(truth(which=i), det)) / max(len(parts[i][1]), 1) for i in range(len(parts))},
        "notes_given_to_other_instruments": len(other),
    }
    who = {id(det[e]): t[r]["player"] for r, e in m}
    n = len(parts)

    def right(groups):
        part = {id(x): v for v, g in enumerate(groups) for x in g}
        ok = np.array([part.get(i) == w for i, w in who.items()])
        per = {names[v]: float(np.mean([part.get(i) == v for i, w in who.items() if w == v])) for v in range(n)}
        return float(ok.mean()), per

    legal = [sum(1 for f in floors if det_note.pitch >= f) for det_note in det if id(det_note) in who]
    report["right_part"] = {"guessing among the parts whose range allows the note": float(np.mean([1.0 / max(k, 1) for k in legal]))}
    for label, groups in (
        ("stack by pitch", duet.stack_by_pitch(list(det), n, floors)),
        ("line following, no range limits", duet.split_voices(list(det), n)),
        ("line following with range limits", duet.split_voices(list(det), n, floors)),
    ):
        overall, per = right(groups)
        report["right_part"][label] = overall
        report.setdefault("right_part_by_player", {})[label] = per
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
