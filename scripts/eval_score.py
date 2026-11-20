"""Check a two-violin transcription against sheet music (MusicXML, .mxl or MIDI).

The score is stretched onto the recording's timing (chroma alignment), then each written
note is looked for in the transcription.

usage: python -I scripts/eval_score.py <score> <audio> <out_dir> [--parts 0,1] [--tol 0.15]
--parts: which staves of the score are the two violins (0-based, default: first two)
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import duet  # noqa: E402
import eval_urmp  # noqa: E402

HOP = 1024


def read_score(path):
    """Return [(part name, [(start, end, midi)])] in nominal seconds (tempo is irrelevant later)."""
    path = Path(path)
    parts = []
    if path.suffix.lower() in (".mid", ".midi"):
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(path))
        for inst in pm.instruments:
            if not inst.is_drum:
                parts.append((inst.name or "part", sorted((n.start, n.end, n.pitch) for n in inst.notes)))
        return parts
    from music21 import converter

    sc = converter.parse(str(path))
    try:
        sc = sc.expandRepeats()
    except Exception:
        pass  # no repeats, or repeats music21 cannot unfold
    for p in sc.parts:
        notes = []
        for el in p.stripTies().flatten().notes:
            if getattr(el.duration, "isGrace", False) or el.quarterLength == 0:
                continue
            for pitch in el.pitches:
                notes.append((float(el.offset), float(el.offset + el.quarterLength), int(pitch.midi)))
        parts.append((p.partName or "part", sorted(notes)))
    return parts


def roll_chroma(notes, n_frames, sr=amt.SR):
    c = np.zeros((12, n_frames))
    for s, e, p in notes:
        a, b = int(s * sr / HOP), max(int(e * sr / HOP), int(s * sr / HOP) + 1)
        c[p % 12, a:b] += 1.0
        c[(p + 7) % 12, a:b] += 0.25  # third harmonic lands a fifth up
    return c


def align(score_notes, y, sr=amt.SR):
    """Map score time to recording time. Returns a function and the mean alignment cost."""
    end = max(e for _, e, _ in score_notes)
    S = roll_chroma(score_notes, int(end * sr / HOP) + 1)
    A = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    S = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-6) + 1e-3
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-6) + 1e-3
    D, wp = librosa.sequence.dtw(X=S, Y=A, metric="cosine", step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]))
    wp = wp[::-1]
    st, at = wp[:, 0] * HOP / sr, wp[:, 1] * HOP / sr
    keep = np.r_[True, np.diff(st) > 0]
    return (lambda t: np.interp(t, st[keep], at[keep])), float(D[-1, -1] / len(wp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--parts", default="0,1")
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--min-ms", type=float, default=58.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parts = read_score(args.score)
    print("score staves:", [(i, n, len(notes)) for i, (n, notes) in enumerate(parts)])
    vi = [int(x) for x in args.parts.split(",")]
    y = amt.load_mono(args.audio)
    warp, cost = align([n for _, notes in parts for n in notes], y)

    truth, others = [], []
    for i, (_, notes) in enumerate(parts):
        for s, e, p in notes:
            row = dict(start=float(warp(s)), end=float(warp(e)), pitch=p, player=vi.index(i) if i in vi else -1)
            (truth if i in vi else others).append(row)
    truth.sort(key=lambda g: g["start"])

    raw = amt.poly_notes(args.audio, fmin=185.0, fmax=4200.0, min_ms=args.min_ms)
    A, B = duet.split_two_voices(list(raw))
    A, B = duet.trim_tails(A), duet.trim_tails(B)
    final = A + B
    label = {id(n): 0 for n in A} | {id(n): 1 for n in B}
    eval_urmp.TOL = args.tol
    m = eval_urmp.match(truth, final)
    found, used = {r for r, _ in m}, {e for _, e in m}
    agree = sum(1 for r, e in m if truth[r]["player"] == label[id(final[e])])
    extra = [n for j, n in enumerate(final) if j not in used]
    from_other = sum(1 for n in extra if any(abs(o["start"] - n.start) <= args.tol and o["pitch"] == n.pitch for o in others))
    high_blips = sum(1 for n in extra if n.dur < 0.15 and n.pitch >= 84)
    report = {
        "alignment_cost": cost,
        "violin_notes_in_score": len(truth),
        "found": len(m) / max(len(truth), 1),
        "written_notes_that_are_in_violin_parts": len(m) / max(len(final), 1),
        "right_player": max(agree, len(m) - agree) / max(len(m), 1),
        "extra_notes": len(extra),
        "extra_notes_that_are_another_staff": from_other,
        "extra_short_notes_above_C6": high_blips,
    }
    for k, name in enumerate(parts[i][0] for i in vi):
        idx = [i for i, g in enumerate(truth) if g["player"] == k]
        report["found_in_%s_(staff %d)" % (name, vi[k])] = sum(i in found for i in idx) / max(len(idx), 1)
    print(json.dumps(report, indent=1))
    (out / "score_check.json").write_text(json.dumps(report, indent=1))
    return truth, warp


if __name__ == "__main__":
    main()
