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


def align(score_notes, y, sr=amt.SR, onset_w=0.0, smooth=0, partial=False):
    """Map score time to recording time. Returns (warp function, mean alignment cost).

    onset_w  weight of a thirteenth feature marking where notes begin (score) and where the
             recording has attacks, so held chords and rests do not let the path drift
    smooth   median-filter the path over this many frames to remove jitter
    partial  the score covers only part of the recording: find where it fits instead of
             stretching it from the first second to the last
    The returned function has a .span attribute: the (start, end) seconds of the recording
    that the score was matched to.
    """
    end = max(e for _, e, _ in score_notes)
    n_s = int(end * sr / HOP) + 1
    S = roll_chroma(score_notes, n_s)
    A = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    S = S / (np.linalg.norm(S, axis=0, keepdims=True) + 1e-6)
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-6)
    if onset_w:
        so = np.zeros(n_s)
        for s0, _, _ in score_notes:
            so[min(int(s0 * sr / HOP), n_s - 1)] = 1.0
        so = np.maximum(so, 0.5 * np.r_[so[1:], 0], 0.5 * np.r_[0, so[:-1]])
        ao = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)[: A.shape[1]]
        ao = np.clip(ao / (np.percentile(ao, 95) + 1e-9), 0, 1)
        S = np.vstack([S, onset_w * so[None, :]])
        A = np.vstack([A, onset_w * np.pad(ao, (0, A.shape[1] - len(ao)))[None, :]])
    D, wp = librosa.sequence.dtw(X=S + 1e-3, Y=A + 1e-3, metric="euclidean", subseq=partial, step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]))
    wp = wp[::-1]
    cost = float((D[wp[-1][0], wp[-1][1]] - (D[wp[0][0], wp[0][1]] if partial else 0.0)) / len(wp))
    st, at = wp[:, 0] * HOP / sr, wp[:, 1] * HOP / sr
    keep = np.r_[True, np.diff(st) > 0]
    st, at = st[keep], at[keep]
    if smooth:
        import scipy.ndimage as ndi

        trend = at[0] + (st - st[0]) * (at[-1] - at[0]) / max(st[-1] - st[0], 1e-9)
        at = np.maximum.accumulate(ndi.median_filter(at - trend, size=smooth, mode="nearest") + trend)

    def warp(t):
        return np.interp(t, st, at)

    warp.span = (float(at[0]), float(at[-1]))
    return warp, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--parts", default="0,1")
    ap.add_argument("--tol", type=float, default=0.3)
    ap.add_argument("--min-ms", type=float, default=58.0)
    ap.add_argument("--partial", action="store_true", help="the score covers only part of the recording")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parts = read_score(args.score)
    print("score staves:", [(i, n, len(notes)) for i, (n, notes) in enumerate(parts)])
    vi = [int(x) for x in args.parts.split(",")]
    y = amt.load_mono(args.audio)
    warp, cost = align([n for _, notes in parts for n in notes], y, onset_w=1.0, smooth=9, partial=args.partial)
    lo, hi = warp.span

    truth, others = [], []
    for i, (_, notes) in enumerate(parts):
        for s, e, p in notes:
            row = dict(start=float(warp(s)), end=float(warp(e)), pitch=p, player=vi.index(i) if i in vi else -1)
            (truth if i in vi else others).append(row)
    truth.sort(key=lambda g: g["start"])

    raw = amt.poly_notes(args.audio, fmin=185.0, fmax=4200.0, min_ms=args.min_ms)
    A, B = duet.split_two_voices(list(raw))
    A, B = duet.trim_tails(A), duet.trim_tails(B)
    A = [n for n in A if lo - args.tol <= n.start <= hi + args.tol]  # judge only where the score reaches
    B = [n for n in B if lo - args.tol <= n.start <= hi + args.tol]
    final = A + B
    label = {id(n): 0 for n in A} | {id(n): 1 for n in B}
    eval_urmp.TOL = args.tol
    m = eval_urmp.match(truth, final)
    found, used = {r for r, _ in m}, {e for _, e in m}
    agree = sum(1 for r, e in m if truth[r]["player"] == label[id(final[e])])
    extra = [n for j, n in enumerate(final) if j not in used]
    from_other = sum(1 for n in extra if any(abs(o["start"] - n.start) <= args.tol and o["pitch"] == n.pitch for o in others))
    high_blips = sum(1 for n in extra if n.dur < 0.15 and n.pitch >= 84)
    in_range = [g for g in truth if g["pitch"] >= 55]
    report = {
        "alignment_cost": cost,
        "recording_seconds_covered_by_score": [round(lo, 1), round(hi, 1)],
        "recording_length_sec": round(len(y) / amt.SR, 1),
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
