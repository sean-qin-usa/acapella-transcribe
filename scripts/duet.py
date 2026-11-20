"""Transcribe two like-sounding instruments from one recording and split the notes into two parts.

usage: python -I scripts/duet.py <audio> <out_dir> [--fmin 185 --fmax 3600]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # isolated mode drops the script folder
import amt  # noqa: E402


def drop_overtone_ghosts(notes, tol=0.06):
    """Remove quiet notes that shadow a louder note exactly an octave/12th/2 octaves below."""
    keep = []
    for n in notes:
        ghost = False
        for m in notes:
            if m is n or m.pitch >= n.pitch:
                continue
            if (n.pitch - m.pitch) in (12, 19, 24) and n.amp < 0.6 * m.amp:
                if m.start - tol <= n.start and n.end <= m.end + tol:
                    ghost = True
                    break
        if not ghost:
            keep.append(n)
    return keep


def drop_blips(notes, max_dur=0.15, tol=0.06):
    """Remove short notes sitting on an overtone of a louder note that is sounding at the same time.

    Unlike drop_overtone_ghosts this keeps sustained octave doublings: a real second violin an
    octave up holds its note; a detector ghost flickers."""
    keep = []
    for n in notes:
        ghost = False
        if n.dur < max_dur:
            for m in notes:
                if m is not n and (n.pitch - m.pitch) in (12, 19, 24, 28, 31) and m.amp >= n.amp:
                    if m.start - tol <= n.start and n.end <= m.end + tol:
                        ghost = True
                        break
        if not ghost:
            keep.append(n)
    return keep


def drop_leap_blips(part, max_dur=0.15, leap=9):
    """Within one part, remove a short note that jumps far above both the note before and the note after."""
    part = sorted(part, key=lambda n: (n.start, n.pitch))
    keep = []
    for i, n in enumerate(part):
        if n.dur < max_dur and 0 < i < len(part) - 1:
            if n.pitch - part[i - 1].pitch >= leap and n.pitch - part[i + 1].pitch >= leap:
                continue
        keep.append(n)
    return keep


def merge_repeats(notes, gap=0.03):
    """Join back-to-back notes of the same pitch: usually one held note the detector chopped up."""
    out, last = [], {}
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        p = last.get(n.pitch)
        if p is not None and n.start - p.end <= gap:
            p.end = max(p.end, n.end)
            p.amp = max(p.amp, n.amp)
        else:
            out.append(n)
            last[n.pitch] = n
    return out


def effective_durations(notes, gap=0.05):
    """Give each note the length of the same-pitch chain it sits in, so a held note that the
    detector chopped into pieces still counts as long."""
    chains = {}
    for n in sorted(notes, key=lambda n: n.start):
        c = chains.get(n.pitch)
        if c is not None and n.start - c[-1].end <= gap:
            c.append(n)
        else:
            c = chains[n.pitch] = [n]
        total = c[-1].end - c[0].start
        for m in c:
            m.eff = total
    for n in notes:
        n.eff = n.eff or n.dur


def split_two_voices(notes, beam=96, idle=0.0, dur_w=0.0, ring_ok=False, role_w=0.0, role_split=0.45, tone=None, tone_w=0.0, register_w=4.0):
    """Beam search over assignments: each part should move by small steps and rarely play two notes at once.

    idle   extra cost for handing a note to the part that is resting while the other part is the
           one playing (keeps a solo passage in one part instead of dividing it by register)
    dur_w  cost per doubling/halving of note length relative to the part's previous note
           (keeps a run of short notes together and a sustained line together)
    ring_ok  treat an overlap as a second string only if it also covers half of the new note
    register_w  cost, in semitones, of giving the higher of two sounding notes to the lower part
    tone     optional {id(note): (cost if player 0, cost if player 1)} from a tone classifier;
             with it part A is player 0 and part B is player 1
    role_w   split by role instead of by player: part A is the sustained line and part B the
             moving line. Cost per doubling that a note's held length is on the wrong side of
             `role_split` seconds for the part it is given to.
    """
    if role_w:
        effective_durations(notes)

    def step_cost(prev, n, other):
        resting = prev is None or n.start - prev.end > 1.0
        other_busy = other is not None and n.start - other.end < 0.4
        other_sounding = other is not None and other.end - n.start > 0.15
        cost = idle if (resting and other_busy and not other_sounding) else 0.0
        if prev is None:
            return cost
        gap = n.start - prev.end
        overlap = min(prev.end, n.end) - n.start
        if overlap > 0.15 and (not ring_ok or overlap > 0.5 * n.dur):  # else legato or room reverb
            cost += 5.0 + 14.0 * overlap / max(n.dur, 0.05)
        memory = 0.25 + 0.75 * math.exp(-max(gap, 0.0) / 1.5)  # a long rest loosens the link
        if dur_w:
            cost += dur_w * min(abs(math.log2(max(n.dur, 0.03) / max(prev.dur, 0.03))), 3.0)
        return cost + abs(n.pitch - prev.pitch) * memory

    # state: (cost, last note in part A, last note in part B, labels so far)
    states = [(0.0, None, None, ())]
    for n in notes:
        nxt = {}
        for cost, a, b, lab in states:
            for v, (pa, pb) in enumerate(((n, b), (a, n))):
                other = b if v == 0 else a
                c = cost + step_cost(a if v == 0 else b, n, other)
                if tone is not None and id(n) in tone:
                    c += tone_w * tone[id(n)][v]
                if role_w:
                    ratio = math.log2(max(n.eff, 0.03) / role_split)
                    c += role_w * min(max(-ratio if v == 0 else ratio, 0.0), 3.0)
                # mild preference for not swapping registers while both parts sound
                if other is not None and other.end > n.start + 0.04:
                    upper_is_a = (n.pitch > other.pitch) == (v == 0)
                    c += 0.0 if upper_is_a else register_w
                key = (id(pa), id(pb))
                if key not in nxt or c < nxt[key][0]:
                    nxt[key] = (c, pa, pb, lab + (v,))
        states = sorted(nxt.values(), key=lambda s: s[0])[:beam]
    labels = states[0][3]
    A = [n for n, v in zip(notes, labels) if v == 0]
    B = [n for n, v in zip(notes, labels) if v == 1]
    if not role_w and tone is None and np.mean([n.pitch for n in A] or [0]) < np.mean([n.pitch for n in B] or [0]):
        A, B = B, A
    return A, B


def trim_tails(part, same_onset=0.05, tail=0.15):
    """Within one part, cut short ring-over into the next note, but keep real chords.

    Notes that start together are a chord. A note that keeps sounding well into the next one
    (more than `tail` seconds and more than half of that next note) is a held double stop and
    is kept; anything shorter is legato or room reverb and is cut at the next note's start."""
    part = sorted(part, key=lambda n: (n.start, n.pitch))
    for i, n in enumerate(part):
        for m in part[i + 1 :]:
            if m.start - n.start > same_onset:
                overlap = min(n.end, m.end) - m.start
                if 0 < overlap and not (overlap > tail and overlap > 0.5 * m.dur):
                    n.end = m.start
                break
    return [n for n in part if n.dur >= 0.06]


def split_parts(notes, audio_path, roles=False, use_stereo=True, min_spread_db=5.0):
    """Divide the notes into two parts, choosing the evidence that suits the recording.

    A stereo recording whose notes are spread widely between left and right is split by side:
    each note goes to the part on its side of the recording's centre. On the test piece (two
    violins and piano, the violins about 7 dB apart) that put 83% of violin notes in the right
    part, where pitch movement managed 62% and any mixture of the two did worse, because the
    piano's notes pull the pitch-based logic around. One source alone spreads over roughly
    3-4 dB, so a spread of `min_spread_db` or more is taken to mean sources at different
    places; that bar comes from three recordings and is a starting point.

    Otherwise (mono, or a narrow stereo image) the split is by pitch movement and register.
    Returns (part A, part B, what was used)."""
    if roles:
        A, B = split_two_voices(notes, role_w=3.0)
        return A, B, {"split_by": "sustained line / moving line"}
    pans = amt.pan_per_note(audio_path, notes) if use_stereo else None
    if pans is not None and len(notes) >= 40:
        spread = float(np.percentile(pans, 75) - np.percentile(pans, 25))
        centre = float(np.median(pans))
        if spread >= min_spread_db:
            A = [n for n, p in zip(notes, pans) if p < centre]
            B = [n for n, p in zip(notes, pans) if p >= centre]
            return A, B, {"split_by": "left/right position", "spread_db": round(spread, 1), "centre_db": round(centre, 1)}
        A, B = split_two_voices(notes)
        return A, B, {"split_by": "pitch movement and register", "stereo_spread_db": round(spread, 1), "why": "stereo image too narrow to separate players"}
    A, B = split_two_voices(notes)
    return A, B, {"split_by": "pitch movement and register", "why": "mono recording" if use_stereo else "left/right position switched off"}


def tone_disputes(y, A, B):
    """Second opinion on the split from tone alone, with no labels needed.

    Each violin's tone is learned from the notes where the split is safest (both parts sounding
    at once), then every note is asked which violin it sounds like. Returns the ids of notes
    where tone disagrees with the split, or None when there is too little overlap to learn from.
    On the URMP benchmark the split is right for about 90% of notes where the two opinions agree
    and about 71% where they disagree, so a disagreement marks a note worth checking by ear."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    import tone_experiment as te

    S, freqs = te.spectrogram(y)
    notes = A + B
    part = {id(n): 0 for n in A} | {id(n): 1 for n in B}
    other = {0: B, 1: A}
    X = np.array([te.tone(S, freqs, n, [o for o in notes if o is not n and o.start < n.end and o.end > n.start]) for n in notes])
    lab = np.array([part[id(n)] for n in notes])
    safe = np.array([any(min(o.end, n.end) - max(o.start, n.start) > 0.5 * n.dur for o in other[part[id(n)]]) for n in notes])
    if safe.sum() < 40 or len(set(lab[safe])) < 2:
        return None
    Xs, mean = te.fill(X[safe])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xs, lab[safe])
    vote = clf.predict(te.fill(X, mean)[0])
    return {id(n) for n, v in zip(notes, vote) if v != part[id(n)]}


def crossings(A, B, step=0.05):
    end = max(n.end for n in A + B)
    grid = np.arange(0, end, step)

    def line(notes):
        p = np.full(len(grid), np.nan)
        for n in notes:
            p[int(n.start / step) : int(n.end / step)] = n.pitch
        return p

    a, b = line(A), line(B)
    both = np.isfinite(a) & np.isfinite(b)
    sign = np.sign(a[both] - b[both])
    sign = sign[sign != 0]
    return int((np.diff(sign) != 0).sum()), float((sign < 0).mean()) if len(sign) else 0.0, float(both.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--fmin", type=float, default=185.0)
    ap.add_argument("--fmax", type=float, default=4200.0)
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--min-ms", type=float, default=58.0, help="shortest note to keep")
    ap.add_argument("--no-stereo", action="store_true", help="ignore left/right position even when the recording is stereo")
    ap.add_argument("--check-overtones", action="store_true", help="remove notes the recording says are only an overtone of a lower note (for pieces with piano)")
    ap.add_argument("--keep-rewrites", action="store_true", help="do not join a held note that the detector wrote twice")
    ap.add_argument("--solo", default="", help="time ranges where only one violin plays, e.g. 0-13,95-102 (seconds)")
    ap.add_argument("--roles", action="store_true", help="split into a sustained line and a moving line instead of guessing which player is which")
    ap.add_argument("--keep-leaps", action="store_true", help="keep short notes that jump far above their neighbours")
    ap.add_argument("--drop-blips", action="store_true", help="remove short notes on an overtone of a louder note (also hides some real octave doublings)")
    ap.add_argument("--drop-overtones", action="store_true", help="remove quiet notes an octave above a louder one (also removes real octave doublings)")
    ap.add_argument("--frame", type=float, default=0.3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    y = amt.load_mono(args.audio)
    raw = amt.poly_notes(args.audio, fmin=args.fmin, fmax=args.fmax, onset=args.onset, frame=args.frame, min_ms=args.min_ms)
    notes = drop_overtone_ghosts(raw) if args.drop_overtones else list(raw)
    if not args.keep_rewrites:
        notes = amt.merge_held(notes, y)
    if args.check_overtones:
        notes = amt.drop_checked_overtones(notes, y)
    if args.drop_blips:
        notes = drop_blips(notes)
    A, B, stereo = split_parts(notes, args.audio, roles=args.roles, use_stereo=not args.no_stereo)
    for span in filter(None, "" if args.roles else args.solo.split(",")):  # a hint from someone who knows the piece
        lo, hi = (float(x) for x in span.split("-"))
        inA = [n for n in A if lo <= n.start < hi]
        inB = [n for n in B if lo <= n.start < hi]
        if len(inA) >= len(inB):
            A, B = A + inB, [n for n in B if not lo <= n.start < hi]
        else:
            A, B = [n for n in A if not lo <= n.start < hi], B + inA
    if args.roles:
        A = merge_repeats(A, gap=0.05)  # in the sustained line, back-to-back same pitches are one held note
    A, B = trim_tails(A), trim_tails(B)
    if not args.keep_leaps:
        A, B = drop_leap_blips(A), drop_leap_blips(B)
    if not args.roles and np.mean([n.pitch for n in A] or [0]) < np.mean([n.pitch for n in B] or [0]):
        A, B = B, A
    names = ("Sustained line", "Moving line") if args.roles else ("Violin 1", "Violin 2")
    disputed = None if args.roles else tone_disputes(y, A, B)
    for n in A + B:
        n.conf = 0.0 if disputed and id(n) in disputed else 1.0
    beats, bpm, jitter = amt.beat_grid(y)
    n_cross, frac_inverted, frac_both = crossings(A, B)

    report = {
        "notes_raw": len(raw),
        "notes_after_joining_held_notes": len(notes),
        "part1_notes": len(A),
        "part2_notes": len(B),
        "part1_range": [amt.name(min(n.pitch for n in A)), amt.name(max(n.pitch for n in A))],
        "part2_range": [amt.name(min(n.pitch for n in B)), amt.name(max(n.pitch for n in B))],
        "simultaneous_notes_share_after_cleanup": amt.polyphony_profile(notes),
        "part1_double_stop_share": amt.polyphony_profile(A).get(2, 0.0),
        "part2_double_stop_share": amt.polyphony_profile(B).get(2, 0.0),
        "share_of_time_both_parts_play": frac_both,
        "part_crossings": n_cross,
        "share_of_duet_time_part2_above_part1": frac_inverted,
        "chroma_agreement_with_audio": amt.chroma_agreement(A + B, y),
        "how_parts_were_split": stereo,
        "notes_where_tone_disputes_the_split": None if disputed is None else len(disputed) / max(len(A) + len(B), 1),
        "tempo_bpm": bpm,
        "beat_jitter": jitter,
    }
    print(json.dumps(report, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))

    import soundfile as sf

    amt.write_check(out / "check_duet.wav", y, A + B)  # recording left, transcription right
    pa, pb = amt.synth(A, len(y)), amt.synth(B, len(y))
    peak = max(np.abs(pa).max(), np.abs(pb).max(), 1e-9)
    sf.write(str(out / "check_parts.wav"), np.stack([pa, pb], axis=1) / peak * 0.7, amt.SR)  # part 1 left, part 2 right

    # real-time MIDI keeps the performance timing; the gridded copy is for notation
    amt.write_midi(out / "duet_performance.mid", [(names[0], 40, A), (names[1], 40, B)], bpm)
    gA, gB = amt.warp(A, beats, bpm, grid=4), amt.warp(B, beats, bpm, grid=4)
    amt.write_midi(out / "duet_gridded.mid", [(names[0], 40, gA), (names[1], 40, gB)], bpm)
    amt.write_musicxml(out / "duet.musicxml", [(names[0], "treble", gA), (names[1], "treble", gB)], bpm, unsure_below=0.5, unsure_style="color")


if __name__ == "__main__":
    main()
