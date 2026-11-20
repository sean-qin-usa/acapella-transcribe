"""Turn separated stems of a pop song into a lead line, backing harmonies, bass line and chords.

usage: python -I scripts/pop.py <stems_dir> <out_dir> [--title NAME]
Expects lead.flac, backing.flac, bass.flac, drums.flac, other.flac in <stems_dir>.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # isolated mode drops the script folder
import amt  # noqa: E402


def beat_subdivision(y, beats, sr=amt.SR):
    """2 if the beat splits in halves (straight), 3 if it splits in thirds (swing / 12-8)."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=256)
    score = {}
    for div in (2, 3):
        pts = []
        for a, b in zip(beats[:-1], beats[1:]):
            pts += [a + (b - a) * k / div for k in range(1, div)]
        score[div] = float(np.interp(pts, t, env).mean())
    off = []
    for a, b in zip(beats[:-1], beats[1:]):
        off += [a + (b - a) * k / 12 for k in (1, 5, 7, 11)]
    floor = float(np.interp(off, t, env).mean())
    return (3 if score[3] > score[2] else 2), {"halves": score[2], "thirds": score[3], "elsewhere": floor}


def downbeat_phase(bass, beats, per_bar=4, sr=amt.SR):
    env = librosa.onset.onset_strength(y=bass, sr=sr, hop_length=256)
    t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=256)
    at = np.interp(beats, t, env)
    strength = [float(at[p::per_bar].mean()) for p in range(per_bar)]
    return int(np.argmax(strength)), strength


def sections(notes, gap=3.0):
    """Stretches of continuous singing, with a rough sung-vs-spoken indicator."""
    out, cur = [], []
    for n in notes:
        if cur and n.start - cur[-1].end > gap:
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    rows = []
    for s in out:
        d = np.array([n.dur for n in s])
        rows.append(
            {
                "from": round(s[0].start, 1),
                "to": round(s[-1].end, 1),
                "notes": len(s),
                "range": "%s-%s" % (amt.name(min(n.pitch for n in s)), amt.name(max(n.pitch for n in s))),
                "median_note_sec": round(float(np.median(d)), 2),
                "long_note_share": round(float((d > 0.3).mean()), 2),
            }
        )
    return rows


def in_key_share(notes, key):
    tonic, mode = key.split()
    root = amt.PITCH_NAMES.index(tonic)
    scale = {(root + k) % 12 for k in ((0, 2, 4, 5, 7, 9, 11) if mode == "major" else (0, 2, 3, 5, 7, 8, 10))}
    w = sum(n.dur for n in notes)
    return float(sum(n.dur for n in notes if n.pitch % 12 in scale) / w) if w else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems")
    ap.add_argument("out")
    ap.add_argument("--title", default="song")
    ap.add_argument("--unsure", type=float, default=0.3, help="lead notes below this confidence are treated as unclear")
    ap.add_argument("--no-autotune", action="store_true", help="keep off-scale lead notes exactly as sung")
    args = ap.parse_args()
    stems, out = Path(args.stems), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    y = {k: amt.load_mono(stems / (k + ".flac")) for k in ("lead", "backing", "bass", "drums", "other")}

    # rhythm: beats from the drums, feel from the full backing track, bar start from the bass
    beats, bpm, jitter = amt.beat_grid(y["drums"] + 0.5 * y["bass"])
    div, div_scores = beat_subdivision(y["drums"] + y["other"], beats)
    phase, phase_strength = downbeat_phase(y["bass"], beats)
    beats = beats[phase:]

    # harmony
    chords, chroma_mean = amt.beat_chords(y["other"] + y["bass"], beats, per=2)
    key, key_conf = amt.estimate_key(chroma_mean)

    # parts
    lead, lead_tuning = amt.mono_notes(y["lead"], fmin=65, fmax=1000, cache=out / "f0_lead.npz")
    lead_found = len(lead)
    unsure_time = sum(n.dur for n in lead if n.conf < args.unsure) / max(sum(n.dur for n in lead), 1e-9)
    lead, settle = amt.settle_uncertain(lead, args.unsure)
    # scale: where the clearly sung notes and the instruments agree
    w_lead = amt.pitch_class_weights([n for n in lead if n.conf >= args.unsure])
    scale, scale_name, scale_fit, runner_up = amt.best_scale(w_lead / (w_lead.sum() + 1e-9) + chroma_mean / (chroma_mean.sum() + 1e-9))
    off_scale_before = sum(n.dur for n in lead if n.pitch % 12 not in scale) / max(sum(n.dur for n in lead), 1e-9)
    tuned = 0
    if not args.no_autotune:
        lead, tuned = amt.autotune(lead, scale)
    settle.update(scale=scale_name, scale_fit=scale_fit, next_best_scale_fit=runner_up, off_scale_share_before_autotune=off_scale_before, notes_autotuned=tuned)
    bass, _ = amt.mono_notes(y["bass"], fmin=30, fmax=260, cache=out / "f0_bass.npz", min_dur=0.15)
    backing = amt.poly_notes(stems / "backing.flac", fmin=80, fmax=1100, onset=0.4, frame=0.2, min_ms=100)
    pad = amt.pad_from_harmony(y["other"] + 0.5 * y["bass"], beats)
    keys = amt.poly_notes(stems / "other.flac", fmin=60, fmax=2000, onset=0.6, frame=0.4, min_ms=150)
    drums = amt.drum_hits(y["drums"])

    # notation timeline: straight feel = 16th grid; swung feel = 12/8 with an eighth-note grid
    if div == 2:
        q_per_beat, grid, meter = 1.0, 4, "4/4"
    else:
        q_per_beat, grid, meter = 1.5, 6, "12/8"
    qbpm = bpm * q_per_beat
    g = {k: amt.warp(v, beats, bpm, grid=grid) for k, v in (("lead", lead), ("backing", backing), ("bass", bass), ("keys", keys), ("drums", drums))}
    gpad = [[n for n in amt.warp(v, beats, bpm, grid=grid) if n.start >= 0] for v in pad]
    for part in g.values():  # drop anything before bar 1
        part[:] = [n for n in part if n.start >= 0]
    amt.write_check(out / "check_pad.wav", y["other"], [n for v in pad for n in v])  # instruments left, pad right
    for k in ("lead", "bass", "backing"):  # recording left, transcription right
        amt.write_check(out / ("check_%s.wav" % k), y[k], {"lead": lead, "bass": bass, "backing": backing}[k], shift=12 if k == "bass" else 0)

    amt.write_midi(
        out / (args.title + ".mid"),
        [
            ("Lead vocal", 52, [n for n in g["lead"] if n.conf >= args.unsure]),
            ("Lead vocal (unclear pitch)", 52, [n for n in g["lead"] if n.conf < args.unsure]),
            ("Backing vocals (as recorded)", 53, g["backing"]),
            *[("Backing pad %d (from instruments)" % (len(gpad) - i), 52, v) for i, v in enumerate(reversed(gpad))],
            ("Bass", 33, g["bass"]),
            ("Instruments (reference)", 0, g["keys"]),
            ("Drums", "drums", g["drums"]),
        ],
        qbpm,
    )
    amt.write_midi(
        out / (args.title + "_performance.mid"),
        [("Lead vocal", 52, lead), ("Backing vocals", 53, backing), ("Bass", 33, bass), ("Instruments (reference)", 0, keys), ("Drums", "drums", drums)],
        bpm,
    )
    amt.write_musicxml(
        out / (args.title + ".musicxml"),
        [("Lead", "treble8vb", g["lead"]), ("Backing (as recorded)", "treble8vb", g["backing"])]
        + [("Pad %d (from instruments)" % (len(gpad) - i), "treble8vb", v) for i, v in enumerate(reversed(gpad))]
        + [("Bass", "bass", g["bass"])],
        qbpm,
        key=key,
        meter=meter,
        unsure_below=args.unsure,
    )

    bars = [chords[i : i + 2] for i in range(0, len(chords), 2)]
    lines = ["%s  key: %s  tempo: %.1f bpm  feel: %s" % (args.title, key, bpm, "straight" if div == 2 else "swung/12-8"), ""]
    for i in range(0, len(bars), 4):
        tsec = beats[min(i * 4, len(beats) - 1)]
        row = " | ".join(" ".join(b) if len(set(b)) > 1 else b[0] for b in bars[i : i + 4])
        lines.append("bar %3d (%3d:%02d)  | %s |" % (i + 1, tsec // 60, tsec % 60, row))
    (out / "chords.txt").write_text("\n".join(lines) + "\n")

    key_notes, key_notes_conf = amt.estimate_key(amt.pitch_class_weights(lead + bass + backing))
    report = {
        "key_from_transcribed_notes": key_notes,
        "key_from_notes_confidence": key_notes_conf,
        "tempo_bpm": bpm,
        "beat_jitter": jitter,
        "beat_subdivision": div,
        "subdivision_scores": div_scores,
        "downbeat_phase_strength": phase_strength,
        "key": key,
        "key_confidence": key_conf,
        "lead_tuning_offset_semitones": lead_tuning,
        "lead_confidence": {"notes_found": lead_found, "unclear_share_of_singing_time": unsure_time, "threshold": args.unsure, **settle, "unclear_notes_left": sum(n.conf < args.unsure for n in lead)},
        "lead": {"notes": len(lead), "in_key_share": in_key_share(lead, key), "chroma_agreement": amt.chroma_agreement(lead, y["lead"])},
        "bass": {"notes": len(bass), "in_key_share": in_key_share(bass, key), "chroma_agreement": amt.chroma_agreement(bass, y["bass"])},
        "backing": {
            "notes": len(backing),
            "in_key_share": in_key_share(backing, key),
            "chroma_agreement": amt.chroma_agreement(backing, y["backing"]),
            "simultaneous_notes_share": amt.polyphony_profile(backing),
        },
        "drums": {"kick": sum(n.pitch == 36 for n in drums), "snare": sum(n.pitch == 38 for n in drums), "hihat": sum(n.pitch == 42 for n in drums)},
        "pad_from_instruments": {"voices": len(pad), "notes_per_voice": [len(v) for v in pad], "share_of_song_covered": float(sum(n.dur for n in pad[0]) / (len(y["other"]) / amt.SR)) if pad and pad[0] else 0.0},
        "backing_vocals_active_from_sec": round(min((n.start for n in backing), default=-1.0), 1),
        "lead_sections": sections(lead),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
