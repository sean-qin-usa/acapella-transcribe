"""When the detector writes the same pitch twice in a row, is the second note a new attack
or the first note still sounding?

Measured on the URMP two-violin pieces, whose annotations say which. For each back-to-back
pair of same-pitch detected notes, the recording is examined at that pitch's harmonics around
the second note's start:
  rise   the biggest frame-to-frame jump in level there (a new bow stroke or key strike)
  dip    how far the level falls just before it
A threshold is chosen on some pieces and tested on the others, then applied as a merge rule
to see what it does to the overall scores.

usage: python -I scripts/rearticulation_experiment.py <urmp_dir> <out_dir>
"""
import copy
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import eval_urmp  # noqa: E402

N_FFT, HOP = 2048, 256
GAP = 0.08


def level_track(S, freqs, pitch):
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    e = np.zeros(S.shape[1])
    for k in range(1, 7):
        if f0 * k > 9000:
            break
        m = (freqs > f0 * k * 0.97) & (freqs < f0 * k * 1.03)
        if m.sum():
            e += (S[m] ** 2).sum(axis=0)
    return 10 * np.log10(e + 1e-10)


def boundary_features(track, t, sr=amt.SR):
    """Rise and dip of a pitch's level around time t."""
    c = int(t * sr / HOP)
    a, b = max(c - 6, 1), min(c + 8, len(track) - 1)
    if b - a < 4:
        return None
    w = track[a:b]
    rise = float(np.max(w[2:] - w[:-2]))
    before = track[max(c - 14, 0) : max(c - 5, 1)]
    after = track[min(c + 6, len(track) - 1) : min(c + 16, len(track))]
    ref = min(before.mean() if len(before) else w.mean(), after.mean() if len(after) else w.mean())
    dip = float(ref - w.min())
    return rise, dip


def same_pitch_pairs(notes):
    last, pairs = {}, []
    for n in sorted(notes, key=lambda n: n.start):
        p = last.get(n.pitch)
        if p is not None and n.start - p.end <= GAP:
            pairs.append((p, n))
        last[n.pitch] = n
    return pairs


def merge_continuations(notes, S, freqs, rise_min, dip_min):
    """Join a note onto the previous same-pitch note unless the recording shows a new attack."""
    tracks, out, last = {}, [], {}
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        p = last.get(n.pitch)
        if p is not None and n.start - p.end <= GAP:
            if n.pitch not in tracks:
                tracks[n.pitch] = level_track(S, freqs, n.pitch)
            f = boundary_features(tracks[n.pitch], n.start)
            if f is not None and f[0] < rise_min and f[1] < dip_min:
                p.end = max(p.end, n.end)
                continue
        out.append(n)
        last[n.pitch] = n
    return out


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    pieces = sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("Notes_1_vn_*.txt")))
    data = []
    for folder in pieces:
        mix, truth = eval_urmp.load_piece(folder, out / folder.name)
        y = amt.load_mono(mix)
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
        freqs = librosa.fft_frequencies(sr=amt.SR, n_fft=N_FFT)
        notes = amt.poly_notes(mix, fmin=185.0, fmax=4200.0, min_ms=58)
        rows, tracks = [], {}
        for a, b in same_pitch_pairs(notes):
            if b.pitch not in tracks:
                tracks[b.pitch] = level_track(S, freqs, b.pitch)
            f = boundary_features(tracks[b.pitch], b.start)
            if f is None:
                continue
            new = any(g["pitch"] == b.pitch and abs(g["start"] - b.start) <= 0.10 for g in truth)
            held = any(g["pitch"] == b.pitch and g["start"] < b.start - 0.10 and g["end"] > b.start for g in truth)
            if new != held:
                rows.append((f[0], f[1], int(new)))
        data.append(dict(name=folder.name, truth=truth, notes=notes, S=S, freqs=freqs, rows=np.array(rows)))
        r = data[-1]["rows"]
        print("%-28s same-pitch pairs with a clear answer: %4d (new attack %d, continuation %d)" % (folder.name, len(r), int(r[:, 2].sum()) if len(r) else 0, int((1 - r[:, 2]).sum()) if len(r) else 0))

    allrows = np.vstack([d["rows"] for d in data if len(d["rows"])])
    print("\nall pieces: %d pairs | new attacks %d | continuations %d" % (len(allrows), allrows[:, 2].sum(), (1 - allrows[:, 2]).sum()))
    for j, name in ((0, "rise (dB)"), (1, "dip (dB)")):
        print("  %-10s median for new attacks %.1f, for continuations %.1f" % (name, np.median(allrows[allrows[:, 2] == 1, j]), np.median(allrows[allrows[:, 2] == 0, j])))

    # choose thresholds on the even-numbered pieces, test on the odd-numbered ones, and the reverse
    def fit(rows):
        best = (0, 0, 0)
        for r in np.percentile(rows[:, 0], np.arange(5, 96, 5)):
            for dmin in np.percentile(rows[:, 1], np.arange(5, 96, 5)):
                pred = ((rows[:, 0] >= r) | (rows[:, 1] >= dmin)).astype(int)
                acc = 0.5 * ((pred[rows[:, 2] == 1] == 1).mean() + (pred[rows[:, 2] == 0] == 0).mean())
                best = max(best, (acc, r, dmin))
        return best[1], best[2]

    results = {}
    for label, train, test in (("fit on even pieces, test on odd", data[0::2], data[1::2]), ("fit on odd pieces, test on even", data[1::2], data[0::2])):
        r, dmin = fit(np.vstack([d["rows"] for d in train if len(d["rows"])]))
        te = np.vstack([d["rows"] for d in test if len(d["rows"])])
        pred = ((te[:, 0] >= r) | (te[:, 1] >= dmin)).astype(int)
        keep_new, drop_cont = (pred[te[:, 2] == 1] == 1).mean(), (pred[te[:, 2] == 0] == 0).mean()
        tot = dict(truth=0, est0=0, hit0=0, est1=0, hit1=0)
        for d in test:
            merged = merge_continuations(copy.deepcopy(d["notes"]), d["S"], d["freqs"], r, dmin)
            eval_urmp.TOL = 0.10
            tot["truth"] += len(d["truth"])
            tot["est0"] += len(d["notes"])
            tot["hit0"] += len(eval_urmp.match(d["truth"], d["notes"]))
            tot["est1"] += len(merged)
            tot["hit1"] += len(eval_urmp.match(d["truth"], merged))
        print("\n%s (new attack if rise >= %.1f dB or dip >= %.1f dB)" % (label, r, dmin))
        print("  on held-out pairs: real new attacks kept %.1f%%, continuations caught %.1f%%" % (100 * keep_new, 100 * drop_cont))
        print("  whole pieces, before: found %.1f%% of real notes, %.1f%% of written notes real" % (100 * tot["hit0"] / tot["truth"], 100 * tot["hit0"] / tot["est0"]))
        print("  whole pieces, after:  found %.1f%% of real notes, %.1f%% of written notes real" % (100 * tot["hit1"] / tot["truth"], 100 * tot["hit1"] / tot["est1"]))
        results[label] = dict(rise_min=float(r), dip_min=float(dmin), kept_new=float(keep_new), caught=float(drop_cont), before=[tot["hit0"] / tot["truth"], tot["hit0"] / tot["est0"]], after=[tot["hit1"] / tot["truth"], tot["hit1"] / tot["est1"]])
    (out / "rearticulation_experiment.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
