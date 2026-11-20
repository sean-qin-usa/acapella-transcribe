"""Can struck notes (piano) be told from bowed notes (violin) one note at a time?

Uses a score dataset as the answer key: a detected note is labelled by which staff of the
score has that pitch starting at that moment. Features come from the recording alone and
follow from how the instruments make sound: a struck string decays, a bowed one is held and
its pitch wobbles (vibrato).

usage: python -I scripts/instrument_experiment.py <dataset dir> <recording> [--partial]
The dataset's score.mid must name the bowed parts with "violin" and the struck one "piano".
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import eval_score  # noqa: E402

N_FFT, HOP = 4096, 512
FEATURES = ["decay_db_per_s", "end_minus_start_db", "attack_db", "pitch_wobble_cents", "length_s"]


def note_features(S, freqs, note, sr=amt.SR):
    a = int(note.start * sr / HOP)
    b = max(int(note.end * sr / HOP), a + 4)
    seg = S[:, a:b]
    if seg.shape[1] < 4:
        return None
    f0 = 440.0 * 2 ** ((note.pitch - 69) / 12)
    track = np.zeros(seg.shape[1])
    for k in range(1, 7):
        m = (freqs > f0 * k * 0.97) & (freqs < f0 * k * 1.03)
        if m.sum() and f0 * k < 9000:
            track += (seg[m] ** 2).sum(axis=0)
    db = 10 * np.log10(track + 1e-10)
    t = np.arange(len(db)) * HOP / sr
    slope = float(np.polyfit(t, db, 1)[0])
    third = max(len(db) // 3, 1)
    m1 = (freqs > f0 * 0.97) & (freqs < f0 * 1.03)
    if m1.sum() >= 2:
        peak = freqs[m1][seg[m1].argmax(axis=0)]
        loud = seg[m1].max(axis=0) > 0.3 * seg[m1].max()
        wobble = float(np.std(1200 * np.log2(peak[loud] / f0))) if loud.sum() >= 3 else 0.0
    else:
        wobble = 0.0
    return [slope, float(db[-third:].mean() - db[:third].mean()), float(db[:2].max() - db.mean()), wobble, float(note.end - note.start)]


def main():
    d, audio = Path(sys.argv[1]), sys.argv[2]
    partial = "--partial" in sys.argv[3:]
    parts = eval_score.read_score(d / "score.mid")
    y = amt.load_mono(audio)
    warp, _ = eval_score.align([n for _, notes in parts for n in notes], y, onset_w=1.0, smooth=9, partial=partial)
    lo, hi = warp.span
    truth = {"violin": [], "piano": []}
    for name, notes in parts:
        kind = "piano" if "piano" in name.lower() else "violin"
        truth[kind] += [(float(warp(s)), p) for s, _, p in notes]
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=amt.SR, n_fft=N_FFT)
    det = [n for n in amt.poly_notes(audio, fmin=185.0, fmax=4200.0, min_ms=58) if lo <= n.start <= hi]

    def near(kind, n, tol=0.3):
        return any(p == n.pitch and abs(t - n.start) <= tol for t, p in truth[kind])

    X, lab, when = [], [], []
    tally = {"violin only": 0, "piano only": 0, "both": 0, "neither": 0}
    for n in det:
        v, p = near("violin", n), near("piano", n)
        tally["both" if v and p else "violin only" if v else "piano only" if p else "neither"] += 1
        if v == p:
            continue
        f = note_features(S, freqs, n)
        if f is None:
            continue
        X.append(f)
        lab.append(1 if v else 0)
        when.append(n.start)
    X, lab, when = np.array(X), np.array(lab), np.array(when)
    print("detected notes in the scored span: %d | %s" % (len(det), tally))
    print("usable labelled notes: %d violin, %d piano" % (lab.sum(), (1 - lab).sum()))
    cut = np.median(when)
    tr, te = when < cut, when >= cut
    print("\nfeature               violin median   piano median   one-feature accuracy on the later half")
    for j, name in enumerate(FEATURES):
        best = (0, 0, 1)
        for thr in np.percentile(X[tr, j], np.arange(5, 96, 5)):
            for sign in (1, -1):
                acc = (((X[tr, j] - thr) * sign > 0).astype(int) == lab[tr]).mean()
                best = max(best, (acc, thr, sign))
        _, thr, sign = best
        acc = (((X[te, j] - thr) * sign > 0).astype(int) == lab[te]).mean()
        print("%-20s %12.2f %14.2f %12.1f%%   (violin if %s %.2f)" % (name, np.median(X[lab == 1, j]), np.median(X[lab == 0, j]), 100 * acc, ">" if sign > 0 else "<", thr))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(X[tr], lab[tr])
    pred = clf.predict(X[te])
    print("\nall five together, trained on the earlier half, tested on the later half (%d notes):" % te.sum())
    print("  right overall %.1f%% | violin notes kept %.1f%% | piano notes removed %.1f%%" % (100 * (pred == lab[te]).mean(), 100 * pred[lab[te] == 1].mean(), 100 * (1 - pred[lab[te] == 0]).mean()))
    print("  always guessing the commoner class would score %.1f%%" % (100 * max(lab[te].mean(), 1 - lab[te].mean())))
    (d / "instrument_experiment.json").write_text(json.dumps({"tally": tally, "features": FEATURES, "test_accuracy": float((pred == lab[te]).mean())}, indent=1))


if __name__ == "__main__":
    main()
