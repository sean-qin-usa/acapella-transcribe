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
    det = amt.merge_held(amt.poly_notes(audio, fmin=185.0, fmax=4200.0, min_ms=58), y)
    det = [n for n in det if lo <= n.start <= hi]
    pans = amt.pan_per_note(audio, det)
    centre = float(np.median(pans)) if pans is not None else 0.0

    def near(kind, n, tol=0.3):
        return any(p == n.pitch and abs(t - n.start) <= tol for t, p in truth[kind])

    X, lab, when = [], [], []
    tally = {"violin only": 0, "piano only": 0, "both": 0, "neither": 0}
    for i, n in enumerate(det):
        v, p = near("violin", n), near("piano", n)
        tally["both" if v and p else "violin only" if v else "piano only" if p else "neither"] += 1
        if v == p:
            continue
        f = note_features(S, freqs, n)
        if f is None:
            continue
        pan = pans[i] if pans is not None else 0.0
        X.append(f + [pan, abs(pan - centre), float(n.pitch)])
        lab.append(1 if v else 0)
        when.append(n.start)
    X, lab, when = np.array(X), np.array(lab), np.array(when)
    names = FEATURES + ["right_minus_left_db", "distance_from_centre_db", "pitch"]
    print("detected notes in the scored span: %d | %s" % (len(det), tally))
    print("usable labelled notes: %d violin, %d piano | centre of the stereo image %+.1f dB" % (lab.sum(), (1 - lab).sum(), centre))
    cut = np.median(when)
    tr, te = when < cut, when >= cut
    print("always guessing the commoner class on the later half: %.1f%%" % (100 * max(lab[te].mean(), 1 - lab[te].mean())))
    print("\nfeature                    violin median   piano median")
    for j, name in enumerate(names):
        print("%-26s %12.2f %14.2f" % (name, np.median(X[lab == 1, j]), np.median(X[lab == 0, j])))
    results = {}
    sets = {"struck or bowed (5 features)": [0, 1, 2, 3, 4], "position only": [5, 6], "pitch only": [7], "struck or bowed + position": [0, 1, 2, 3, 4, 5, 6], "all eight": list(range(8))}
    print("\ntrained on the earlier half, tested on the later half (%d notes):" % te.sum())
    from sklearn.ensemble import GradientBoostingClassifier

    for label, cols in sets.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(X[tr][:, cols], lab[tr])
        pred = clf.predict(X[te][:, cols])
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(X[tr][:, cols], lab[tr])
        pg = gb.predict(X[te][:, cols])
        results[label] = float((pred == lab[te]).mean())
        print("  %-30s straight-line model %.1f%% (violin kept %.0f%%, piano removed %.0f%%) | tree model %.1f%% (kept %.0f%%, removed %.0f%%)" % (label, 100 * (pred == lab[te]).mean(), 100 * pred[lab[te] == 1].mean(), 100 * (1 - pred[lab[te] == 0]).mean(), 100 * (pg == lab[te]).mean(), 100 * pg[lab[te] == 1].mean(), 100 * (1 - pg[lab[te] == 0]).mean()))
    (d / "instrument_experiment.json").write_text(json.dumps({"tally": tally, "features": names, "test_accuracy": results}, indent=1))


if __name__ == "__main__":
    main()
