"""When the detector writes a note an octave, twelfth or two octaves above another sounding
note, is it a second instrument or just an overtone of the lower note?

Measured on the URMP two-violin pieces. For each such candidate the recording and the
detector output give:
  stickout     how far the lower note's partials that coincide with the candidate rise above
               their neighbouring partials (a real upper note adds energy exactly there)
  amp_ratio    detector strength of the candidate relative to the lower note
  onset_late   how long after the lower note the candidate starts
  length_ratio candidate length relative to the lower note
  own_rise     jump in level at the candidate's partials at its start, beyond the jump in the
               lower note's other partials at that moment (an attack of its own)
A classifier is fitted on half the pieces with its cut-off set to keep 95% of real notes
there, then tested on the other half.

usage: python -I scripts/overtone_experiment.py <urmp_dir> <out_dir>
"""
import copy
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
import eval_urmp  # noqa: E402

FEATURES = ["stickout_db", "amp_ratio", "onset_late_s", "length_ratio", "own_rise_db"]


def collect(notes, y):
    """Candidates (note, features) for every note sitting on an overtone of another."""
    S, freqs, hop = amt.overtone_spectrogram(y)
    out = []
    for n, parent, ratio in amt.overtone_candidates(notes):
        f = amt.overtone_features(S, freqs, hop, n, parent, ratio)
        if f is not None:
            out.append((n, f))
    return out


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("Notes_1_vn_*.txt"))):
        mix, truth = eval_urmp.load_piece(folder, out / folder.name)
        y = amt.load_mono(mix)
        notes = amt.merge_held(amt.poly_notes(mix, fmin=185.0, fmax=4200.0, min_ms=58), y)
        rows = []
        for n, f in collect(notes, y):
            ov = [g for g in truth if g["pitch"] == n.pitch and min(g["end"], n.end) - max(g["start"], n.start) > 0]
            real = any(min(g["end"], n.end) - max(g["start"], n.start) > 0.5 * n.dur for g in ov)
            if real or not ov:
                rows.append(f + [int(real)])
        data.append(dict(name=folder.name, truth=truth, notes=notes, y=y, rows=np.array(rows)))
        r = data[-1]["rows"]
        print("%-28s candidates %4d (real upper note %d, overtone only %d)" % (folder.name, len(r), int(r[:, -1].sum()) if len(r) else 0, int((1 - r[:, -1]).sum()) if len(r) else 0))
    allr = np.vstack([d["rows"] for d in data if len(d["rows"])])
    print("\nall pieces: %d candidates | real %d | overtone only %d" % (len(allr), allr[:, -1].sum(), (1 - allr[:, -1]).sum()))
    for j, name in enumerate(FEATURES):
        print("  %-14s median for real %.2f, for overtone only %.2f" % (name, np.median(allr[allr[:, -1] == 1, j]), np.median(allr[allr[:, -1] == 0, j])))

    results = {}
    for label, train, test in (("fit on even pieces, test on odd", data[0::2], data[1::2]), ("fit on odd pieces, test on even", data[1::2], data[0::2])):
        tr = np.vstack([d["rows"] for d in train if len(d["rows"])])
        te = np.vstack([d["rows"] for d in test if len(d["rows"])])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(tr[:, :-1], tr[:, -1])
        cut = np.percentile(clf.predict_proba(tr[tr[:, -1] == 1, :-1])[:, 1], 5)  # keep 95% of real notes on the training half
        p = clf.predict_proba(te[:, :-1])[:, 1]
        kept, caught = (p[te[:, -1] == 1] >= cut).mean(), (p[te[:, -1] == 0] < cut).mean()
        tot = dict(truth=0, est0=0, hit0=0, est1=0, hit1=0, up=0, up0=0, up1=0)
        eval_urmp.TOL = 0.10
        for d in test:
            cands = collect(d["notes"], d["y"])
            drop = {id(n) for n, f in cands if clf.predict_proba(np.array([f]))[0, 1] < cut}
            after = [n for n in d["notes"] if id(n) not in drop]
            m0, m1 = eval_urmp.match(d["truth"], d["notes"]), eval_urmp.match(d["truth"], after)
            h0, h1 = {r for r, _ in m0}, {r for r, _ in m1}
            upper = [i for i, g in enumerate(d["truth"]) if g["with"] == "octave" and g["upper"]]
            tot["truth"] += len(d["truth"]); tot["est0"] += len(d["notes"]); tot["hit0"] += len(m0); tot["est1"] += len(after); tot["hit1"] += len(m1)
            tot["up"] += len(upper); tot["up0"] += sum(i in h0 for i in upper); tot["up1"] += sum(i in h1 for i in upper)
        print("\n%s" % label)
        print("  held-out candidates: real upper notes kept %.1f%%, overtones caught %.1f%%" % (100 * kept, 100 * caught))
        print("  whole pieces, before: found %.1f%%, written notes real %.1f%%, upper note of an octave found %.1f%%" % (100 * tot["hit0"] / tot["truth"], 100 * tot["hit0"] / tot["est0"], 100 * tot["up0"] / max(tot["up"], 1)))
        print("  whole pieces, after:  found %.1f%%, written notes real %.1f%%, upper note of an octave found %.1f%%" % (100 * tot["hit1"] / tot["truth"], 100 * tot["hit1"] / tot["est1"], 100 * tot["up1"] / max(tot["up"], 1)))
        sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
        print("  weights (per standard deviation): %s" % {n: round(float(w), 2) for n, w in zip(FEATURES, lr.coef_[0])})
        results[label] = dict(kept=float(kept), caught=float(caught), totals=tot)
    (out / "overtone_experiment.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
