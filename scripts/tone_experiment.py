"""Can the two violins be told apart by tone? Measured on the URMP two-violin pieces.

For every detected note that matches an annotated note we know the true player. A classifier
is trained on the first part of each piece and tested on the rest, using
  - the note's pitch only (how far "violin 1 plays higher" gets you),
  - the note's tone measured in the mix (what a real recording offers),
  - the note's tone measured on the player's isolated track (best case, no interference).
Then the tone evidence is added to the part splitter and the right-player rate is compared.

usage: python -I scripts/tone_experiment.py <urmp_dir> <out_dir> [train_share]
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
import duet  # noqa: E402
import eval_urmp  # noqa: E402

N_FFT, HOP = 4096, 512
GRID = np.geomspace(250.0, 8000.0, 24)


def spectrogram(y):
    return np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)), librosa.fft_frequencies(sr=amt.SR, n_fft=N_FFT)


def tone(S, freqs, note, others=()):
    """Spectral envelope of one note: level of each harmonic, laid on a fixed frequency grid,
    with the note's overall loudness removed. Harmonics that collide with a harmonic of another
    note sounding at the same time are skipped."""
    a, b = int(note.start * amt.SR / HOP), max(int(note.end * amt.SR / HOP), int(note.start * amt.SR / HOP) + 2)
    seg = S[:, a:b]
    if seg.shape[1] == 0:
        return np.full(len(GRID), np.nan)
    f0 = 440.0 * 2 ** ((note.pitch - 69) / 12)
    taken = [440.0 * 2 ** ((o.pitch - 69) / 12) * k for o in others for k in range(1, 25)]
    fs, lv = [], []
    for k in range(1, 41):
        f = f0 * k
        if f > 8000:
            break
        if any(abs(f - t) < 0.03 * f for t in taken):
            continue
        m = (freqs > f * 0.97) & (freqs < f * 1.03)
        if m.sum():
            fs.append(f)
            lv.append(20 * np.log10(np.median(seg[m].max(axis=0)) + 1e-7))
    if len(fs) < 3:
        return np.full(len(GRID), np.nan)
    fs, lv = np.array(fs), np.array(lv)
    env = np.interp(np.log(GRID), np.log(fs), lv, left=np.nan, right=np.nan)
    return env - np.nanmean(env)


def fill(X, mean=None):
    mean = np.nanmean(X, axis=0) if mean is None else mean
    mean = np.where(np.isfinite(mean), mean, 0.0)
    X = np.where(np.isfinite(X), X, mean)
    return X, mean


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    tone_w = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    rows, pooled = [], dict(n=0, pitch=0, mix=0, stem=0, split=0, fused=0)
    votes = dict(n=0, agree=0, agree_right=0, split_right_when_disagree=0, move_right=0)
    selfv = {k: dict(n=0, split_right=0, agree=0, agree_right=0, dis=0, dis_split_right=0) for k in ("all", "alone")}
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("Notes_1_vn_*.txt"))):
        mix, truth = eval_urmp.load_piece(folder, out / folder.name)
        y = amt.load_mono(mix)
        stems = [amt.load_mono(sorted(folder.glob("AuSep_%d_vn_*.wav" % k))[0]) for k in (1, 2)]
        S, freqs = spectrogram(y)
        Ss = [spectrogram(s)[0] for s in stems]
        notes = amt.poly_notes(mix, fmin=185.0, fmax=4200.0, min_ms=58)
        eval_urmp.TOL = 0.10
        pairs = eval_urmp.match(truth, notes)
        who = {e: truth[r]["player"] for r, e in pairs}
        idx = sorted(who, key=lambda e: notes[e].start)
        if len(idx) < 60:
            continue

        def overlapping(n):
            return [o for o in notes if o is not n and o.start < n.end and o.end > n.start]

        X_mix = np.array([tone(S, freqs, notes[e], overlapping(notes[e])) for e in idx])
        X_stem = np.array([tone(Ss[who[e]], freqs, notes[e]) for e in idx])
        X_pitch = np.array([[notes[e].pitch] for e in idx], float)
        lab = np.array([who[e] for e in idx])
        cut = int(share * len(idx))
        tr, te = slice(0, cut), slice(cut, None)
        if len(set(lab[tr])) < 2:
            continue
        res = {}
        models = {}
        for name, X in (("pitch", X_pitch), ("mix", X_mix), ("stem", X_stem)):
            Xtr, mean = fill(X[tr])
            Xte, _ = fill(X[te], mean)
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xtr, lab[tr])
            res[name] = float((clf.predict(Xte) == lab[te]).mean())
            models[name] = (clf, mean)

        # splitter alone vs splitter + tone evidence, judged on the same held-out notes
        clf, mean = models["mix"]
        Xall, _ = fill(X_mix, mean)
        logp = clf.predict_log_proba(Xall)
        A, B = duet.split_two_voices(list(notes))
        part = {id(n): 0 for n in A} | {id(n): 1 for n in B}
        agree = np.array([part[id(notes[e])] == who[e] for e in idx])
        res["split"] = float(max(agree[te].mean(), 1 - agree[te].mean()))
        # two independent votes: pitch movement (the splitter) and tone. Name the splitter's parts
        # from the training stretch, as a user labelling a known passage would.
        flip = agree[tr].mean() < 0.5
        vote_move = np.array([part[id(notes[e])] ^ int(flip) for e in idx])[te]
        vote_tone = clf.predict(Xall)[te]
        same = vote_move == vote_tone
        votes["n"] += len(same)
        votes["agree"] += int(same.sum())
        votes["agree_right"] += int((vote_move[same] == lab[te][same]).sum())
        votes["split_right_when_disagree"] += int((vote_move[~same] == lab[te][~same]).sum())
        votes["move_right"] += int((vote_move == lab[te]).sum())
        # no labels at all: learn tone from the notes where the split is safest (both violins
        # sounding at once), using the splitter's own answer as the label, then vote everywhere
        other_part = {0: B, 1: A}

        def both_sounding(n):
            return any(min(o.end, n.end) - max(o.start, n.start) > 0.5 * n.dur for o in other_part[part[id(n)]])

        safe = np.array([both_sounding(notes[e]) for e in idx])
        plab = np.array([part[id(notes[e])] for e in idx])
        if safe.sum() >= 40 and len(set(plab[safe])) == 2 and (~safe).sum() >= 10:
            Xs, m2 = fill(X_mix[safe])
            clf2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xs, plab[safe])
            tone2 = clf2.predict(fill(X_mix, m2)[0])
            truth_part = lab ^ int(agree.mean() < 0.5)  # truth expressed in the splitter's part names
            for name, mask in (("all", np.ones(len(idx), bool)), ("alone", ~safe)):
                same2 = (tone2 == plab) & mask
                diff2 = (tone2 != plab) & mask
                selfv[name]["n"] += int(mask.sum())
                selfv[name]["split_right"] += int((plab == truth_part)[mask].sum())
                selfv[name]["agree"] += int(same2.sum())
                selfv[name]["agree_right"] += int((plab == truth_part)[same2].sum())
                selfv[name]["dis"] += int(diff2.sum())
                selfv[name]["dis_split_right"] += int((plab == truth_part)[diff2].sum())
        evidence = {id(notes[e]): (-logp[i, 0], -logp[i, 1]) for i, e in enumerate(idx)}
        A2, B2 = duet.split_two_voices(list(notes), tone=evidence, tone_w=tone_w)
        part2 = {id(n): 0 for n in A2} | {id(n): 1 for n in B2}
        res["fused"] = float(np.mean([part2[id(notes[e])] == who[e] for e in idx[cut:]]))
        n_te = len(idx) - cut
        rows.append((folder.name, n_te, res))
        pooled["n"] += n_te
        for k in ("pitch", "mix", "stem", "split", "fused"):
            pooled[k] += res[k] * n_te
        print("%-28s test notes %4d | pitch only %.0f%% | tone in mix %.0f%% | tone on isolated track %.0f%% | splitter %.0f%% | splitter + tone %.0f%%" % (folder.name, n_te, *(100 * res[k] for k in ("pitch", "mix", "stem", "split", "fused"))))
    tot = {k: pooled[k] / pooled["n"] for k in ("pitch", "mix", "stem", "split", "fused")}
    print("\nALL PIECES (%d held-out notes, trained on first %.0f%% of each piece)" % (pooled["n"], 100 * share))
    print("  right player from pitch only:            %.1f%%" % (100 * tot["pitch"]))
    print("  right player from tone in the mix:       %.1f%%" % (100 * tot["mix"]))
    print("  right player from tone on isolated track: %.1f%%" % (100 * tot["stem"]))
    print("  current splitter:                        %.1f%%" % (100 * tot["split"]))
    print("  splitter + tone evidence:                %.1f%%" % (100 * tot["fused"]))
    dis = votes["n"] - votes["agree"]
    print("\nTWO VOTES (pitch movement, tone), parts named from the training stretch")
    print("  pitch movement alone right:        %.1f%%" % (100 * votes["move_right"] / votes["n"]))
    print("  votes agree on %.1f%% of notes; right when they agree: %.1f%%" % (100 * votes["agree"] / votes["n"], 100 * votes["agree_right"] / max(votes["agree"], 1)))
    print("  votes disagree on %.1f%% of notes; pitch movement right there: %.1f%%, tone right there: %.1f%%" % (100 * dis / votes["n"], 100 * votes["split_right_when_disagree"] / max(dis, 1), 100 * (1 - votes["split_right_when_disagree"] / max(dis, 1))))
    print("\nSELF-TAUGHT TONE VOTE (no labels; tone learned where both violins sound together)")
    for name, label in (("all", "all notes"), ("alone", "notes where one violin plays alone")):
        v = selfv[name]
        if v["n"]:
            print("  %s (%d): splitter right %.1f%% | votes agree on %.1f%%, right there %.1f%% | disagree on %.1f%%, splitter right there %.1f%%" % (label, v["n"], 100 * v["split_right"] / v["n"], 100 * v["agree"] / v["n"], 100 * v["agree_right"] / max(v["agree"], 1), 100 * v["dis"] / v["n"], 100 * v["dis_split_right"] / max(v["dis"], 1)))
    tot["votes"] = votes
    tot["self_taught"] = selfv
    (out / "tone_experiment.json").write_text(json.dumps(dict(train_share=share, pooled=tot, pieces=[(n, k, r) for n, k, r in rows]), indent=1))


if __name__ == "__main__":
    main()
