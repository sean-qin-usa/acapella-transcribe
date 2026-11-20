"""Place a score dataset's notes on a recording's clock.

Reads <dataset>/score.mid, aligns it to the recording, and writes <dataset>/notes_aligned.csv
with one row per note: part, start_sec, end_sec, pitch_midi. This is the same shape as the
URMP and MusicNet note annotations, so the evaluation scripts can use it as ground truth.

usage: python -I scripts/align_dataset.py <dataset dir> <recording>
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402
import eval_score  # noqa: E402


def main():
    d, audio = Path(sys.argv[1]), sys.argv[2]
    parts = eval_score.read_score(d / "score.mid")
    y = amt.load_mono(audio)
    warp, cost = eval_score.align([n for _, notes in parts for n in notes], y, onset_w=1.0, smooth=9)
    with open(d / "notes_aligned.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part", "start_sec", "end_sec", "pitch_midi"])
        for name, notes in parts:
            for s, e, p in notes:
                w.writerow([name, "%.3f" % float(warp(s)), "%.3f" % float(warp(e)), p])
    meta_path = d / "score_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    end = max(e for _, notes in parts for _, e, _ in notes)
    meta["alignment"] = {"recording": Path(audio).name, "mean_cost": cost, "recording_span_sec": [round(float(warp(0.0)), 1), round(float(warp(end)), 1)], "recording_length_sec": round(len(y) / amt.SR, 1)}
    meta_path.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["alignment"], indent=1))


if __name__ == "__main__":
    main()
