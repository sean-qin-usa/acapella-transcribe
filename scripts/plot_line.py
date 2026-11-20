"""Plot a cached pitch curve with the notes cut from it, to eyeball segmentation.

usage: python -I scripts/plot_line.py <f0.npz> <out.png> <start_sec> <end_sec> [more start/end pairs]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt  # noqa: E402

d = np.load(sys.argv[1])
midi, rms = d["midi"], d["rms"]
notes, tuning = amt.segment_notes(midi, rms, d["prob"])
t = np.arange(len(midi)) * amt.HOP / amt.SR
spans = [(float(a), float(b)) for a, b in zip(sys.argv[3::2], sys.argv[4::2])]
fig, axes = plt.subplots(len(spans), 1, figsize=(16, 3.6 * len(spans)), squeeze=False)
for ax, (a, b) in zip(axes[:, 0], spans):
    m = (t >= a) & (t < b)
    ax.plot(t[m], midi[m] - tuning, ".", ms=2, color="#888888")
    for n in notes:
        if n.end > a and n.start < b:
            ax.hlines(n.pitch, n.start, n.end, color="#c0392b" if n.conf >= 0.5 else "#e0a030", lw=3)
            ax.text(n.start, n.pitch + 0.5, amt.name(n.pitch), fontsize=7)
    vis = midi[m][np.isfinite(midi[m])]
    if len(vis):
        ax.set_ylim(np.percentile(vis, 1) - 3, np.percentile(vis, 99) + 3)
    ax.set_xlim(a, b)
    ax.set_yticks(range(int(ax.get_ylim()[0]), int(ax.get_ylim()[1]) + 1))
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25)
    ax.set_title("%.0f-%.0f s   grey = tracked pitch, red = notes" % (a, b), fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=80)
print(len(notes), "notes; median dur %.2f" % np.median([n.dur for n in notes]))
