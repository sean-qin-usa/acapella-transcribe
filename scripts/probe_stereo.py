"""Report stereo width and pitch-band energy per channel for a recording."""
import sys
import numpy as np
import soundfile as sf
import librosa

path = sys.argv[1]
y, sr = sf.read(path, always_2d=True)
L, R = y[:, 0], y[:, 1]
mid, side = (L + R) / 2, (L - R) / 2
print("duration %.1fs sr %d" % (len(L) / sr, sr))
print("L/R correlation: %.3f" % np.corrcoef(L, R)[0, 1])
print("side/mid energy: %.3f" % (np.sum(side**2) / np.sum(mid**2)))
print("L rms %.4f  R rms %.4f" % (np.sqrt(np.mean(L**2)), np.sqrt(np.mean(R**2))))
# per-time-frequency panning histogram: where does energy sit between L and R?
SL = np.abs(librosa.stft(L[: sr * 120].astype(np.float32), n_fft=4096, hop_length=1024))
SR = np.abs(librosa.stft(R[: sr * 120].astype(np.float32), n_fft=4096, hop_length=1024))
pan = (SR - SL) / (SR + SL + 1e-9)
w = (SL + SR) ** 2
f = librosa.fft_frequencies(sr=sr, n_fft=4096)
for lo, hi in [(60, 190), (190, 660), (660, 2600), (2600, 8000)]:
    m = (f >= lo) & (f < hi)
    h, _ = np.histogram(pan[m], bins=np.linspace(-1, 1, 12), weights=w[m])
    print("%5d-%5dHz pan hist (L..R): %s" % (lo, hi, " ".join("%2d" % round(100 * v) for v in h / h.sum())))
