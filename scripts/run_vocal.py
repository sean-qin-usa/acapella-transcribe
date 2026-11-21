"""Run two published models for unaccompanied vocal ensembles on a recording.

  notes   Late/Deep CNN of Cuesta, McFee and Gomez, "Multiple F0 Estimation in Vocal Ensembles
          using Convolutional Neural Networks" (ISMIR 2020; MIT licence), which turns a
          recording into a map of which pitches are being sung at each moment.
  voices  VoasCNN of Cuesta and Gomez, "Voice Assignment in Vocal Quartets using Deep Learning
          Models based on Pitch Salience" (TISMIR 2022; MIT licence), which divides that map
          among soprano, alto, tenor and bass.

Both networks are rebuilt here layer for layer from the authors' code and filled with the
authors' trained weights, which are expected under tools/vocal_multif0/ and tools/vocal_voas/
(their GitHub repositories). The sound is prepared exactly as in training, with the pumpp
library, and that needs the older libraries in .venv_vocal.

Writes <out.mid> with every note on one track, and <out stem>_satb.mid with one track per voice.

usage: .venv_vocal/bin/python scripts/run_vocal.py <recording> <out.mid> [<recording> <out.mid> ...]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NOTES_WEIGHTS = ROOT / "tools" / "vocal_multif0" / "models" / "exp3multif0.pkl"  # an HDF5 file despite the name
VOICES_WEIGHTS = ROOT / "tools" / "vocal_voas" / "models" / "voas_cnn.h5"
SR, HOP, FMIN, BINS_PER_SEMITONE, OCTAVES = 22050, 256, 32.7, 5, 6
NOTE_THRESHOLD = 0.5  # the authors' value for this network
VOICE_THRESHOLDS = (0.23, 0.17, 0.15, 0.17)  # the authors' values: soprano, alto, tenor, bass
VOICES = ("Soprano", "Alto", "Tenor", "Bass")
PATCH = 128
GAP_FRAMES, MIN_FRAMES = 4, 9  # close gaps up to 46 ms; drop notes shorter than 104 ms


def notes_network():
    from tensorflow.keras import Model
    from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Input

    count = [0]

    def norm(x):
        count[0] += 1
        return BatchNormalization(name="batch_normalization_%d" % count[0])(x)

    def branch(x, letter):
        x = norm(x)
        for name, filters, kernel in (("conv1", 16, (5, 5)), ("conv2", 32, (5, 5)), ("conv3", 32, (5, 5)), ("conv4", 32, (5, 5)), ("harm1", 32, (70, 3)), ("harm2", 32, (70, 3))):
            x = norm(Conv2D(filters, kernel, padding="same", activation="relu", name=name + letter)(x))
        return x

    loudness, phase = Input(shape=(None, None, 5)), Input(shape=(None, None, 5))
    x = Concatenate()([branch(loudness, "a"), branch(phase, "b")])
    x = norm(Conv2D(64, (3, 3), padding="same", activation="relu", name="conv7")(x))
    x = norm(Conv2D(64, (3, 3), padding="same", activation="relu", name="conv8")(x))
    x = norm(Conv2D(8, (360, 1), padding="same", activation="relu", name="distribution")(x))
    x = Conv2D(1, (1, 1), padding="same", activation="sigmoid", name="squishy")(x)
    model = Model(inputs=[loudness, phase], outputs=x)
    model.load_weights(str(NOTES_WEIGHTS), by_name=True)
    return model


def voices_network():
    from tensorflow.keras import Model
    from tensorflow.keras.layers import BatchNormalization, Conv2D, Input

    count = [0]

    def norm(x):
        name = "batch_normalization" + ("_%d" % count[0] if count[0] else "")
        count[0] += 1
        return BatchNormalization(name=name)(x)

    def conv(x, name, filters=16, kernel=(3, 3), activation="relu"):
        return Conv2D(filters, kernel, padding="same", activation=activation, name=name)(x)

    x_in = Input(shape=(360, PATCH, 1))
    x = norm(x_in)
    x = norm(conv(x, "conv1", 32))
    x = norm(conv(x, "conv2", 32))
    x = norm(conv(x, "conv_harm_1", 16, (70, 3)))
    x = norm(conv(x, "conv_harm_2", 16, (70, 3)))
    outs = []
    firsts = [norm(conv(x, "conv%da" % (i + 1))) for i in range(4)]
    for i, voice in enumerate(("soprano", "alto", "tenor", "bass")):
        outs.append(conv(conv(firsts[i], "conv%db" % (i + 1)), "conv_" + voice, 1, (1, 1), "sigmoid"))
    model = Model(inputs=x_in, outputs=outs)
    model.load_weights(str(VOICES_WEIGHTS), by_name=True)
    return model


def check_loaded(model, path):
    """by_name loading skips layers silently, so make sure every saved layer found its place."""
    import h5py

    with h5py.File(path, "r") as f:
        g = f["model_weights"]
        saved = {n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"] if len(g[n].attrs["weight_names"])}
    built = {layer.name for layer in model.layers if layer.weights}
    if saved != built:
        raise SystemExit("weights and network do not match: only in file %s, only in network %s" % (sorted(saved - built), sorted(built - saved)))


def pitch_map(model, path):
    """(360 pitch steps x frames) map of how strongly each pitch is being sung, 0..1."""
    import pumpp

    feature = pumpp.feature.HCQTPhaseDiff(name="dphase", sr=SR, hop_length=HOP, fmin=FMIN, n_octaves=OCTAVES, over_sample=BINS_PER_SEMITONE, harmonics=[1, 2, 3, 4, 5], log=True)
    data = pumpp.Pump(feature)(audio_f=str(path))
    loud = data["dphase/mag"][0].transpose(1, 0, 2)[np.newaxis]  # (1, pitch, time, harmonic)
    phase = data["dphase/dphase"][0].transpose(1, 0, 2)[np.newaxis]
    out = [model.predict([loud[:, :, t : t + 5000], phase[:, :, t : t + 5000]], verbose=0)[0, :, :, 0] for t in range(0, loud.shape[2], 5000)]
    return np.hstack(out).astype(np.float32)


def voice_maps(model, pitches):
    """Four maps like the input, one per voice."""
    n = pitches.shape[1]
    padded = np.zeros((360, int(np.ceil(n / PATCH)) * PATCH), np.float32)
    padded[:, :n] = pitches
    chunks = np.stack([padded[:, t : t + PATCH] for t in range(0, padded.shape[1], PATCH)])[..., np.newaxis]
    out = model.predict(chunks, batch_size=8, verbose=0)
    return [np.hstack(list(o[..., 0]))[:, :n] for o in out]


def semitone(step):
    return int(round(69 + 12 * np.log2(FMIN * 2 ** (step / (12 * BINS_PER_SEMITONE)) / 440.0)))


def runs_to_notes(active, strength):
    """active: {pitch: boolean array over frames} -> [(start, end, pitch, strength)]."""
    notes = []
    for pitch, on in active.items():
        idx = np.flatnonzero(on)
        if not len(idx):
            continue
        breaks = np.flatnonzero(np.diff(idx) > GAP_FRAMES + 1)
        for a, b in zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]]):
            if b - a + 1 >= MIN_FRAMES:
                notes.append((a * HOP / SR, (b + 1) * HOP / SR, pitch, float(np.mean(strength[pitch][a : b + 1][on[a : b + 1]]))))
    return sorted(notes)


def notes_from_map(pitches):
    """Every pitch that stands out at each moment (a peak along the pitch axis above the threshold)."""
    peak = (pitches >= NOTE_THRESHOLD) & (pitches > np.roll(pitches, 1, axis=0)) & (pitches >= np.roll(pitches, -1, axis=0))
    active, strength = {}, {}
    for step, t in zip(*np.nonzero(peak)):
        p = semitone(step)
        if p not in active:
            active[p], strength[p] = np.zeros(pitches.shape[1], bool), np.zeros(pitches.shape[1], np.float32)
        active[p][t] = True
        strength[p][t] = max(strength[p][t], pitches[step, t])
    return runs_to_notes(active, strength)


def notes_from_voice(voice_map, threshold):
    """One line: the strongest pitch at each moment, where it is strong enough."""
    top = voice_map.argmax(axis=0)
    value = voice_map.max(axis=0)
    active, strength = {}, {}
    for t in np.flatnonzero((value >= threshold) & (top > 0)):
        p = semitone(top[t])
        if p not in active:
            active[p], strength[p] = np.zeros(voice_map.shape[1], bool), np.zeros(voice_map.shape[1], np.float32)
        active[p][t] = True
        strength[p][t] = value[t]
    return runs_to_notes(active, strength)


def write_midi(path, tracks):
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    for name, notes in tracks:
        inst = pretty_midi.Instrument(program=52, name=name)  # choir aahs
        inst.notes = [pretty_midi.Note(velocity=int(40 + 70 * min(v, 1.0)), pitch=p, start=s, end=e) for s, e, p, v in notes]
        pm.instruments.append(inst)
    pm.write(str(path))


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        sys.exit("usage: run_vocal.py <recording> <out.mid> [<recording> <out.mid> ...]")
    notes_model, voices_model = notes_network(), voices_network()
    check_loaded(notes_model, NOTES_WEIGHTS)
    check_loaded(voices_model, VOICES_WEIGHTS)
    for audio, out in zip(args[0::2], args[1::2]):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pitches = pitch_map(notes_model, audio)
        np.save(out.with_name(out.stem + "_pitch_map.npy"), pitches)
        everything = notes_from_map(pitches)
        write_midi(out, [("Voices", everything)])
        maps = voice_maps(voices_model, pitches)
        np.save(out.with_name(out.stem + "_voice_maps.npy"), np.stack(maps).astype(np.float16))
        lines = [(VOICES[i], notes_from_voice(maps[i], VOICE_THRESHOLDS[i])) for i in range(4)]
        write_midi(out.with_name(out.stem + "_satb.mid"), lines)
        print("wrote", out, "| frames", pitches.shape[1], "| notes", len(everything), "| per voice", {name: len(n) for name, n in lines}, flush=True)


if __name__ == "__main__":
    main()
