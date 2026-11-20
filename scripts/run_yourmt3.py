"""Run the YourMT3+ multi-instrument transcription model on a recording.

YourMT3+ (Chang et al., 2024; Apache-2.0) writes a MIDI file with one track per instrument it
believes it hears, so unlike the note detector used elsewhere here it says which instrument
plays each note. This wrapper expects the model's inference code and one checkpoint under
tools/yourmt3/ (the "YourMT3" Space and model repository of the authors on Hugging Face) and
its own environment (.venv_mt3).

usage: .venv_mt3/bin/python scripts/run_yourmt3.py <recording> <out.mid>
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "yourmt3"
CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
ARGS = [CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
        "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
        "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1"]


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2:
        sys.exit("usage: run_yourmt3.py <recording> <out.mid> [<recording> <out.mid> ...]")
    jobs = [(Path(a).resolve(), Path(b).resolve()) for a, b in zip(args[0::2], args[1::2])]
    os.chdir(TOOL)  # the model code finds its checkpoint relative to this folder
    sys.path.insert(0, str(TOOL / "amt" / "src"))
    sys.path.insert(0, str(TOOL))
    import torchaudio
    from model_helper import load_model_checkpoint
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes
    from utils.utils import write_model_output_as_midi

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_checkpoint(args=ARGS + ["-pr", "16" if device == "cuda" else "32"], device="cpu").to(device)

    for audio_path, out_path in jobs:
        y, sr = sf.read(str(audio_path), always_2d=True)
        audio = torch.from_numpy(y.mean(axis=1).astype("float32")).unsqueeze(0)
        audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
        frames = model.audio_cfg["input_frames"]
        segments = slice_padded_array(audio.numpy(), frames, frames)
        segments = torch.from_numpy(segments.astype("float32")).to(device).unsqueeze(1)
        with torch.no_grad():
            tokens, _ = model.inference_file(bsz=4, audio_segments=segments)

        starts = [frames * i / model.audio_cfg["sample_rate"] for i in range(segments.shape[0])]
        notes_by_channel, errors = [], Counter()
        for ch in range(model.task_manager.num_decoding_channels):
            zipped, _, _ = model.task_manager.detokenize_list_batches([arr[:, ch, :] for arr in tokens], starts, return_events=True)
            notes, err = merge_zipped_note_events_and_ties_to_notes(zipped)
            notes_by_channel.append(notes)
            errors += err
        out_dir = out_path.parent / "_yourmt3_tmp"
        write_model_output_as_midi(mix_notes(notes_by_channel), str(out_dir) + "/", out_path.stem, model.midi_output_inverse_vocab)
        produced = out_dir / "model_output" / (out_path.stem + ".mid")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(produced, out_path)
        print("wrote", out_path, "| segments", segments.shape[0], "| decoding errors", dict(errors), flush=True)


if __name__ == "__main__":
    main()
