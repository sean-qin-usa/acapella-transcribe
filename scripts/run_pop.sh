#!/usr/bin/env bash
# Separate a song into stems and transcribe it.  usage: scripts/run_pop.sh <audio file> <name>
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export PATH="$(python -c "import static_ffmpeg, shutil, os; static_ffmpeg.add_paths(); print(os.path.dirname(shutil.which(\"ffmpeg\")))"):$PATH"
audio=$1; name=$2; S=out/$name/stems
mkdir -p "$S"
sep() { audio-separator "$1" -m "$2" --model_file_dir models --output_dir "$3" --output_format FLAC --log_level WARNING; }
pick() { mv "$(ls "$1"/*"($2)"*.flac | head -1)" "$3"; }
sep "$audio" model_bs_roformer_ep_317_sdr_12.9755.ckpt "$S/a"          # vocals / instrumental
pick "$S/a" Vocals "$S/vocals.flac"; pick "$S/a" Instrumental "$S/instrumental.flac"
sep "$S/vocals.flac" mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt "$S/b"   # lead / backing
pick "$S/b" Vocals "$S/lead.flac"; pick "$S/b" Instrumental "$S/backing.flac"
sep "$S/instrumental.flac" htdemucs_ft.yaml "$S/c"                     # bass / drums / other
pick "$S/c" Bass "$S/bass.flac"; pick "$S/c" Drums "$S/drums.flac"; pick "$S/c" Other "$S/other.flac"; pick "$S/c" Vocals "$S/sample_vox.flac"
rm -rf "$S/a" "$S/b" "$S/c"
python -I scripts/pop.py "$S" "out/$name" --title "$name"
