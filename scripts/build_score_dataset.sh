#!/usr/bin/env bash
# Sheet music (a PDF or a folder of page images) -> a dataset folder of notes.
#
# usage: scripts/build_score_dataset.sh <score.pdf | folder of page images> <name>
#            [--layout "Violin 1,Violin 2,Piano,Piano"] [--beats 4] [--audio recording.wav]
#
# Writes datasets/<name>/:
#   pages/page-NN.png      the page images that were read
#   omr/page-NN.musicxml   what the reader made of each page (kept so a bad page can be redone alone)
#   pages_helped/, omr_helped/   the same for pages that needed a helper barline
#   score.mid              all pages joined, one track per part, in score time
#   notes_score.csv        one row per note: part, measure, beat, duration, pitch
#   notes_aligned.csv      the same notes placed on the recording's clock (only with --audio)
#   score_meta.json        layout, page and measure counts, reading checks, alignment check
#
# Pages that are already read are skipped, so the command can be rerun after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
src=$1; name=$2; shift 2
layout="Violin 1,Violin 2,Piano,Piano"; beats=4; audio=""
while [ $# -gt 0 ]; do
  case $1 in
    --layout) layout=$2; shift 2;;
    --beats) beats=$2; shift 2;;
    --audio) audio=$2; shift 2;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
D=datasets/$name
mkdir -p "$D/pages" "$D/omr"

# 1. page images, numbered in order
if [ -d "$src" ]; then
  i=0
  for f in $(ls "$src" | grep -iE '\.(png|jpg|jpeg)$' | sort -V); do
    i=$((i + 1)); dst=$(printf "%s/pages/page-%02d.png" "$D" "$i")
    [ -e "$dst" ] || cp "$src/$f" "$dst"
  done
else
  [ -e "$D/pages/page-01.png" ] || pdftoppm -r 200 -png "$src" "$D/pages/p"
  i=0
  for f in $(ls "$D/pages" | grep -E '^p-' | sort -V); do
    i=$((i + 1)); mv "$D/pages/$f" "$(printf "%s/pages/page-%02d.png" "$D" "$i")"
  done
fi

# 2. give single-measure systems a helper barline (the reader cannot group their staves
#    otherwise), then read each page several at a time. Pages already read are skipped.
. .venv/bin/activate
mkdir -p "$D/pages_helped" "$D/omr_helped"
for img in "$D"/pages/page-*.png; do
  python -I scripts/helper_barlines.py "$img" "$D/pages_helped/$(basename "$img")" > /dev/null
done
export D
ls "$D"/pages/page-*.png | xargs -P "${JOBS:-3}" -I{} sh -c '
  n=$(basename "{}" .png)
  [ -e "$D/omr/$n.musicxml" ] || .venv_omr/bin/oemer "{}" -o "$D/omr" --without-deskew --save-cache > "$D/omr/$n.log" 2>&1 || echo "page $n could not be read as is (see $D/omr/$n.log)"
  if grep -q "\"helper\": true" "$D/pages_helped/$n.png.json" && [ ! -e "$D/omr_helped/$n.musicxml" ]; then
    .venv_omr/bin/oemer "$D/pages_helped/$n.png" -o "$D/omr_helped" --without-deskew --save-cache > "$D/omr_helped/$n.log" 2>&1 || echo "page $n could not be read with helper barlines (see $D/omr_helped/$n.log)"
  fi
'

# 3. join pages into one score, 4. optionally place it on a recording
python -I scripts/omr_stitch.py "$D/omr" "$D" --layout "$layout" --beats "$beats" --helped "$D/omr_helped" --page-info "$D/pages_helped"
if [ -n "$audio" ]; then
  python -I scripts/align_dataset.py "$D" "$audio"
fi
