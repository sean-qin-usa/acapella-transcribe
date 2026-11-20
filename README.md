# acapella-transcribe

Turn a recording into a draft score. The tools here split a song into stems, write down the notes in each one, and export MIDI and MusicXML that open in MuseScore, so that arranging for a vocal group or a string duo starts from a draft instead of a blank page.

There are two paths, because two kinds of recording fail in different ways. A pop song can be pulled apart by timbre (voice, bass, drums, everything else) and each stem transcribed on its own. A duet of two violins cannot, since the separation models rely on the instruments sounding different; there the notes are transcribed from the mix and divided into two parts afterwards.

## How well it works

The duet path is scored against nine two-violin recordings from the [URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html), which come with hand-checked annotations for every note of every player: 4,318 notes in total.

| | |
|---|---|
| Real notes found | **89.9%** |
| Written notes that are real | **88.1%** |
| Found notes placed in the right player's part | **84.2%** |
| Fast notes (under 0.15 s) found | 81.6% |
| Upper note found when the violins play in octaves | 73.4% |
| Notes 10 dB or more quieter than the other violin found | 84.5% |

The detector tends to write one held note as two or three, and on these pieces a back-to-back pair of the same pitch is a real repeated note about as often as it is a continuation, so merging them blindly costs 17 points of notes found. The recording settles it instead: at a real attack the level of that pitch jumps by a median of 6.8 dB and dips by 6.3 dB first, against 1.9 and -0.1 for a continuation, and a note is joined to the one before it only when neither shows. That check took the share of written notes that are real from 74.7% to 88.1% for 2.4 points of notes found; its two thresholds were chosen on half of the pieces, so part of that gain is in-sample (the held-out halves gave 13 and 15 points). What remains among the extra notes is mostly overtones an octave or a twelfth above a real note.

## Where it falls short

Telling the two players apart is the weak point. In a mono recording nothing in the pitches says which violin is playing when one of them plays alone or when they trade melody and accompaniment, and four attempts at smarter splitting rules left the 85% figure where it was. A `--solo` hint for passages you know are one player, and a `--roles` mode that divides the music into a sustained line and a moving line, are the workarounds for now.

Chords inside a single part are kept when the notes are held together, but the benchmark players never play double stops, so that behaviour is unmeasured. Recordings with a piano are out of scope for the duet path, because the piano's notes end up in the violin parts.

The pop path has no ground truth behind it yet. Sung melodies and bass lines come out as single lines with a confidence score per note; spoken or rapped passages are flagged as unclear and drawn with x noteheads, since their pitches are guesses. Off-scale notes are moved a semitone onto the song's scale unless `--no-autotune` is given. Chord roots are stable from bar to bar, while major versus minor is not reliable.

## Running it

Python 3.10 and a GPU with about 8 GB of memory are enough. The separation models download themselves on first use and take a few gigabytes.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

For a pop song, starting from an audio file you have the right to use:

```bash
scripts/run_pop.sh path/to/song.wav my_song
```

This writes `out/my_song/` with the stems, a tempo-aligned MIDI file, a MusicXML score (lead, backing vocals, bass), a chord chart and a report. For two violins:

```bash
python -I scripts/duet.py path/to/duet.wav out/my_duet
python -I scripts/duet.py path/to/duet.wav out/my_duet --solo 0-13     # one player alone in the first 13 s
python -I scripts/duet.py path/to/duet.wav out/my_duet --roles         # sustained line / moving line
```

## Checking by ear

Every run writes `check_*.wav` files with the recording in the left ear and the transcription, played as plain tones, in the right. For duets `check_parts.wav` puts one part in each ear, which is the quickest way to hear where the split goes wrong.

## Checking against a score

`scripts/eval_urmp.py` reproduces the table above (the `merge_held` variant is the default pipeline) from a local copy of the URMP pieces that contain two violins, and reports the same figures broken down by note length, loudness and interval, which is how each setting was chosen. `scripts/eval_score.py` compares a transcription with sheet music in MusicXML or MIDI by stretching the score onto the recording's timing, and with `--partial` it accepts a score that covers only part of the recording. On four benchmark pieces its alignment puts 90% of notes within 150 ms of the annotated onset (median 49 ms), and its found-rate came out 2 to 7 points below the annotation-based figure on three of them and about 20 below on the fourth, so read its numbers as a lower bound.

## Turning sheet music into a dataset

When a score exists, it is a better reference than any annotation made by ear, so there is a path from page images to a table of notes. `scripts/build_score_dataset.sh` takes a PDF or a folder of page images, reads the pages several at a time with the [oemer](https://github.com/BreezeWhite/oemer) score reader, joins the pages into one score with a named part per staff, and, given a recording, places every note on that recording's clock.

```bash
scripts/build_score_dataset.sh score.pdf my_piece --layout "Violin 1,Violin 2,Piano,Piano" --audio recording.wav
```

Each piece ends up in `datasets/my_piece/` as the page images, the reader's output per page, a `score.mid`, a `notes_score.csv` (part, measure, beat, duration, pitch) and a `notes_aligned.csv` (part, start, end, pitch), the last in the same shape as the URMP annotations. The reader runs in its own environment (`.venv_omr`) because it needs older versions of NumPy and OpenCV than the rest. The reader decides which staves belong together from the barlines that cross them, so a system that is one long measure, with no barline inside it, comes out scrambled; `scripts/helper_barlines.py` finds such systems and draws one extra barline at the emptiest column near the middle, and the two halves are merged back when the pages are joined. On the fifteen-page test score that turned five unreadable pages into fifteen joined ones. A page whose staves still cannot be placed is skipped and listed in `score_meta.json`, never guessed at. Rhythms are the weak part of what it reads, with roughly six bars in ten adding up to a full measure on a clean engraving, and nothing is hand-checked, so treat the result as a draft reference until someone has looked it over.

## A second opinion on who is playing

The duet tool also asks a question that does not depend on pitch movement: which violin does this note sound like? It learns each instrument's tone from the passages where both play at once, with no labels, and compares that opinion with the split. On the benchmark the split is right for 90% of notes where the two opinions agree and 71% where they disagree, so disagreements are coloured red in the MusicXML as the places to check by ear first. Using tone to overrule the split, rather than to flag it, did not help: 88.6% against 88.8% without it (`scripts/tone_experiment.py`).

## What is in here

- `scripts/amt.py` holds the shared pieces: pitch tracking and note cutting for single lines, the multi-note detector, beat grid, chords, drums, MIDI and MusicXML export.
- `scripts/pop.py` and `scripts/run_pop.sh` are the pop path.
- `scripts/duet.py` is the two-instrument path, including the part splitter.
- `scripts/eval_urmp.py` and `scripts/eval_score.py` are the two checkers.
- `scripts/build_score_dataset.sh`, `scripts/helper_barlines.py`, `scripts/omr_stitch.py` and `scripts/align_dataset.py` are the sheet-music path.
- `scripts/tone_experiment.py` is the tone experiment behind the second opinion, `scripts/rearticulation_experiment.py` the one behind the held-note check, and `scripts/instrument_experiment.py` an attempt to tell struck notes from bowed ones that did not work well enough to use (69% against 59% for guessing).
- `scripts/plot_line.py` and `scripts/probe_stereo.py` are small diagnostics.

No audio, stems, scores or transcriptions of commercial recordings are included.
