# acapella-transcribe

Turn a recording into a draft score. The tools here split a song into stems, write down the notes in each one, and export MIDI and MusicXML that open in MuseScore, so that arranging for a vocal group or a string duo starts from a draft instead of a blank page.

There are two paths, because two kinds of recording fail in different ways. A pop song can be pulled apart by timbre (voice, bass, drums, everything else) and each stem transcribed on its own. A duet of two violins cannot, since the separation models rely on the instruments sounding different; there the notes are transcribed from the mix and divided into two parts afterwards.

## How well it works

The duet path is scored against nine two-violin recordings from the [URMP dataset](https://labsites.rochester.edu/air/projects/URMP.html), which come with hand-checked annotations for every note of every player: 4,318 notes in total.

| | |
|---|---|
| Real notes found | **92.3%** |
| Written notes that are real | **74.7%** |
| Found notes placed in the right player's part | **85.3%** |
| Fast notes (under 0.15 s) found | 83.7% |
| Upper note found when the violins play in octaves | 75.1% |
| Notes 10 dB or more quieter than the other violin found | 84.5% |

The settings lean toward finding notes at the cost of writing extra ones, on the view that deleting a wrong note in a notation program is quicker than hearing and adding a missing one. Most of the extra notes are a held note written twice, and a smaller share are overtones an octave or a twelfth above a real note.

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

`scripts/eval_urmp.py` reproduces the table above (the `leap_filter` variant is the default pipeline) from a local copy of the URMP pieces that contain two violins, and reports the same figures broken down by note length, loudness and interval, which is how each setting was chosen. `scripts/eval_score.py` compares a transcription with sheet music in MusicXML or MIDI by stretching the score onto the recording's timing; its alignment lands within 150 ms of the annotated onset for only 70 to 88% of notes on three benchmark pieces, so treat its numbers as a lower bound until that improves.

## What is in here

- `scripts/amt.py` holds the shared pieces: pitch tracking and note cutting for single lines, the multi-note detector, beat grid, chords, drums, MIDI and MusicXML export.
- `scripts/pop.py` and `scripts/run_pop.sh` are the pop path.
- `scripts/duet.py` is the two-instrument path, including the part splitter.
- `scripts/eval_urmp.py` and `scripts/eval_score.py` are the two checkers.
- `scripts/plot_line.py` and `scripts/probe_stereo.py` are small diagnostics.

No audio, stems or transcriptions of commercial recordings are included.
