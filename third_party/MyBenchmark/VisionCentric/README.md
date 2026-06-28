
# VideoThinkBench Vision-Centric Toolkit

This eval-only fork hosts inference and evaluation utilities for the vision-centric portion of the “Thinking with Video” study. Benchmark data synthesis code and concrete puzzle generators are intentionally omitted.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Evaluators run on Python 3.10+. Optional dependencies such as Whisper, ffmpeg, and GPU-accelerated OpenCV improve throughput for large batches but are not required for small experiments.

## Repository layout

- `puzzle/`: evaluator implementations for the task families.
	- Eyeballing puzzles live in directories named after the geometric target (`circle_center/`, `angle_bisector/`, …).
	- ARC-AGI-2 abstractions are implemented in `arcagi/`.
	- Maze variants (`maze/`, `maze_hexagon/`, `maze_labyrinth/`) share common helpers in `maze_base.py`.
- `data/`: ignored local data/output root.
- `scripts/`: inference, fixed-dataset evaluation, transcription, and result summaries.

> **Note**: The repository still carries earlier puzzle prototypes (jigsaw, Sudoku, mirror, rectangles, etc.). They are preserved for completeness but were not part of the published experiments.

## General scripts

`scripts/veo3.py` and `scripts/gpt5.py` call corresponding API to get model generations. `veo3.py` is for video generation models, and `gpt5.py` is for VLMs.
`scripts/mirrorVote.py` generates multiple responses for one puzzle.
`scripts/fixed_dataset.py` evaluates puzzles on fixed dataset instead of generating new puzzles. Our dataset and mini testset can be found ![here](https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench). Ensure the files are arranged as example below:

- `dataset/`: arbitrary folder name.
	- `maze_square/`: puzzle type name.
		- `puzzles/`: input images of puzzles.
		- `solutions/`: solution images of puzzles.
		- `data.json`: data of puzzles.
	- `.../`: other puzzle types, with same `puzzles/` `solutions/` and `data.json` inside.

Then run `scripts/fixed_dataset.py`, for example `python scripts/fixed_dataset.py --dataset-root dataset --workers 16 --resume` to evaluate on a fixed dataset.

## Eyeballing puzzles

Eyeballing puzzles require the model to mark the correct geometric element from five options while optionally verbalizing the choice. We evaluate three groups:

- **Point Tasks**: `circle_center`, `circumcenter`, `fermat_point`, `incenter`, `midpoint`, `orthocenter`, `point_reflection`, `ray_intersection`, `triangle_center`.
- **Line Tasks**: `angle_bisector`, `arc_connect`, `circle_tangent_line`, `circle_tangent_point`, `parallel`, `perpendicular`, `perpendicular_bisector`, `ray_reflect`.
- **Shape Tasks**: `isosceles_trapezoid`, `parallelogram`, `right_triangle`, `square_outlier`.

Each task inherits from the shared point-target scaffolding in `point_target_base.py`, so the CLI and output layout are consistent.

### Evaluate predictions

```bash
python -m puzzle.circle_center.evaluator data/circle_center/data.json <PUZZLE_ID> attempts/0001/final.png --video-stride 3
```

The evaluator reports the option inferred from:

- the red highlight in the candidate image,
- parsed captions or transcripts located next to the attempt,
- sampled frames from the accompanying video.

Most leaderboard scores quoted in the paper use majority voting over frames (“Major Frame”), last-frame inspection, or the audio transcript.

`scripts/multiple_choice_summary.py` outputs summary for all eyeballing puzzles.

## ARC-AGI-2 abstractions

Our ARC implementation turns few-shot grid reasoning into a video-friendly format: training exemplars appear on the left, the target input is rendered on the right, and the answer grid remains blank for the model to fill.

```bash
python -m puzzle.arcagi.evaluator data/arcagi/data.json <PUZZLE_ID> attempts/arcagi/final.png
```

Key helpers:

- `scripts/arcagi_range_vote.py`: aggregates self-consistency runs (supports GPT-5, Claude 4.5, Gemini 2.5 Pro, and Sora-2 outputs). The paper’s ablations rely on these ranges.

Evaluation converts colored cells back to ARC palette indices and prints JSON with per-cell agreement, enabling downstream voting or qualitative review.

## Maze families

Maze benchmarks test dynamic path drawing.

```bash
python -m puzzle.maze_square.evaluator data/maze/data.json <PUZZLE_ID> attempts/maze/final.png
```

Mazes highlight the start cell and the goal in red. The evaluator verifies that a continuous red stroke connects them without bleeding into walls. `scripts/maze_summary.py` collects aggregate accuracy from batches of attempts.

## Legacy evaluators not in the paper

Directories such as `puzzle/jigsaw/`, `puzzle/sudoku/`, `puzzle/mirror/`, and `puzzle/rects/` remain in the tree for archival reasons.

