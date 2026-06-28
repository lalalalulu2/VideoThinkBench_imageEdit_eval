# VideoThinkBench Image-Edit Eval

Reusable LoRA-only inference and evaluation utilities for Qwen image-edit models on the VideoThinkBench Vision-Centric split.

This repository vendors the official evaluator code from `lalalalulu2/MyBenchmark` under `third_party/MyBenchmark` so the evaluation wrapper can run without a separate clone. The vendored copy used here is from commit `9cd84d2084b33da316366958330884edbf3ec328`.

## What Is Included

- `scripts/infer_vtb_lora_only.py`: single-pass LoRA-only Qwen-Image-Edit inference for VideoThinkBench image-edit prompts.
- `scripts/evaluate_vtb_official_and_galleries.py`: official evaluator wrapper plus per-task summary gallery generation.
- `third_party/MyBenchmark`: official VideoThinkBench evaluator code copied from MyBenchmark.

The repository intentionally does not include VideoThinkBench data, model weights, LoRA weights, generated predictions, or evaluation outputs.

## Expected Directory Layout

You can keep data and weights inside this repo or pass absolute paths:

```text
VideoThinkBench_imageEdit_eval/
  data/VideoThinkBench/minitest_Vision-Centric_Reasoning/
  models/Qwen-Image-Edit-2511/
  outputs/qwenedit_lora/final/
```

## LoRA-Only Inference

```bash
python3 scripts/infer_vtb_lora_only.py \
  --dataset-root data/VideoThinkBench/minitest_Vision-Centric_Reasoning \
  --model-dir models/Qwen-Image-Edit-2511 \
  --lora-dir outputs/qwenedit_lora/final \
  --output-root outputs/vtb_lora_only \
  --run-name best_lora_minitest \
  --steps 20 \
  --true-cfg-scale 2.8 \
  --max-pixels 4800000
```

The script uses the original dataset prompt as-is. Output resolution is set to the nearest model-compatible resolution; very large images are scaled to the nearest feasible size under `--max-pixels`.

## Official Evaluation And Summary Galleries

```bash
python3 scripts/evaluate_vtb_official_and_galleries.py \
  --manifest outputs/vtb_lora_only/best_lora_minitest/predictions.jsonl \
  --output-dir outputs/vtb_lora_only/best_lora_minitest/official_eval
```

The output directory contains:

- `official_eval_summary.json`
- per-sample `evaluation.json` files beside each prediction
- one PNG summary gallery per task in `galleries/`

## Notes

- Eyeballing, maze, and ARC tasks use the vendored official MyBenchmark evaluators.
- Visual puzzle tasks use the same static image metric family as MyBenchmark's `visual_puzzles/eval/find_best_frame.py`.
- If an official evaluator raises on a malformed model output, the wrapper records that sample as `FAIL` and stores the exception in its `evaluation.json`.
