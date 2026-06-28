from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list")
    return payload


def resolve_path(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else base / value


def manifest_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def nearest_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def scaled_size(width: int, height: int, multiple: int, scale: float) -> tuple[int, int]:
    return nearest_multiple(max(1, int(width * scale)), multiple), nearest_multiple(max(1, int(height * scale)), multiple)


def closest_model_size(width: int, height: int, multiple: int, max_pixels: int) -> tuple[int, int, str]:
    base_w, base_h = nearest_multiple(width, multiple), nearest_multiple(height, multiple)
    if max_pixels <= 0 or base_w * base_h <= max_pixels:
        return base_w, base_h, "nearest_multiple"
    scale = math.sqrt(max_pixels / float(base_w * base_h))
    w, h = scaled_size(width, height, multiple, scale)
    while w * h > max_pixels and (w > multiple or h > multiple):
        scale *= 0.98
        w, h = scaled_size(width, height, multiple, scale)
    return w, h, f"pixel_cap_{max_pixels}"


def iter_tasks(dataset_root: Path, categories: list[str]) -> Iterable[tuple[str, Path]]:
    if "eyeballing" in categories:
        base = dataset_root / "eyeballing_puzzles" / "eyeballing_puzzles"
        for d in sorted(base.iterdir()):
            if (d / "data.json").exists():
                yield f"eyeballing/{d.name}", d
    if "mazes" in categories:
        base = dataset_root / "mazes" / "mazes"
        for d in sorted(base.iterdir()):
            if (d / "data.json").exists():
                yield f"mazes/{d.name}", d
    if "visual" in categories:
        base = dataset_root / "visual_puzzles" / "visual_puzzles"
        for d in sorted(base.iterdir()):
            if (d / "data.json").exists():
                yield f"visual/{d.name}", d
    if "arcagi" in categories:
        d = dataset_root / "arcagi" / "arcagi"
        if (d / "data.json").exists():
            yield "arcagi/arcagi", d


def load_completed(manifest: Path) -> set[tuple[str, str]]:
    if not manifest.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completed.add((str(row["task"]), str(row["id"])))
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA-only Qwen-Image-Edit inference for VideoThinkBench.")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/VideoThinkBench/minitest_Vision-Centric_Reasoning"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/Qwen-Image-Edit-2511"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/vtb_lora_only"))
    parser.add_argument("--lora-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="lora_minitest")
    parser.add_argument("--categories", nargs="+", default=["eyeballing", "mazes", "visual", "arcagi"])
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--true-cfg-scale", type=float, default=2.8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2511)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=4_800_000, help="Use the closest feasible model resolution above this area.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = resolve_path(project_root, args.dataset_root)
    model_dir = resolve_path(project_root, args.model_dir)
    output_root = resolve_path(project_root, args.output_root)
    lora_dir = resolve_path(project_root, args.lora_dir)

    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.load_lora_weights(str(lora_dir))
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    multiple = pipe.vae_scale_factor * 2
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = run_dir / "predictions.jsonl"
    completed = load_completed(predictions) if args.resume else set()
    mode = "a" if args.resume else "w"

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    generated = 0
    skipped = 0
    with predictions.open(mode, encoding="utf-8") as manifest:
        for task_name, task_dir in iter_tasks(dataset_root, args.categories):
            entries = read_json(task_dir / "data.json")
            if args.limit_per_task > 0:
                entries = entries[: args.limit_per_task]
            for entry in entries:
                row_key = (task_name, str(entry["id"]))
                if row_key in completed:
                    skipped += 1
                    continue

                image_path = task_dir / str(entry["image"])
                image = Image.open(image_path).convert("RGB")
                width, height, resize_reason = closest_model_size(image.width, image.height, multiple, args.max_pixels)
                infer_image = image if (width, height) == image.size else image.resize((width, height), Image.Resampling.LANCZOS)
                prompt = str(entry.get("prompt") or entry.get("question") or "")

                oom_retries = 0
                while True:
                    try:
                        out = pipe(
                            image=[infer_image],
                            prompt=prompt,
                            negative_prompt="",
                            true_cfg_scale=args.true_cfg_scale,
                            guidance_scale=args.guidance_scale,
                            num_inference_steps=args.steps,
                            generator=generator,
                            height=height,
                            width=width,
                        ).images[0]
                        break
                    except torch.OutOfMemoryError:
                        if args.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        oom_retries += 1
                        if oom_retries > 4:
                            raise
                        width, height = scaled_size(image.width, image.height, multiple, 0.88**oom_retries)
                        infer_image = image.resize((width, height), Image.Resampling.LANCZOS)
                        resize_reason = f"oom_fallback_{oom_retries}"

                out_dir = run_dir / task_name / str(entry["id"])
                out_dir.mkdir(parents=True, exist_ok=True)
                final_path = out_dir / "final.png"
                out.save(final_path)
                (out_dir / "content.txt").write_text("Image-only LoRA prediction.\n", encoding="utf-8")

                row = {
                    "task": task_name,
                    "id": entry["id"],
                    "image": manifest_path(image_path, project_root),
                    "prediction": manifest_path(final_path, project_root),
                    "metadata": manifest_path(task_dir / "data.json", project_root),
                    "prompt_used": prompt,
                    "input_width": image.width,
                    "input_height": image.height,
                    "output_width": out.width,
                    "output_height": out.height,
                    "model_width": width,
                    "model_height": height,
                    "resize_reason": resize_reason,
                    "lora_dir": manifest_path(lora_dir, project_root),
                }
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest.flush()
                generated += 1
                print(
                    json.dumps(
                        {
                            "generated": generated,
                            "skipped": skipped,
                            "task": task_name,
                            "id": entry["id"],
                            "input_size": [image.width, image.height],
                            "output_size": [out.width, out.height],
                            "resize_reason": resize_reason,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    print(json.dumps({"run_dir": str(run_dir), "generated": generated, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
