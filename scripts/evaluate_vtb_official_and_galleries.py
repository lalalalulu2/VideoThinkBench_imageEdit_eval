from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
COLOR_VISUAL_TASKS = {
    "color_size",
    "color_grid",
    "color_hexagon",
    "color_overlap_squares",
    "polygon_sides_color",
    "rectangle_height_color",
}
SHAPE_VISUAL_TASKS = {"size_grid", "shape_reflect", "shape_size_grid", "size_cycle"}
CATEGORY_LABELS = {
    "arcagi": "ARC-AGI-2",
    "eyeballing": "Eyeballing_Puzzles",
    "mazes": "Mazes",
    "visual": "Visual_Puzzles",
}


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def install_official_evaluators(mybenchmark_root: Path) -> dict[str, Any]:
    vision_root = mybenchmark_root / "VisionCentric"
    if str(vision_root) not in sys.path:
        sys.path.insert(0, str(vision_root))

    from puzzle.arcagi.evaluator import ArcPuzzleEvaluator
    from puzzle.maze_hexagon.evaluator import MazeHexagonEvaluator
    from puzzle.maze_labyrinth.evaluator import MazeLabyrinthEvaluator
    from puzzle.maze_square.evaluator import MazeEvaluator as MazeSquareEvaluator
    from puzzle.point_target_base import PointTargetPuzzleEvaluator

    return {
        "arc": ArcPuzzleEvaluator,
        "point": PointTargetPuzzleEvaluator,
        "mazes": {
            "maze_square": MazeSquareEvaluator,
            "maze_hexagon": MazeHexagonEvaluator,
            "maze_labyrinth": MazeLabyrinthEvaluator,
        },
    }


def get_window(image: np.ndarray, width: int = 512, height: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    width = min(width, w)
    height = min(height, h)
    x0 = max(0, (w - width) // 2)
    y0 = max(0, (h - height) // 2)
    return image[y0 : y0 + height, x0 : x0 + width]


def visual_difference(candidate_path: Path, solution_path: Path, metric: str) -> dict[str, Any]:
    cand = cv2.imread(str(candidate_path))
    sol = cv2.imread(str(solution_path))
    if cand is None:
        raise FileNotFoundError(f"Could not read candidate image: {candidate_path}")
    if sol is None:
        raise FileNotFoundError(f"Could not read solution image: {solution_path}")
    cand_win = get_window(cand)
    sol_win = get_window(sol)
    if cand_win.shape != sol_win.shape:
        cand_win = cv2.resize(cand_win, (sol_win.shape[1], sol_win.shape[0]))
    if metric == "coverage":
        gray1 = cv2.cvtColor(cand_win, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(sol_win, cv2.COLOR_BGR2GRAY)
        _, bin1 = cv2.threshold(gray1, 245, 255, cv2.THRESH_BINARY)
        _, bin2 = cv2.threshold(gray2, 245, 255, cv2.THRESH_BINARY)
        diff = float(np.sum(bin1 != bin2))
        denom = float(bin1.size)
    else:
        arr1 = cand_win.astype(np.float32)
        arr2 = sol_win.astype(np.float32)
        per_pixel = np.sqrt(np.sum((arr1 - arr2) ** 2, axis=2))
        diff = float(np.sum(per_pixel))
        denom = float(per_pixel.size * math.sqrt(3 * 255 * 255))
    similarity = 1.0 - min(1.0, diff / denom) if denom else 0.0
    return {
        "metric": metric,
        "difference": diff,
        "normalized_similarity": similarity,
        "is_correct": diff == 0.0,
        "message": f"{metric} diff={diff:.1f}, similarity={similarity:.4f}",
    }


def evaluate_row(row: dict[str, Any], project_root: Path, evaluators: dict[str, Any]) -> dict[str, Any]:
    task = row["task"]
    task_name = task.split("/", 1)[1]
    metadata = resolve_path(project_root, row["metadata"])
    prediction = resolve_path(project_root, row["prediction"])
    base_dir = metadata.parent

    try:
        if task.startswith("eyeballing/"):
            evaluator = evaluators["point"](metadata, base_dir=base_dir)
            result = evaluator.evaluate(str(row["id"]), prediction, video_sample_stride=5)
            payload = result.to_dict()
            payload["red_pixel_count"] = getattr(result, "red_pixel_count", None)
            payload["red_centroid"] = getattr(result, "red_centroid", None)
            payload["is_correct"] = payload.get("image_option") == payload.get("correct_option")
            payload["official_mode"] = "image_option"
            payload["message"] = f"image={payload.get('image_option')} correct={payload.get('correct_option')} red={payload.get('red_pixel_count')}"
        elif task.startswith("mazes/"):
            evaluator_cls = evaluators["mazes"][task_name]
            result = evaluator_cls(metadata, base_dir=base_dir).evaluate(str(row["id"]), prediction)
            payload = result.to_dict()
            payload["is_correct"] = bool(payload.get("connected")) and not bool(payload.get("overlaps_walls"))
            payload["official_mode"] = "maze_path"
        elif task.startswith("arcagi/"):
            result = evaluators["arc"](metadata, base_dir=base_dir).evaluate(str(row["id"]), prediction)
            payload = result.to_dict()
            payload["is_correct"] = payload.get("accuracy") == 1.0
            payload["official_mode"] = "arc_cell_accuracy"
            payload["message"] = f"cells {payload.get('correct_cells')}/{payload.get('total_cells')} acc={payload.get('accuracy'):.4f}"
        elif task.startswith("visual/"):
            records = {str(item["id"]): item for item in load_json(metadata)}
            record = records[str(row["id"])]
            solution = base_dir / str(record["solution_image_path"])
            metric = "euclidean" if task_name in COLOR_VISUAL_TASKS else "coverage"
            payload = visual_difference(prediction, solution, metric)
            payload["puzzle_id"] = row["id"]
            payload["is_correct"] = bool(payload["is_correct"])
            payload["official_mode"] = "visual_best_frame_static_metric"
        else:
            raise ValueError(f"Unsupported task: {task}")
    except Exception as exc:
        payload = {
            "puzzle_id": row["id"],
            "is_correct": False,
            "official_mode": "official_evaluator_exception",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "message": f"official evaluator exception: {type(exc).__name__}: {exc}",
        }

    payload["task"] = task
    payload["id"] = row["id"]
    payload["prediction"] = row["prediction"]
    payload["input_size"] = [row.get("input_width"), row.get("input_height")]
    payload["output_size"] = [row.get("output_width"), row.get("output_height")]
    payload["resize_reason"] = row.get("resize_reason")
    save_json(prediction.parent / "evaluation.json", payload)
    return payload


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[result["task"]].append(result)
    tasks: dict[str, Any] = {}
    total = len(results)
    total_correct = sum(1 for r in results if r.get("is_correct"))
    for task, rows in sorted(by_task.items()):
        correct = sum(1 for r in rows if r.get("is_correct"))
        item: dict[str, Any] = {"count": len(rows), "correct": correct, "accuracy": correct / len(rows) if rows else 0.0}
        if task.startswith("arcagi/"):
            cells = sum(int(r.get("total_cells", 0)) for r in rows)
            cell_correct = sum(int(r.get("correct_cells", 0)) for r in rows)
            item["cell_accuracy"] = cell_correct / cells if cells else 0.0
            item["correct_cells"] = cell_correct
            item["total_cells"] = cells
        if task.startswith("visual/"):
            item["mean_similarity"] = sum(float(r.get("normalized_similarity", 0.0)) for r in rows) / len(rows)
            item["mean_difference"] = sum(float(r.get("difference", 0.0)) for r in rows) / len(rows)
        tasks[task] = item
    return {"total": total, "correct": total_correct, "accuracy": total_correct / total if total else 0.0, "tasks": tasks}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_width: int, max_lines: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def result_lines(result: dict[str, Any]) -> list[str]:
    task = result["task"]
    lines: list[str] = []
    if task.startswith("mazes/"):
        lines.append(str(result.get("message", "")))
        output_size = result.get("output_size") or []
        out = "x".join(str(x) for x in output_size if x is not None)
        lines.append(f"red={result.get('red_pixel_count')} out={out}")
    elif task.startswith("eyeballing/"):
        lines.append(str(result.get("message", "")))
        lines.append(f"centroid={result.get('red_centroid')}")
    elif task.startswith("arcagi/"):
        lines.append(str(result.get("message", "")))
        lines.append(f"task_id={result.get('task_id', '')}")
    else:
        lines.append(str(result.get("message", "")))
        lines.append(f"out={result.get('output_size')} mode={result.get('official_mode')}")
    return lines


def gallery_subtitle(task: str, rows: list[dict[str, Any]], dataset_label: str) -> str:
    group, task_name = task.split("/", 1)
    category = CATEGORY_LABELS.get(group, group)
    prefix = f"VideoThinkBench {dataset_label} {category}/{task_name}"
    if task.startswith("mazes/"):
        wall = sum(1 for r in rows if bool(r.get("overlaps_walls")))
        missed_start = sum(1 for r in rows if not bool(r.get("touches_start")))
        missed_goal = sum(1 for r in rows if not bool(r.get("touches_goal")))
        disconnected = sum(1 for r in rows if not bool(r.get("connected")))
        return f"{prefix} | official strict evaluator | wall:{wall} missed_start:{missed_start} missed_goal:{missed_goal} disconnected:{disconnected}"
    modes = sorted({str(r.get("official_mode")) for r in rows})
    return f"{prefix} | official evaluator | mode:{', '.join(modes)}"


def make_gallery(task: str, rows: list[dict[str, Any]], out_path: Path, title_prefix: str, project_root: Path, dataset_label: str) -> None:
    cols = 5
    card_w = 230
    image_h = 300
    text_h = 76
    gap = 14
    margin = 18
    header_h = 86
    card_h = image_h + text_h + 16
    rows_count = math.ceil(len(rows) / cols)
    width = margin * 2 + cols * card_w + (cols - 1) * gap
    height = header_h + margin + rows_count * card_h + max(0, rows_count - 1) * gap + margin
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    title_font = font(24, bold=True)
    sub_font = font(12)
    label_font = font(14, bold=True)
    small_font = font(10)
    ok = sum(1 for r in rows if r.get("is_correct"))
    title = f"{title_prefix} {task.replace('/', '_')}: {ok}/{len(rows)} PASS"
    draw.text((margin, 12), title, fill=(31, 41, 55), font=title_font)
    draw.text((margin, 44), gallery_subtitle(task, rows, dataset_label), fill=(75, 85, 99), font=sub_font)
    draw.rectangle((margin, 66, margin + 14, 80), fill=(16, 185, 129))
    draw.text((margin + 20, 64), "PASS", fill=(31, 41, 55), font=sub_font)
    draw.rectangle((margin + 84, 66, margin + 98, 80), fill=(239, 68, 68))
    draw.text((margin + 104, 64), "FAIL", fill=(31, 41, 55), font=sub_font)

    for idx, row in enumerate(rows):
        col = idx % cols
        rr = idx // cols
        x = margin + col * (card_w + gap)
        y = header_h + rr * (card_h + gap)
        passed = bool(row.get("is_correct"))
        color = (5, 150, 105) if passed else (220, 38, 38)
        draw.rectangle((x, y, x + card_w, y + card_h), outline=color, width=5)
        image_path = resolve_path(project_root, row["prediction"])
        try:
            im = Image.open(image_path).convert("RGB")
        except Exception:
            im = Image.new("RGB", (card_w - 16, image_h), (245, 245, 245))
        im.thumbnail((card_w - 16, image_h), Image.Resampling.LANCZOS)
        im_x = x + (card_w - im.width) // 2
        im_y = y + 8 + (image_h - im.height) // 2
        page.paste(im, (im_x, im_y))
        text_y = y + image_h + 12
        status = "PASS" if passed else "FAIL"
        short_id = str(row.get("id", ""))[:8]
        draw.text((x + 10, text_y), f"{idx:02d} {status}", fill=color, font=label_font)
        draw.text((x + 88, text_y + 2), short_id, fill=(107, 114, 128), font=small_font)
        line_y = text_y + 21
        for raw in result_lines(row):
            for line in text_wrap(draw, raw, small_font, card_w - 20, 2):
                draw.text((x + 10, line_y), line, fill=(75, 85, 99), font=small_font)
                line_y += 12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official VideoThinkBench evaluators and make per-task galleries.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--mybenchmark-root", type=Path, default=REPO_ROOT / "third_party" / "MyBenchmark")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--title-prefix", default="Qwen-Image-Edit LoRA-only official eval")
    parser.add_argument("--dataset-label", default="minitest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest = resolve_path(project_root, args.manifest)
    output_dir = resolve_path(project_root, args.output_dir) if args.output_dir else manifest.parent / "official_eval"
    evaluators = install_official_evaluators(resolve_path(project_root, args.mybenchmark_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, row in enumerate(read_manifest(manifest), 1):
        result = evaluate_row(row, project_root, evaluators)
        results.append(result)
        print(json.dumps({"evaluated": idx, "task": result["task"], "id": result["id"], "pass": result.get("is_correct")}, ensure_ascii=False), flush=True)

    summary = summarize(results)
    save_json(output_dir / "official_eval_summary.json", {"summary": summary, "results": results})

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[result["task"]].append(result)
    gallery_dir = output_dir / "galleries"
    for task, task_rows in sorted(by_task.items()):
        task_rows.sort(key=lambda r: str(r.get("id")))
        safe = task.replace("/", "__")
        make_gallery(task, task_rows, gallery_dir / f"{safe}.png", args.title_prefix, project_root, args.dataset_label)

    print(json.dumps({"summary_path": str(output_dir / "official_eval_summary.json"), "gallery_dir": str(gallery_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
