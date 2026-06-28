"""Maze puzzle evaluator for path-following tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeEvaluator(MazePuzzleEvaluator):
    """Evaluate maze solutions by using the shared pixel-based maze pipeline."""

    RED_DOMINANCE = 80

    def _build_generator(self, record: Dict[str, Any]) -> object:
        raise RuntimeError(
            "Text-to-image maze reconstruction is unavailable in this eval-only fork. "
            "Provide a candidate image directly."
        )


__all__ = ["MazeEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeEvaluator.main(argv)


if __name__ == "__main__":
    MazeEvaluator.main()
