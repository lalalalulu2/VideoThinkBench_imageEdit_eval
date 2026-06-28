"""Evaluator for hexagonal maze puzzles."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeHexagonEvaluator(MazePuzzleEvaluator):
    """Reuse the shared maze evaluation while tweaking color sensitivity for thin walls."""

    RED_DOMINANCE = 75

    def _build_generator(self, record: Dict[str, Any]) -> object:
        raise RuntimeError(
            "Text-to-image maze reconstruction is unavailable in this eval-only fork. "
            "Provide a candidate image directly."
        )


__all__ = ["MazeHexagonEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeHexagonEvaluator.main(argv)


if __name__ == "__main__":
    MazeHexagonEvaluator.main()
