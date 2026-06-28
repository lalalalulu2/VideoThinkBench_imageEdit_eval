"""Evaluator for circular labyrinth maze puzzles."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeLabyrinthEvaluator(MazePuzzleEvaluator):
    """Reuse the shared pixel-based evaluation with adjusted color sensitivity."""

    RED_DOMINANCE = 75

    def _build_generator(self, record: Dict[str, Any]) -> object:
        raise RuntimeError(
            "Text-to-image maze reconstruction is unavailable in this eval-only fork. "
            "Provide a candidate image directly."
        )


__all__ = ["MazeLabyrinthEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeLabyrinthEvaluator.main(argv)


if __name__ == "__main__":
    MazeLabyrinthEvaluator.main()
