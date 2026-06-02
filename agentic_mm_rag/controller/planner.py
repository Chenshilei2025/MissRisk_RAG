from __future__ import annotations

from agentic_mm_rag.observation.obligations import SearchObligation


class ObligationPlanner:
    """Minimal obligation planner.

    The first implementation keeps a single broad obligation. A dataset- or
    model-backed planner can later replace this without changing the runtime.
    """

    def plan(self, question: str) -> list[SearchObligation]:
        return [
            SearchObligation(
                id="o1",
                text=f"Check answer-bearing evidence for: {question}",
                required_modalities=[],
            )
        ]
