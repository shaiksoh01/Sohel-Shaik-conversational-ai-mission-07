from __future__ import annotations
import math
from typing import Callable, Dict, List, Tuple


class OutcomeRewardEvaluator:
    """Evaluates only the final answer, vulnerable to false-positive reasoning traps."""

    def evaluate(self, final_answer: str, ground_truth: str) -> float:
        return 1.0 if final_answer.strip() == ground_truth.strip() else 0.0


class ProcessRewardEvaluator:
    """Evaluates intermediate steps, aggregates trajectories, and runs Math-Shepherd rollouts."""

    @staticmethod
    def reduce_min(step_scores: List[float]) -> float:
        if not step_scores:
            return 0.0
        return min(step_scores)

    @staticmethod
    def reduce_prod(step_scores: List[float]) -> float:
        if not step_scores:
            return 0.0
        return math.prod(step_scores)

    @staticmethod
    def reduce_geometric_mean(step_scores: List[float]) -> float:
        if not step_scores:
            return 0.0
        return math.prod(step_scores) ** (1.0 / len(step_scores))

    def aggregate_trajectory(self, step_scores: List[float]) -> Dict[str, float]:
        return {
            "min": self.reduce_min(step_scores),
            "prod": self.reduce_prod(step_scores),
            "geometric_mean": self.reduce_geometric_mean(step_scores),
        }

    @staticmethod
    def math_shepherd_estimation(
        rollout_answers: List[str],
        ground_truth: str,
    ) -> Tuple[float, int]:
        """
        Computes Math-Shepherd Soft and Hard labels from M rollout completions:
        y_soft = (1/M) * sum(I(ans_j == y*))
        y_hard = I(y_soft > 0)
        """
        if not rollout_answers:
            return 0.0, 0
        matches = sum(1 for ans in rollout_answers if ans.strip() == ground_truth.strip())
        y_soft = matches / len(rollout_answers)
        y_hard = 1 if y_soft > 0.0 else 0
        return y_soft, y_hard


def run_orm_vs_prm_benchmark() -> Dict[str, object]:
    """
    Demonstrates ORM false positive where two sign errors cancel out,
    while PRM catches the invalid intermediate step.
    """
    ground_truth = "42"
    flawed_steps = [
        "Step 1: Compute (-6) * (-7) = -42",   # Invalid sign error #1 (score: 0.05)
        "Step 2: Multiply by -1 -> -(-42) = 42" # Invalid sign error #2 canceling out (score: 0.10)
    ]
    step_scores = [0.05, 0.10]
    final_answer = "42"

    orm = OutcomeRewardEvaluator()
    prm = ProcessRewardEvaluator()

    orm_score = orm.evaluate(final_answer, ground_truth)
    prm_metrics = prm.aggregate_trajectory(step_scores)

    return {
        "steps": flawed_steps,
        "orm_score": orm_score,
        "prm_metrics": prm_metrics,
        "false_positive_caught": orm_score == 1.0 and prm_metrics["min"] < 0.5,
    }


if __name__ == "__main__":
    benchmark = run_orm_vs_prm_benchmark()
    print("ORM Score (False Positive):", benchmark["orm_score"])
    print("PRM Aggregations:", benchmark["prm_metrics"])
    print("False Positive Caught:", benchmark["false_positive_caught"])

    # Math-Shepherd Demo
    rollouts = ["42", "42", "41", "42", "0", "42", "42", "42"]
    soft, hard = ProcessRewardEvaluator.math_shepherd_estimation(rollouts, "42")
    print(f"Math-Shepherd Labels -> Soft: {soft:.4f}, Hard: {hard}")