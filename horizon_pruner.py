from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TelemetryAlert:
    step_index: int
    step_text: str
    raw_score: float
    adjusted_score: float
    reason: str


class EventHorizonPruner:
    SYCOPHANTIC_PATTERNS = [
        r"100%\s*rigorous",
        r"by\s+trivial\s+inspection",
        r"obviously",
        r"let\s+us\s+carefully\s+verify\s+this",
        r"undeniably\s+true",
        r"as\s+anyone\s+can\s+clearly\s+see",
        r"flawless\s+deduction",
    ]

    def __init__(
        self,
        tau_horizon: float = 0.50,
        hacking_penalty_factor: float = 0.25,
    ) -> None:
        self.tau_horizon = tau_horizon
        self.hacking_penalty_factor = hacking_penalty_factor
        self.alerts: List[TelemetryAlert] = []

    @staticmethod
    def _normalize_math_tokens(text: str) -> str:
        # Strip filler words and whitespace to isolate core algebraic/symbolic content
        cleaned = re.sub(r"[^a-zA-Z0-9=+\-*/^()]", "", text.lower())
        return cleaned

    def check_reward_hacking(
        self,
        step: str,
        prior_steps: List[str],
    ) -> Tuple[bool, str]:
        """
        Detects:
        1. Sycophantic Filler Injection
        2. Step-Splitting Inflation (duplicate/trivial restatements of prior equations)
        """
        lower_step = step.lower()
        for pattern in self.SYCOPHANTIC_PATTERNS:
            if re.search(pattern, lower_step):
                return True, f"Sycophantic Filler Detected ({pattern})"

        # Step-Splitting Inflation check
        norm_current = self._normalize_math_tokens(step)
        if len(norm_current) < 6:
            return True, "Step-Splitting Inflation (Vacuous micro-step)"

        for prev in prior_steps:
            norm_prev = self._normalize_math_tokens(prev)
            if norm_current == norm_prev:
                return True, "Step-Splitting Inflation (Duplicate equation restatement)"

        return False, ""

    def prune_trajectory(
        self,
        steps: List[str],
        step_scores: List[float],
    ) -> Dict[str, object]:
        """
        Iterates step-by-step through (steps, step_scores).
        Applies multiplicative penalty when reward hacking triggers, and severs
        the trajectory if adjusted score < self.tau_horizon.
        """
        self.alerts.clear()
        surviving_steps: List[str] = []
        adjusted_scores: List[float] = []
        severed = False
        sever_index = -1

        for idx, (step, raw_score) in enumerate(zip(steps, step_scores)):
            is_hacked, hack_reason = self.check_reward_hacking(step, surviving_steps)
            adj_score = raw_score * self.hacking_penalty_factor if is_hacked else raw_score

            if adj_score < self.tau_horizon:
                severed = True
                sever_index = idx
                alert_reason = (
                    f"{hack_reason} -> Adjusted score {adj_score:.3f} < tau_horizon ({self.tau_horizon})"
                    if is_hacked
                    else f"Hallucination Drift -> Raw score {raw_score:.3f} < tau_horizon ({self.tau_horizon})"
                )
                self.alerts.append(
                    TelemetryAlert(
                        step_index=idx,
                        step_text=step,
                        raw_score=raw_score,
                        adjusted_score=adj_score,
                        reason=alert_reason,
                    )
                )
                break

            surviving_steps.append(step)
            adjusted_scores.append(adj_score)

        return {
            "surviving_steps": surviving_steps,
            "adjusted_scores": adjusted_scores,
            "severed": severed,
            "sever_index": sever_index,
            "alerts": self.alerts,
        }


if __name__ == "__main__":
    pruner = EventHorizonPruner(tau_horizon=0.50, hacking_penalty_factor=0.25)
    test_steps = [
        "r_s = 2 * G * M / c^2",
        "r_s = 2 * G * M / c^2",  # Step-splitting duplicate
        "Obviously, by trivial inspection, r_s = 1.227e10 m",  # Sycophantic filler
    ]
    test_scores = [0.95, 0.90, 0.92]

    result = pruner.prune_trajectory(test_steps, test_scores)
    print("Severed:", result["severed"], "at index:", result["sever_index"])
    for alert in result["alerts"]:
        print("TELEMETRY ALERT:", alert)