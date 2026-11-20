from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class MCTSNode:
    state: str
    prior_prob: float = 1.0
    parent: Optional[MCTSNode] = None
    children: List[MCTSNode] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTSEngine:
    def __init__(self, c_puct: float = 1.414) -> None:
        self.c_puct = c_puct

    def puct_score(self, child: MCTSNode) -> float:
        """
        Exact PUCT formula:
        PUCT(s, a) = Q(s, a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s, a))
        """
        parent_visits = max(1, child.parent.visit_count) if child.parent else 1
        q_sa = child.q_value
        u_sa = (
            self.c_puct
            * child.prior_prob
            * (math.sqrt(parent_visits) / (1 + child.visit_count))
        )
        return q_sa + u_sa

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=self.puct_score)

    def select_leaf(self, root: MCTSNode) -> MCTSNode:
        """Phase 1: Traverse from root using PUCT until reaching an unexpanded or terminal leaf."""
        current = root
        while not current.is_leaf and not current.is_terminal:
            current = self.select_best_child(current)
        return current

    def expand(
        self,
        node: MCTSNode,
        candidate_priors: List[Tuple[str, float, bool]],
    ) -> List[MCTSNode]:
        """Phase 2: Instantiate child MCTSNode objects for each (thought_state, prior_prob, is_terminal)."""
        if node.is_terminal:
            return []
        for thought_state, prior_prob, is_term in candidate_priors:
            child = MCTSNode(
                state=thought_state,
                prior_prob=prior_prob,
                parent=node,
                is_terminal=is_term,
            )
            node.children.append(child)
        return node.children

    def simulate_rollout(
        self,
        node: MCTSNode,
        rollout_evaluator: Callable[[MCTSNode], float],
    ) -> float:
        """Phase 3: Evaluate newly expanded node via trajectory rollout or PRM value estimate in [0, 1]."""
        reward = rollout_evaluator(node)
        return max(0.0, min(1.0, reward))

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Phase 4: Traverse upward from node to root updating visit_count and value_sum."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent

    def run_search(
        self,
        root: MCTSNode,
        policy_fn: Callable[[MCTSNode], List[Tuple[str, float, bool]]],
        rollout_evaluator: Callable[[MCTSNode], float],
        num_simulations: int = 25,
    ) -> MCTSNode:
        for _ in range(num_simulations):
            leaf = self.select_leaf(root)
            if not leaf.is_terminal:
                priors = policy_fn(leaf)
                children = self.expand(leaf, priors)
                target = children[0] if children else leaf
            else:
                target = leaf
            reward = self.simulate_rollout(target, rollout_evaluator)
            self.backpropagate(target, reward)

        return max(root.children, key=lambda c: c.visit_count) if root.children else root


if __name__ == "__main__":
    engine = MCTSEngine(c_puct=1.414)
    root_node = MCTSNode("Start Orbital Stabilization")

    def mock_policy(n: MCTSNode) -> List[Tuple[str, float, bool]]:
        if n.state == "Start Orbital Stabilization":
            return [
                ("Apply Kerr metric frame-dragging correction", 0.75, False),
                ("Apply Newtonian approximation", 0.25, False),
            ]
        return [(f"{n.state} -> Trajectory Verified", 0.90, True)]

    def mock_rollout(n: MCTSNode) -> float:
        return 0.95 if "Kerr" in n.state else 0.15

    best_action = engine.run_search(root_node, mock_policy, mock_rollout, num_simulations=20)
    print(f"Selected Action: {best_action.state} | Visits: {best_action.visit_count} | Q: {best_action.q_value:.4f}")