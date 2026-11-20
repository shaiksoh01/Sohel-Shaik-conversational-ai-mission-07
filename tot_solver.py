from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


@dataclass
class ThoughtNode:
    state: str
    depth: int = 0
    step_score: float = 1.0
    cumulative_score: float = 1.0
    parent: Optional[ThoughtNode] = None
    children: List[ThoughtNode] = field(default_factory=list)

    def get_trajectory(self) -> List[str]:
        """Reconstructs the reasoning chain from root to this node."""
        trajectory: List[str] = []
        current: Optional[ThoughtNode] = self
        while current is not None:
            trajectory.append(current.state)
            current = current.parent
        return list(reversed(trajectory))


class TreeOfThoughtsSolver:
    def __init__(
        self,
        beam_width: int = 3,
        max_expansions: int = 20,
        prune_threshold: float = 0.4,
    ) -> None:
        self.beam_width = beam_width
        self.max_expansions = max_expansions
        self.prune_threshold = prune_threshold
        self.expansion_count = 0

    def _expand_node(
        self,
        node: ThoughtNode,
        generator_fn: Callable[[ThoughtNode], List[str]],
        evaluator_fn: Callable[[str, List[str]], float],
    ) -> List[ThoughtNode]:
        if self.expansion_count >= self.max_expansions:
            return []

        self.expansion_count += 1
        candidate_states = generator_fn(node)
        valid_children: List[ThoughtNode] = []
        prefix_traj = node.get_trajectory()

        for state in candidate_states:
            score = evaluator_fn(state, prefix_traj)
            if score >= self.prune_threshold:
                child = ThoughtNode(
                    state=state,
                    depth=node.depth + 1,
                    step_score=score,
                    cumulative_score=node.cumulative_score * score,
                    parent=node,
                )
                node.children.append(child)
                valid_children.append(child)

        return valid_children

    def bfs_beam_search(
        self,
        root: ThoughtNode,
        generator_fn: Callable[[ThoughtNode], List[str]],
        evaluator_fn: Callable[[str, List[str]], float],
        max_depth: int = 4,
    ) -> ThoughtNode:
        """Breadth-First Beam Search under strict expansion budget."""
        self.expansion_count = 0
        frontier: List[ThoughtNode] = [root]
        best_node: ThoughtNode = root

        for _ in range(1, max_depth + 1):
            if not frontier or self.expansion_count >= self.max_expansions:
                break

            next_candidates: List[ThoughtNode] = []
            for node in frontier:
                if self.expansion_count >= self.max_expansions:
                    break
                children = self._expand_node(node, generator_fn, evaluator_fn)
                next_candidates.extend(children)

            if not next_candidates:
                break

            # Keep top-B nodes by cumulative_score
            next_candidates.sort(key=lambda n: n.cumulative_score, reverse=True)
            frontier = next_candidates[: self.beam_width]

            # Track deepest / highest-scoring discovered node
            if frontier[0].depth > best_node.depth or (
                frontier[0].depth == best_node.depth
                and frontier[0].cumulative_score > best_node.cumulative_score
            ):
                best_node = frontier[0]

        return best_node

    def dfs_backtrack_search(
        self,
        root: ThoughtNode,
        generator_fn: Callable[[ThoughtNode], List[str]],
        evaluator_fn: Callable[[str, List[str]], float],
        max_depth: int = 4,
    ) -> ThoughtNode:
        """Depth-First Backtracking Search exploring highest-scoring branches first."""
        self.expansion_count = 0
        best_node: ThoughtNode = root

        def _dfs(current: ThoughtNode) -> None:
            nonlocal best_node
            if current.depth > best_node.depth or (
                current.depth == best_node.depth
                and current.cumulative_score > best_node.cumulative_score
            ):
                best_node = current

            if current.depth >= max_depth or self.expansion_count >= self.max_expansions:
                return

            children = self._expand_node(current, generator_fn, evaluator_fn)
            # Explore highest-scoring child branch first; backtrack chronologically on dead-ends
            children.sort(key=lambda c: c.step_score, reverse=True)
            for child in children:
                if self.expansion_count >= self.max_expansions:
                    break
                _dfs(child)

        _dfs(root)
        return best_node


if __name__ == "__main__":
    # Verification Demo
    def mock_generator(node: ThoughtNode) -> List[str]:
        if node.depth == 0:
            return ["Set up Schwarzschild metric r_s = 2GM/c^2", "Guess linear gravity"]
        if "Schwarzschild" in node.state:
            return ["Substitute M = 4.154e6 M_sun", "Ignore relativistic factor"]
        return ["Compute r_s ≈ 1.227e10 m (Final Answer)"]

    def mock_evaluator(state: str, trajectory: List[str]) -> float:
        if "Guess" in state or "Ignore" in state:
            return 0.20  # Below prune_threshold
        if "Final Answer" in state:
            return 0.96
        return 0.92

    solver = TreeOfThoughtsSolver(beam_width=2, max_expansions=10, prune_threshold=0.5)
    bfs_res = solver.bfs_beam_search(ThoughtNode("Problem: Compute Sgr A* r_s"), mock_generator, mock_evaluator, max_depth=3)
    print("BFS Best Trajectory:", bfs_res.get_trajectory(), "| Score:", round(bfs_res.cumulative_score, 4))

    dfs_res = solver.dfs_backtrack_search(ThoughtNode("Problem: Compute Sgr A* r_s"), mock_generator, mock_evaluator, max_depth=3)
    print("DFS Best Trajectory:", dfs_res.get_trajectory(), "| Score:", round(dfs_res.cumulative_score, 4))