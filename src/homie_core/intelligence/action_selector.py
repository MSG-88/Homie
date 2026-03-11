"""Monte Carlo Tree Search (UCT) action selector.

Implements the UCT (Upper Confidence bounds applied to Trees) algorithm
for selecting the best action from a set of candidates.

Reference: Kocsis & Szepesvari (2006) "Bandit based Monte-Carlo Planning"
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Optional


class ActionNode:
    """A node in the MCTS search tree representing an action."""

    def __init__(
        self,
        action: Optional[str] = None,
        parent: Optional[ActionNode] = None,
        untried_actions: Optional[list[str]] = None,
    ) -> None:
        self.action = action
        self.parent = parent
        self.children: list[ActionNode] = []
        self.untried_actions: list[str] = untried_actions or []
        self.visits: int = 0
        self.total_reward: float = 0.0

    @property
    def avg_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1(self, parent_visits: int, exploration_weight: float) -> float:
        """UCB1 formula: Q/N + C * sqrt(ln(parent_N) / N)."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_reward / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_weight: float) -> ActionNode:
        """Return the child with the highest UCB1 value."""
        return max(
            self.children,
            key=lambda c: c.ucb1(self.visits, exploration_weight),
        )


class MCTSActionSelector:
    """Monte Carlo Tree Search action selector using the UCT algorithm."""

    def __init__(
        self,
        exploration_weight: float = 1.414,
        n_iterations: int = 200,
        seed: int = 42,
    ) -> None:
        self.exploration_weight = exploration_weight
        self.n_iterations = n_iterations
        self._rng = random.Random(seed)
        self._root: Optional[ActionNode] = None

    def create_root(self, available_actions: list[str]) -> ActionNode:
        """Create a root node with the given available actions."""
        self._root = ActionNode(
            action="root", untried_actions=list(available_actions)
        )
        return self._root

    def select(self, node: ActionNode) -> ActionNode:
        """Select a node to expand using UCB1, expanding the first untried action found."""
        current = node
        while True:
            if current.untried_actions:
                # Expand: pick a random untried action
                action = self._rng.choice(current.untried_actions)
                current.untried_actions.remove(action)
                child = ActionNode(action=action, parent=current)
                current.children.append(child)
                return child
            elif current.children:
                current = current.best_child(self.exploration_weight)
            else:
                return current

    def simulate(
        self,
        action: str,
        context: dict[str, Any],
        reward_fn: Callable[[str, dict[str, Any]], float],
    ) -> float:
        """Run a simulation by calling the reward function."""
        return reward_fn(action, context)

    @staticmethod
    def backpropagate(node: ActionNode, reward: float) -> None:
        """Walk up from node to root, updating visits and total_reward."""
        current: Optional[ActionNode] = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def search(
        self,
        available_actions: list[str],
        context: dict[str, Any],
        reward_fn: Callable[[str, dict[str, Any]], float],
    ) -> Optional[str]:
        """Run MCTS and return the most-visited child's action."""
        if not available_actions:
            return None

        root = self.create_root(available_actions)

        for _ in range(self.n_iterations):
            selected = self.select(root)
            reward = self.simulate(selected.action, context, reward_fn)
            self.backpropagate(selected, reward)

        if not root.children:
            return None

        # Return the action of the most-visited child
        best = max(root.children, key=lambda c: c.visits)
        return best.action

    def get_action_scores(self) -> dict[str, float]:
        """Return a dict of action -> average reward from root's children."""
        if self._root is None:
            return {}
        return {
            child.action: child.avg_reward
            for child in self._root.children
        }
