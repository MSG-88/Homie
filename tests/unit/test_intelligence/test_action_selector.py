import math
from homie_core.intelligence.action_selector import MCTSActionSelector, ActionNode


def test_create_root():
    selector = MCTSActionSelector()
    root = selector.create_root(available_actions=["a", "b", "c"])
    assert len(root.children) == 0
    assert root.untried_actions == ["a", "b", "c"]


def test_ucb1_formula():
    node = ActionNode(action="test")
    node.visits = 10
    node.total_reward = 7.0
    parent_visits = 100
    ucb = node.ucb1(parent_visits, exploration_weight=1.414)
    expected_exploit = 7.0 / 10
    expected_explore = 1.414 * math.sqrt(math.log(100) / 10)
    assert abs(ucb - (expected_exploit + expected_explore)) < 1e-4


def test_select_expands_untried():
    selector = MCTSActionSelector(exploration_weight=1.414)
    root = selector.create_root(["a", "b", "c"])
    selected = selector.select(root)
    assert selected.action in ["a", "b", "c"]
    assert len(root.children) == 1


def test_simulate_returns_reward():
    selector = MCTSActionSelector()
    def reward_fn(action, context):
        return 1.0 if action == "good" else 0.0
    reward = selector.simulate("good", {}, reward_fn)
    assert reward == 1.0


def test_backpropagate():
    parent = ActionNode(action="root")
    child = ActionNode(action="a", parent=parent)
    parent.children.append(child)
    MCTSActionSelector.backpropagate(child, reward=0.8)
    assert child.visits == 1
    assert child.total_reward == 0.8
    assert parent.visits == 1
    assert parent.total_reward == 0.8


def test_best_action_after_search():
    selector = MCTSActionSelector(exploration_weight=1.0, n_iterations=100)
    def reward_fn(action, context):
        rewards = {"good": 0.9, "ok": 0.5, "bad": 0.1}
        return rewards.get(action, 0.0)
    best = selector.search(
        available_actions=["good", "ok", "bad"],
        context={},
        reward_fn=reward_fn,
    )
    assert best == "good"


def test_get_action_scores():
    selector = MCTSActionSelector(n_iterations=50)
    def reward_fn(action, context):
        return 0.8 if action == "x" else 0.2
    selector.search(["x", "y"], {}, reward_fn)
    scores = selector.get_action_scores()
    assert "x" in scores
    assert scores["x"] > scores["y"]


def test_empty_actions():
    selector = MCTSActionSelector()
    best = selector.search([], {}, lambda a, c: 0.0)
    assert best is None
