from homie_core.intelligence.planner import HTNPlanner, Task, DecompositionRule


def test_add_rule():
    planner = HTNPlanner()
    rule = DecompositionRule(
        abstract_task="write_feature",
        subtasks=["write_test", "implement", "run_tests"],
        preconditions={"has_spec": True},
    )
    planner.add_rule(rule)
    assert len(planner.get_rules("write_feature")) == 1


def test_primitive_task():
    t = Task(name="open_editor", is_primitive=True)
    assert t.is_primitive


def test_decompose_single_level():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="write_feature",
        subtasks=["write_test", "implement", "run_tests"],
    ))
    planner.mark_primitive("write_test")
    planner.mark_primitive("implement")
    planner.mark_primitive("run_tests")
    plan = planner.plan("write_feature")
    assert plan is not None
    assert [t.name for t in plan] == ["write_test", "implement", "run_tests"]


def test_decompose_multi_level():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="ship_feature",
        subtasks=["write_code", "review", "deploy"],
    ))
    planner.add_rule(DecompositionRule(
        abstract_task="write_code",
        subtasks=["design", "implement", "test"],
    ))
    for p in ["design", "implement", "test", "review", "deploy"]:
        planner.mark_primitive(p)
    plan = planner.plan("ship_feature")
    assert plan is not None
    names = [t.name for t in plan]
    assert names == ["design", "implement", "test", "review", "deploy"]


def test_precondition_check():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="deploy",
        subtasks=["push", "monitor"],
        preconditions={"tests_passing": True},
    ))
    planner.mark_primitive("push")
    planner.mark_primitive("monitor")
    plan = planner.plan("deploy", state={"tests_passing": False})
    assert plan is None
    plan = planner.plan("deploy", state={"tests_passing": True})
    assert plan is not None


def test_multiple_rules_picks_first_valid():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(
        abstract_task="fix_bug",
        subtasks=["hotfix", "deploy"],
        preconditions={"is_critical": True},
    ))
    planner.add_rule(DecompositionRule(
        abstract_task="fix_bug",
        subtasks=["investigate", "fix", "test", "deploy"],
    ))
    for p in ["hotfix", "investigate", "fix", "test", "deploy"]:
        planner.mark_primitive(p)
    plan = planner.plan("fix_bug", state={"is_critical": False})
    assert plan is not None
    assert len(plan) == 4
    plan = planner.plan("fix_bug", state={"is_critical": True})
    assert plan is not None
    assert len(plan) == 2


def test_max_depth_prevents_infinite():
    planner = HTNPlanner(max_depth=5)
    planner.add_rule(DecompositionRule(abstract_task="loop_a", subtasks=["loop_b"]))
    planner.add_rule(DecompositionRule(abstract_task="loop_b", subtasks=["loop_a"]))
    plan = planner.plan("loop_a")
    assert plan is None


def test_estimate_cost():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(abstract_task="feature", subtasks=["code", "test"]))
    planner.mark_primitive("code", cost=5.0)
    planner.mark_primitive("test", cost=3.0)
    plan = planner.plan("feature")
    total = planner.estimate_cost(plan)
    assert abs(total - 8.0) < 1e-6


def test_serialize_deserialize():
    planner = HTNPlanner()
    planner.add_rule(DecompositionRule(abstract_task="task", subtasks=["a", "b"]))
    planner.mark_primitive("a")
    planner.mark_primitive("b")
    data = planner.serialize()
    planner2 = HTNPlanner.deserialize(data)
    plan = planner2.plan("task")
    assert plan is not None
    assert len(plan) == 2
