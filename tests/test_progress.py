"""进展检测离线测试：空修改、状态回退、编辑间隔与分级干预。"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from nextcoder.runtime.progress import (
    LoopLimits,
    build_timeline,
    inspect_loop,
)


def _call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _result(call_id: str, text: str) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call_id)


def _edit(path: str, post: str, call_id: str) -> list:
    return [
        _call("apply_patch", {
            "path": path,
            "expected_hash": "pre",
            "old_text": f"old-{post[-8:]}",
            "new_text": f"new-{post[-8:]}",
        }, call_id),
        _result(
            call_id,
            f"[apply_patch] status=completed changed={path} content_hash={post}\n"
            f"已修改 {path}",
        ),
    ]


def _noop(path: str, call_id: str) -> list:
    return [
        _call("apply_patch", {"path": path, "expected_hash": "pre"}, call_id),
        _result(
            call_id,
            "[apply_patch] status=failed\n"
            "error: old_text 与 new_text 相同，文件不会被修改。",
        ),
    ]


def _test_command(call_id: str) -> list:
    return [
        _call("run_command", {"command": "python -m pytest tests/test_x.py -q"}, call_id),
        _result(call_id, "[run_command] status=completed exit_code=0\n1 passed"),
    ]


def _evidence(messages: list) -> tuple[int, int]:
    timeline = build_timeline(messages)
    hashes = {
        (event.path, event.post_hash)
        for event in timeline.edits
        if event.post_hash
    }
    return len(hashes), len(timeline.tests)


def _state(**overrides) -> dict:
    values = {
        "loop_replan_key": "",
        "loop_replan_at": 0,
        "loop_replan_path": "",
        "loop_replan_evidence": (0, 0),
        "loop_block_until": 0,
        "loop_gap_nudged": False,
        "loop_gap_test_count": 0,
    }
    values.update(overrides)
    return values


def test_single_revert_is_not_a_loop():
    messages = [HumanMessage(content="go")]
    messages += _edit("a.py", "a" * 64, "c1")
    messages += _edit("a.py", "b" * 64, "c2")
    messages += _edit("a.py", "a" * 64, "c3")
    decision = inspect_loop(messages, model_calls=4, state=_state())
    assert decision.kind is None


def test_third_return_triggers_replan_and_blocks_one_edit():
    messages = [HumanMessage(content="go")]
    for index, digest in enumerate(["a", "b", "a", "b", "a"], start=1):
        messages += _edit("a.py", digest * 64, f"c{index}")
    decision = inspect_loop(messages, model_calls=6, state=_state())
    assert decision.kind == "replan"
    assert "a.py" in decision.message
    assert decision.replan_path == "a.py"
    assert decision.block_until == 7
    assert decision.replan_at == 6


def test_two_noop_edits_trigger_replan():
    messages = [HumanMessage(content="go")]
    messages += _noop("a.py", "c1")
    messages += _noop("a.py", "c2")
    decision = inspect_loop(messages, model_calls=3, state=_state())
    assert decision.kind == "replan"
    assert "空修改" in decision.message


def test_single_noop_is_not_a_loop():
    messages = [HumanMessage(content="go")]
    messages += _noop("a.py", "c1")
    decision = inspect_loop(messages, model_calls=2, state=_state())
    assert decision.kind is None


def test_eight_edits_without_test_only_nudges():
    messages = [HumanMessage(content="go")]
    for index in range(8):
        messages += _edit("a.py", f"{index:064x}", f"c{index}")
    decision = inspect_loop(messages, model_calls=9, state=_state())
    assert decision.kind == "nudge"
    assert decision.gap_nudged is True


def test_small_edit_gap_does_not_arm_nudge():
    messages = [HumanMessage(content="go")]
    messages += _edit("a.py", "a" * 64, "c1")
    decision = inspect_loop(messages, model_calls=2, state=_state())
    assert decision.kind is None
    assert decision.gap_nudged is False


def test_gap_nudge_rearms_after_test():
    messages = [HumanMessage(content="go")]
    for index in range(8):
        messages += _edit("a.py", f"{index:064x}", f"c{index}")
    messages += _test_command("test")
    messages += _edit("a.py", "f" * 64, "c8")
    decision = inspect_loop(
        messages, model_calls=11, state=_state(loop_gap_nudged=True))
    assert decision.kind is None
    assert decision.gap_nudged is False


def test_nudge_is_not_repeated_once_armed():
    messages = [HumanMessage(content="go")]
    for index in range(8):
        messages += _edit("a.py", f"{index:064x}", f"c{index}")
    decision = inspect_loop(
        messages, model_calls=9, state=_state(loop_gap_nudged=True))
    assert decision.kind is None


def test_stop_after_replan_without_new_evidence():
    messages = [HumanMessage(content="go")]
    for index, digest in enumerate(["a", "b", "a", "b", "a", "b", "a"], start=1):
        messages += _edit("a.py", digest * 64, f"c{index}")
    evidence = _evidence(messages)
    state = _state(
        loop_replan_key="rollback:a.py:" + "A" * 12,
        loop_replan_at=3,
        loop_replan_path="a.py",
        loop_replan_evidence=evidence,
        loop_block_until=4,
    )
    decision = inspect_loop(messages, model_calls=11, state=state)
    assert decision.kind == "stop"
    assert decision.end_run is True
    assert "a.py" in decision.message


def test_replan_clears_when_new_evidence_appears():
    messages = [HumanMessage(content="go")]
    for index, digest in enumerate(["a", "b", "a"], start=1):
        messages += _edit("a.py", digest * 64, f"c{index}")
    state = _state(
        loop_replan_key="rollback:a.py:" + "A" * 12,
        loop_replan_at=3,
        loop_replan_path="a.py",
        loop_replan_evidence=(0, 0),
    )
    decision = inspect_loop(messages, model_calls=8, state=state)
    assert decision.replan_key == ""
    assert decision.replan_at == 0
    assert decision.block_until == 0


def test_three_identical_calls_only_remind():
    messages = [HumanMessage(content="go")]
    for index in range(3):
        call_id = f"c{index}"
        messages.append(_call("read_lines", {"path": "a.py", "start": 1, "end": 5}, call_id))
        messages.append(_result(call_id, "[read_lines] status=completed\n1: x"))
    decision = inspect_loop(messages, model_calls=4, state=_state())
    assert decision.kind is None


def test_five_identical_calls_trigger_replan_and_block_the_next_one():
    messages = [HumanMessage(content="go")]
    for index in range(6):
        call_id = f"c{index}"
        messages.append(_call("read_lines", {"path": "a.py", "start": 1, "end": 5}, call_id))
        messages.append(_result(call_id, "[read_lines] status=completed\n1: x"))
    decision = inspect_loop(messages, model_calls=7, state=_state())
    assert decision.kind == "replan"
    assert decision.replan_signature
    assert decision.block_until == 8


def test_exact_repeat_stops_only_after_replan():
    messages = [HumanMessage(content="go")]
    for index in range(6):
        call_id = f"c{index}"
        messages.append(_call("read_lines", {"path": "a.py", "start": 1, "end": 5}, call_id))
        messages.append(_result(call_id, "[read_lines] status=completed\n1: x"))
    no_replan = inspect_loop(messages, model_calls=7, state=_state())
    assert no_replan.kind != "stop"

    state = _state(
        loop_replan_key="noop:a.py",
        loop_replan_at=3,
        loop_replan_path="a.py",
        loop_replan_evidence=_evidence(messages),
        loop_block_until=4,
    )
    decision = inspect_loop(messages, model_calls=12, state=state, limits=LoopLimits())
    assert decision.kind == "stop"


def test_noop_stop_waits_for_new_test_evidence():
    messages = [HumanMessage(content="go")]
    messages += _noop("a.py", "c1")
    messages += _noop("a.py", "c2")
    first = inspect_loop(messages, model_calls=3, state=_state())
    assert first.kind == "replan"

    messages += _noop("a.py", "c3")
    messages += _noop("a.py", "c4")
    messages += _noop("a.py", "c5")
    messages += _noop("a.py", "c6")
    messages += _test_command("c7")
    state = _state(
        loop_replan_key=first.replan_key,
        loop_replan_at=3,
        loop_replan_path="a.py",
        loop_replan_evidence=first.replan_evidence,
        loop_block_until=4,
    )
    decision = inspect_loop(messages, model_calls=8, state=state)
    assert decision.kind != "stop"
    assert decision.replan_key == ""


def test_noop_stop_after_eight_calls_without_evidence():
    messages = [HumanMessage(content="go")]
    messages += _noop("a.py", "c1")
    messages += _noop("a.py", "c2")
    first = inspect_loop(messages, model_calls=3, state=_state())
    messages += _noop("a.py", "c3")
    messages += _noop("a.py", "c4")
    messages += _noop("a.py", "c5")
    messages += _noop("a.py", "c6")
    state = _state(
        loop_replan_key=first.replan_key,
        loop_replan_at=3,
        loop_replan_path="a.py",
        loop_replan_evidence=first.replan_evidence,
        loop_block_until=4,
    )
    decision = inspect_loop(messages, model_calls=11, state=state)
    assert decision.kind == "stop"
