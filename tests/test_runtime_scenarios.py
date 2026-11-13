"""运行场景驱动测试：NC-09/10 直接驱动工具，NC-11 驱动真实父子进程。"""

from nextcoder.runtime_scenarios import run_runtime_scenario


def _package(tmp_path):
    package = tmp_path / "package"
    (package / "fixtures").mkdir(parents=True)
    (package / "fixtures" / "operation-note.txt").write_text("start", encoding="utf-8")
    return package


def test_nc09_recovery_scenario(tmp_path):
    result = run_runtime_scenario(_package(tmp_path), {"task_id": "NC-09"},
                                  tmp_path / "out")
    assert result["status"] == "passed", result
    assert all(check["passed"] for check in result["checks"])


def test_nc10_unknown_command_scenario(tmp_path):
    result = run_runtime_scenario(_package(tmp_path), {"task_id": "NC-10"},
                                  tmp_path / "out")
    assert result["status"] == "passed", result
    assert all(check["passed"] for check in result["checks"])


def test_nc11_cancel_and_timeout_scenario(tmp_path):
    result = run_runtime_scenario(_package(tmp_path), {"task_id": "NC-11"},
                                  tmp_path / "out")
    assert result["status"] == "passed", result
    assert all(check["passed"] for check in result["checks"])
