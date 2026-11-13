"""一次任务的完整运行：创建会话并在其中执行一轮。"""

from pathlib import Path

from nextcoder.sessions.manager import TurnOutcome, create_session, run_turn
from nextcoder.settings import Settings

RunOutcome = TurnOutcome


def run_task(settings: Settings, task_dir: Path, request: str) -> TurnOutcome:
    """在任务副本上运行助手；结束后导出补丁、操作记录与总结。"""
    session = create_session(settings, task_dir)
    return run_turn(settings, session, request)
