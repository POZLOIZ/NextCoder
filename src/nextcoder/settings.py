"""运行配置：模型、预算与输出上限。全部可从环境变量覆盖。"""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEXTCODER_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: SecretStr = SecretStr("")
    # 模型 profile：config/models.local.toml 的 [[models]] 名称。为空时若环境
    # 已提供 base_url/model 则用环境；否则自动选 primary（或第一条）。
    model_config_file: Path = Path("config/models.local.toml")
    model_profile: str = ""
    # 仅用于报告与 doctor 展示，不参与请求协议选择（目前统一 OpenAI 兼容）。
    llm_provider: str = ""
    # 额外的 OpenAI 兼容请求头；值支持 {env:VAR}，不要写密钥明文。
    llm_extra_headers: dict[str, str] = {}
    # OpenCode Go 要求稳定的 x-opencode-session；其他入口可留空关闭该头。
    llm_session_header: str = "x-opencode-session"
    llm_session: str = "nextcoder"
    # DeepSeek 官方可设为 enabled：适配器会保留并在后续请求回传
    # reasoning_content。不兼容 thinking 的入口保持 disabled；omit 表示完全不发
    # thinking 字段。
    llm_thinking: Literal["disabled", "enabled", "omit"] = "disabled"
    # 可选 reasoning_effort（low/medium/high）；留空时不发送该字段。只有显式
    # 设置时才传给 OpenAI 兼容端点，避免不支持的 provider 因未知参数报错。
    llm_reasoning_effort: str = ""
    # 连接与读取超时：连接超时短一些，读取超时按“两个流式 chunk 之间的等待”
    # 计算，避免长思考被误杀。主入口在首个 chunk 之前超时/断连时切换备援。
    llm_connect_timeout_seconds: float = 10.0
    llm_read_timeout_seconds: float = 300.0
    # 主入口 429 时切换；三元组齐全才启用，不记录密钥。
    llm_fallback_base_url: str = ""
    llm_fallback_model: str = ""
    llm_fallback_api_key: SecretStr = SecretStr("")

    # 会话与运行记录（操作表、检查点、完整日志、补丁）的保存位置。
    runs_root: Path = Path(".nextcoder")

    # 预算：超出即停止并说明原因。
    # model_calls_limit / wall_time_seconds 按每轮统计；
    # application_input_token_budget 是单次请求的总窗口（含系统提示、工具定义、
    # 历史消息、工具结果和输出预留），不是累计费用；累计费用看 usage。
    # 可用输入 = 总窗口 - output_reserve_tokens；压缩在可用输入的 90% 触发。
    model_calls_limit: int = 20
    wall_time_seconds: int = 600
    application_input_token_budget: int = 210000
    output_reserve_tokens: int = 4096
    # 该值同时作为每次模型请求的 max_tokens 上限和输入预算里的输出预留。
    # 它必须是模型实际支持的输出上限；只写在预算里、不发给接口就等于没有约束。
    # 用掉多少比例的模型调用后提醒收敛；0 表示关闭。只提醒，不阻断。
    budget_warn_ratio: float = 0.75

    # 工具输出限制：模型只看到摘录，完整输出落文件，可回读。
    output_excerpt_chars: int = 6000
    max_output_bytes: int = 2_000_000
    command_timeout_seconds: int = 120
    # 命令工具执行的 PATH 前缀：任务自有虚拟环境的 bin 应优先于宿主环境。
    command_path_prepend: str = ""
    # 命令在 Docker 容器内执行：任务目录以 bind mount 出现在 workdir。
    # 留空表示直接在宿主机执行；容器内命令仍按 call_id 做崩溃恢复。
    command_container: str = ""
    command_container_workdir: str = "/testbed"
    # 容器里的环境激活前缀，例如 conda activate testbed；留空则不激活。
    command_container_activate: str = ""
    list_files_limit: int = 300
    max_diff_context_files: int = 50

    # 上下文压缩：只有一个触发线——可用输入的 90%。
    # 触发后先把旧工具正文折叠成占位，重新测量：
    # - 折叠后 <= 低水位（可用输入的 70%）：说明廉价折叠已经够用，直接继续；
    # - 折叠后 > 低水位：即使已经低于 90% 触发线，也请求模型摘要。
    # 低水位只做“折叠后是否摘要”的决策线，不是摘要后的验收目标。
    # 摘要保留最近 16% 的原文，其余压缩成摘要；摘要后仍高于 90% 触发线时，
    # 最多重试一次，第二次把近期预算降到 8%，并检查是否真的收敛。
    compaction_trigger_ratio: float = 0.9
    compaction_low_watermark_ratio: float = 0.7
    # 第一层：先把旧工具正文折叠成占位。关闭后压缩直接进入第二层模型摘要，
    # 也就是“保留最近工作集、总结更早消息”的旧式做法（评测基线）。
    compaction_stub_first: bool = True
    compaction_keep_recent_groups: int = 16
    # 近期工作集软预算：可用输入的 16%，与 dsh 的 retainRatio 一致。
    compaction_recent_token_ratio: float = 0.16
    # 摘要后仍高于触发线时的重试次数与重试时的近期预算（可用输入的 8%）。
    compaction_summary_retry_limit: int = 1
    compaction_retry_recent_ratio: float = 0.08
    compaction_stub_chars: int = 240
    compaction_summary_chars: int = 1800
    # 第二层摘要调用：只总结，不执行工具；输入过长时保留首尾关键过程。
    summary_enabled: bool = True
    summary_input_token_budget: int = 16000
    summary_message_chars: int = 1600

    # 反死循环：连续相同调用只做提醒，不缓存、不阻断、不中止任务。
    # 新的调用即使参数完全相同也正常执行，因为环境可能已经变化
    # （例如先安装依赖，再重跑同一个 pytest 命令）。
    repeat_warn_limit: int = 3
    # 搜索/查找动作共享一个计数，只用于在次数很高时提醒检查方向；
    # 搜索次数不能证明信息已经足够，因此不硬禁用，也不要求先改代码。
    # 真正的运行兜底是 model_calls_limit 与 wall_time_seconds。
    search_warn_limit: int = 16
    # 连续多个命令零输出时提醒检查 stderr/管道退出码；0 表示关闭。
    empty_output_warn_limit: int = 3
    # 模型因输出长度被截断时的重试次数：只丢弃截断响应并提醒拆小步骤，
    # 不把一次截断当成任务失败；超过次数才停止。0 表示不重试。
    length_retry_limit: int = 2
    # 连续多次 apply_patch/write_file 失败后提醒先重新读取；只提醒，不阻断。
    edit_failure_warn_limit: int = 4
    # 连续多次工具调用没有成功修改文件时提醒检查是否只探索不落盘；只提醒，不阻断。
    # 0 表示关闭。
    no_edit_warn_limit: int = 12
    # 进展检测：空修改连续出现、或文件反复回到同一状态时要求重新规划。
    # 单个动作不直接终止；重新规划后仍无新文件状态/测试结果才提前停止。
    loop_noop_replan_limit: int = 2
    loop_rollback_replan_count: int = 3
    loop_rollback_stop_count: int = 4
    loop_edit_gap_nudge: int = 8
    loop_block_calls: int = 1
    loop_stop_after_calls: int = 8

    # 项目指令文件：仓库根的 AGENTS.md，会话开始读取一次，进入稳定系统前缀。
    # 超长截断；文件不存在静默跳过；不做变量展开、不执行内容。
    agents_enabled: bool = True
    agents_file: str = "AGENTS.md"
    agents_max_chars: int = 8000

    # 项目记忆：会话入口注入索引（memory.md 随写入自动重写），详情在
    # .nextcoder/memory/ 下；条目不设有效期，停用只能显式触发。
    memory_enabled: bool = True
    memory_prompt_limit: int = 64
    memory_snapshot_chars: int = 4000
    # 后台整理 agent：会话第一轮与每累计 N 轮用户请求触发一次（按用户请求
    # 轮数计，不是工具轮数）；合并重复、停用过时、从旧会话补提取。
    memory_organizer_enabled: bool = True
    memory_organizer_turn_interval: int = 5

    # 权限模式（从最谨慎到最宽松）：
    # - manual：每次写文件、删文件、运行命令、调用外部工具都先询问；
    # - workspace：任务目录内读写自动放行，运行命令和外部工具先询问；
    # - auto：任务目录内读写自动放行；命令只有在沙箱真实可用时才自动放行，
    #   沙箱不可用或未开启时先询问；
    # - bypass：除不可逆命令底线外全部放行，只用于完全信任的隔离环境。
    # 所有模式都保留不可逆命令底线（rm 根目录、mkfs、dd 写设备、提权等），
    # 文件工具也始终不能越过任务目录。规则文件相对任务目录，deny 优先于 allow。
    # 旧配置里的 standard/aggressive 会自动迁移为 workspace/bypass。
    permission_mode: Literal["manual", "workspace", "auto", "bypass"] = "workspace"
    permissions_file: Path = Path(".nextcoder/permissions.json")

    @field_validator("permission_mode", mode="before")
    @classmethod
    def _migrate_permission_mode(cls, value):
        """兼容旧配置：保留旧名字的行为，不让升级后突然失去权限。"""
        if isinstance(value, str):
            legacy = {"standard": "workspace", "aggressive": "bypass"}
            return legacy.get(value.strip().lower(), value)
        return value

    # 命令沙箱：宿主机命令用 bubblewrap 包一层（容器内命令本就隔离）。
    # 开启后默认 fail-closed：bwrap 缺失时拒绝执行，而不是静默裸跑。
    # 网络默认关闭（--unshare-net）；需要装依赖时显式打开。
    command_sandbox: bool = False
    command_sandbox_fail_closed: bool = True
    command_sandbox_network: bool = False
    sandbox_writable_roots: list[str] = []
    sandbox_readonly_roots: list[str] = []
    # 沙箱内只透传这些环境变量；其余全部清除，避免密钥进入命令进程。
    sandbox_env_passthrough: list[str] = []

    # 生命周期 hooks：before_tool_use/after_tool_use 等事件；
    # 内置重复调用、搜索预算、空输出提醒也走同一套 hook 总线。
    hooks_enabled: bool = True
    hooks_file: Path = Path(".nextcoder/hooks.json")

    # 任务清单：plan_create/plan_update 工具 + 每请求快照注入 + 收尾核对。
    plan_enabled: bool = True

    # 子 agent：父 agent 通过 delegate 派发独立角色；只读角色共享任务目录，
    # tester/coder 在 git worktree 中工作，结束后导出 changes.patch。
    # 子预算按每轮重新计算；后台并发上限决定同时运行的子 agent 数。
    delegation_enabled: bool = True
    delegation_model_calls_limit: int = 12
    delegation_wall_time_seconds: int = 900
    delegation_background_max: int = 2
    # 返回给父模型的报告正文预算；完整内容另存 subagents/<id>/report.md。
    # 超出时按 dsh 的 4:1 头尾比例折叠，元数据与产物路径固定保留。
    delegation_report_chars: int = 8192
    delegation_max_resumes: int = 1
    # 空字符串表示在系统临时目录下创建 worktree 根；不会污染主仓库。
    delegation_worktree_root: str = ""

    # 技能：发现 <task_dir>/.nextcoder/skills、skills_roots 与
    # ~/.nextcoder/skills 下的 SKILL.md；目录注入受字符预算限制。
    skills_enabled: bool = True
    skills_roots: list[str] = []
    skills_prompt_chars: int = 1200

    # MCP：配置文件不存在时零开销（不启动任何子进程、不注入工具）。
    mcp_enabled: bool = True
    mcp_config_file: Path = Path(".nextcoder/mcp.json")
