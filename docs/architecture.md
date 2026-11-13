# 运行时结构

NextCoder 把「模型循环」和「终端界面」分开：界面只消费事件，真正的工具执行、权限判断、上下文压缩、子 agent 和持久化都在 `nextcoder.runtime` 与 `nextcoder.sessions` 里。这样同一套运行时可以被 CLI 和 TUI 共同使用。

## 一次请求的路径

```
CLI / TUI
   │  run_turn(request)
   ▼
sessions.manager ──► AgentRunner（LangGraph）
   │                    │
   │                    ├─ 稳定系统前缀：基础规则 + AGENTS.md + 记忆快照 + 技能目录
   │                    ├─ 每轮临时尾部：任务清单快照 + 最新用户请求
   │                    ├─ 上下文检查：折叠旧工具正文 → 必要时生成语义摘要
   │                    ├─ 模型调用（可流式；429/超时在首块前换备援）
   │                    └─ 工具调用
   │                         │
   │                         ├─ before_tool_use hooks
   │                         ├─ 权限判定（manual/workspace/auto/bypass）
   │                         ├─ 真正执行：文件、命令、计划、MCP、子 agent
   │                         ├─ after_tool_use hooks
   │                         └─ 操作审计 + 事件
   ▼
checkpoint / rollout / report / final.patch
```

关键点：

- **稳定前缀**只在会话开始时组装一次。会话中途改 `AGENTS.md`、记忆或技能目录，下一会话才进入前缀，避免破坏服务端前缀缓存。
- **任务清单**放在请求尾部，不写检查点，也不改稳定前缀；TUI 直接读 `.nextcoder/task-plan.json`。
- **工具调用**统一经过 `ToolBox.execute`：先跑 before hook，再判权限，再执行，最后写操作记录和 after hook。权限拒绝、hook 阻止和工具异常都会留下可读记录。
- **上下文压缩**只有一条触发线：估算输入超过 90% 才进入压缩。先折叠旧工具正文；折叠后不超过 70% 就直接继续，超过 70% 才调用摘要模型。摘要后仍超线时最多重试一次，近期预算从 16% 降到 8%，只在结果变小且回到线下时接受。

## 目录与模块

| 路径 | 职责 |
|------|------|
| `src/nextcoder/cli.py` | Typer 入口：`doctor`、`models`、`hooks`、`run`、`chat`、`tui`、`compact`、`sessions`、`memory` |
| `src/nextcoder/settings.py` | Pydantic 配置；模型、上下文、权限、沙箱、hooks、子 agent 的默认值 |
| `src/nextcoder/model_config.py` | `config/models.local.toml` 的多 provider profile 解析与选择 |
| `src/nextcoder/llm.py` | Chat 模型、thinking 内容、流式、首块前备援、超时 |
| `src/nextcoder/runtime/agent.py` | LangGraph 状态机、模型节点、工具节点、压缩与恢复 |
| `src/nextcoder/runtime/context.py` | token 估算、工具正文折叠、摘要请求与裁剪视图 |
| `src/nextcoder/runtime/permissions.py` | 四档权限、路径/命令/MCP 判定、不可逆命令底线 |
| `src/nextcoder/runtime/hooks.py` | `before_tool_use` / `after_tool_use` 总线与内置提醒 |
| `src/nextcoder/runtime/delegation.py` | 子 agent 角色、worktree、后台任务、取消与恢复 |
| `src/nextcoder/runtime/mcp.py` | stdio 与 Streamable HTTP MCP 服务器管理 |
| `src/nextcoder/runtime/planning.py` | 任务清单的唯一事实源 `.nextcoder/task-plan.json` |
| `src/nextcoder/runtime/instructions.py` | 读取任务目录的 `AGENTS.md` |
| `src/nextcoder/runtime/skills.py` | `SKILL.md` 发现与目录快照 |
| `src/nextcoder/runtime/patch.py` | 基线复制、补丁导出、忽略目录剪枝 |
| `src/nextcoder/sessions/manager.py` | 会话创建/加载、回合执行、标题、报告与缓存统计 |
| `src/nextcoder/sessions/store.py` | 工具操作审计 SQLite 与状态机 |
| `src/nextcoder/sessions/rollout.py` | 人类可读的 `rollout.jsonl` 事件流 |
| `src/nextcoder/sessions/memory.py` | 项目记忆 SQLite 与 Markdown 投影 |
| `src/nextcoder/sessions/organizer.py` | 后台记忆整理 agent |
| `src/nextcoder/tools/` | 文件、命令、编辑、计划、记忆、技能、MCP 工具 |
| `src/nextcoder/tui/` | Textual 界面、弹窗、补全、Markdown 回合卡片 |

## 会话产物

会话保存在任务目录下：

```
<repo>/.nextcoder/
├── sessions/<session_id>/
│   ├── session.json          # id、任务目录、created_at、title
│   ├── rollout.jsonl         # 人类可读事件流，仅供查看/回放
│   ├── checkpoints.sqlite    # 崩溃恢复与多轮续聊的事实源
│   ├── operations.sqlite     # 工具操作审计与状态机
│   ├── baseline/             # 会话开始时的仓库副本
│   ├── turns/<turn_id>/
│   │   ├── report.json       # 状态、用量、权限、hook、子 agent
│   │   ├── final.patch       # 本轮累计补丁
│   │   └── logs/             # 每个工具调用的完整输出
│   └── subagents/<child_id>/
│       ├── report.md
│       ├── report.json
│       └── changes.patch
├── task-plan.json            # 任务清单
├── memory.sqlite             # 项目记忆
├── memory.md                 # 人可读投影
└── hooks.json / permissions.json / mcp.json   # 可选的项目配置
```

`final.patch` 可以直接用 `git apply` 应用。`rollout.jsonl` 不参与恢复；恢复只看 `checkpoints.sqlite` 和 `operations.sqlite`。进程崩溃后重新进入同一会话，会先核对已开始的文件操作并补齐缺失的工具结果。

## 模型与上下文

默认模型窗口是 210,000 tokens，输出预留 4,096，因此可用输入是 205,904。90% 触发线是 185,313，70% 决策线是 144,132。每次请求的 `max_tokens` 真正使用输出预留值，而不只是预算里的数字。

缓存统计区分冷启动和暖调用：

- 第一次请求没有前缀可命中，显示「缓存 首次」，不把它当成缓存失效；
- 同一轮后续调用接近全命中时，显示总命中率；
- 总命中率低但暖命中率高时，显示「缓存 75%（暖 94%）」，避免一次冷启动把数字拉低。

模型入口由 `config/models.local.toml` 的 profile 描述，支持任意 OpenAI 兼容端点、可选 headers、429 备援三元组。profile 只保存密钥的环境变量名，不保存密钥本身。

## 恢复与审计

每个工具操作在 `operations.sqlite` 里有状态：`prepared`、`started`、`completed`、`failed`。恢复时按调用 ID 找到未配对的操作，核对文件哈希或命令结果，再补齐工具消息。权限请求、hook 触发、计划更新、压缩和子 agent 状态都会写 `rollout.jsonl`，方便回看一轮到底发生了什么。

## 扩展点

- **工具**：在 `tools/` 实现 `Tool` 并在 `registry.py` 注册；白名单、权限和 hook 自动生效。
- **技能**：在任务目录放 `SKILL.md`，会话开始时生成目录，模型用 `load_skill` 读正文。
- **MCP**：在 `.nextcoder/mcp.json` 加 stdio 或 HTTP 服务器，工具名自动变成 `mcp__<server>__<tool>`。
- **hooks**：在 `.nextcoder/hooks.json` 声明 `block`、`note`、`reminder`，不执行任意脚本。
- **模型 provider**：在 `config/models.local.toml` 加 `[[models]]`，用 `--profile` 选择。
- **子 agent**：`researcher` / `reviewer` 只读，`tester` / `coder` 在 git worktree 中写；不能嵌套委派。
