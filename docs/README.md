# NextCoder 文档

NextCoder 是一个在现有代码仓库里工作的终端编程助手。这里放机制设计和日常使用说明；根目录的 [README](../README.md) 负责安装、配置和快速上手。

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 运行时结构、一次请求的完整路径、会话产物与扩展点 |
| [permissions.md](permissions.md) | 四档权限、规则文件、审批流程、bubblewrap 沙箱与安全底线 |
| [tui.md](tui.md) | 启动页、快捷键、斜杠命令、`@` 文件引用、缓存与标题显示 |

## 阅读顺序

第一次接触先看根目录 README 的「快速开始」，然后看 `tui.md` 了解日常操作。想改代码或排查行为时看 `architecture.md`；要调整权限、沙箱或 hooks 时看 `permissions.md`。

## 文档约定

- 命令、字段、路径尽量写真实值；不确定的地方标成「可选」或「未验证」，不写猜测。
- 中文为主，保留 `session`、`checkpoint`、`hook`、`MCP` 等必要英文术语。
- 截图由真实 Textual 界面导出，PNG 在 `docs/assets/`。
