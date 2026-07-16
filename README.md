# 🤖 智能研究助手 (Research Assistant Agent)

基于 **LangChain + LangGraph + Chainlit** 构建的多工具 AI Agent，支持网页搜索、笔记管理和多轮对话记忆。

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.3+-green)
![Chainlit](https://img.shields.io/badge/Chainlit-2.11-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ 功能特性

- 🔍 **网页搜索** — 集成 Tavily API，实时检索最新信息
- 📝 **笔记管理** — 支持保存/列出/读取 Markdown 笔记，本地持久化
- 🧠 **对话记忆** — LangGraph SqliteSaver，跨轮次上下文记忆
- 💬 **流式输出** — Chainlit 构建 Web 界面，逐字流式打印
- 🛠️ **工具调用可视化** — 自动展示 Agent 每一步的工具调用过程

## 🏗️ 架构

```
用户输入 (Chainlit Web UI)
    ↓
LangChain Agent (DeepSeek)
    ↓
┌──────────────────────────────────────────────────┐
│  web_search    save_note    list_notes    read_note  │
│  (Tavily API)    (本地 .md)    (本地目录)    (本地读取)  │
└──────────────────────────────────────────────────┘
    ↓
LangGraph SqliteSaver → SQLite 持久化记忆
    ↓
流式输出 → Chainlit 界面
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/h4752/AgentDemo.git
cd langchain-agent-demo
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置 API Key

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY="你的DeepSeek API Key"
DEEPSEEK_BASE_URL="你的API地址"
TAVILY_API_KEY="你的Tavily API Key"
MODEL_NAME="deepseek-v4-pro"
```

> 💡 Tavily API Key 可在 [tavily.com](https://tavily.com) 免费获取

### 4. 运行

```bash
chainlit run app.py
```

浏览器访问 `http://localhost:8000` 即可使用。

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| Agent 框架 | LangChain + LangGraph |
| Web 界面 | Chainlit |
| 大模型 | DeepSeek |
| 搜索工具 | Tavily Search API |
| 记忆存储 | SQLite (SqliteSaver) |
| 笔记存储 | 本地 Markdown 文件 |

## 🎯 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 记忆存储 | SQLite (SqliteSaver) | 单机部署零依赖，适合 Demo 和个人使用；生产环境可切换 PostgreSQL |
| 搜索工具 | Tavily | 专为 AI Agent 优化的搜索 API，返回结构化结果，减少 token 浪费 |
| UI 框架 | Chainlit | 原生支持流式渲染、工具调用可视化，比 Streamlit/Gradio 更贴合 Agent 场景 |
| 笔记存储 | 本地 Markdown | 人类可读，无需引入数据库依赖，便于版本管理 |

## 📂 项目结构

```
.
├── app.py              # Chainlit UI 入口（WebSocket + 流式输出）
├── src/
│   ├── config.py       # 配置管理（环境变量、常量）
│   ├── tools.py        # Agent 工具集（搜索、笔记 CRUD）
│   ├── agent.py        # Agent 工厂函数
│   └── prompts.py      # 系统提示词
├── notebooks/          # LangChain 学习笔记
│   ├── AI通识与基础/    # OpenAI SDK 基础
│   └── LangChain入门/   # Agent、Memory、Tools、Prompt 等
├── tests/              # 单元测试
├── resources/          # SQLite 数据库
├── pyproject.toml      # 项目依赖与元数据
└── .env                # API Key 配置（不上传）
```

## 🔮 后续改进方向

- [ ] **RAG 检索** — 对接向量数据库（Chroma），让 Agent 能检索本地文档
- [ ] **对话摘要记忆** — 使用 SummarizationMiddleware 自动压缩长对话历史
- [ ] **可观测性** — 接入 LangSmith / LangFuse 追踪 Agent 决策链路
- [ ] **Human-in-the-loop** — 在关键操作（删除笔记等）前请求人工确认
- [ ] **多 Agent 协作** — 搜索 Agent + 写作 Agent 分工完成研究报告

## 📝 License

MIT
