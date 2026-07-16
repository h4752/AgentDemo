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
┌──────────────────────────────────────┐
│  web_search    save_note    list_notes    read_note  │
│  (Tavily API)    (本地 .md)    (本地目录)    (本地读取)  │
└──────────────────────────────────────┘
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
uv add langchain langchain-deepseek langchain-tavily langgraph langgraph-checkpoint-sqlite chainlit python-dotenv
```

### 3. 配置 API Key

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY="你的DeepSeek API Key"
DEEPSEEK_BASE_URL="你的API地址"
TAVILY_API_KEY="你的Tavily API Key"
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

## 📂 项目结构

```
.
├── app.py              # 主程序（Agent 定义 + Chainlit 界面）
├── notebooks/          # 学习笔记（LangChain 入门）
├── notes/              # 研究笔记保存目录
├── checkpoints.db      # 对话记忆数据库
└── .env                # API Key 配置（不上传）
```

## 📝 License

MIT
