# 🤖 智能研究助手 (Research Assistant Agent)

基于 **LangChain + LangGraph + Chainlit** 构建的多工具 AI Agent，支持网页搜索、RAG 知识库检索、文件上传索引、笔记管理和多轮对话记忆。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.3+-green)
![Chainlit](https://img.shields.io/badge/Chainlit-2.11-orange)
![Milvus](https://img.shields.io/badge/Milvus-3.0-00BFFF)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| 🔍 **网页搜索** | 集成 Tavily API，实时检索互联网最新信息 |
| 📚 **知识库检索** | Milvus 向量数据库 + DashScope Embedding，语义搜索本地文档 |
| 📤 **文件上传** | 支持 `.txt` `.md` `.pdf` `.docx`，上传后自动切分并索引到知识库 |
| 📝 **笔记管理** | 保存 / 列出 / 读取 Markdown 笔记，本地持久化 |
| 🧠 **对话记忆** | LangGraph InMemorySaver / SqliteSaver，跨轮次上下文记忆 |
| 💬 **流式输出** | Chainlit Web 界面，逐字流式打印，工具调用折叠展示 |

## 🏗️ 架构

```
用户输入 (Chainlit Web UI)
    │  上传文件 → 自动索引到 Milvus
    ▼
Agent (LangChain + DeepSeek) ── 系统提示词引导工具选择
    │
    ├─ search_knowledge_base  → Milvus 向量相似度搜索
    ├─ index_to_knowledge_base → 文本切分 → Embedding → Milvus 存储
    ├─ web_search             → Tavily API
    ├─ save_research_note     → 本地 .md 文件
    ├─ list_notes             → 本地目录
    └─ read_notes             → 本地读取
    │
    ▼
LangGraph Checkpointer → 对话记忆持久化
    │
    ▼
流式输出 → Chainlit 界面
```

### 数据流：文件上传 → RAG 检索

```
用户上传文件 (.txt/.md/.pdf/.docx)
    │
    ▼
解析提取文本 (pypdf / python-docx / 直接读取)
    │
    ▼
RecursiveCharacterTextSplitter
    │  chunk_size=500, overlap=50
    │  分隔符: \n\n → \n → 。 → . → " "
    ▼
DashScope Embedding (qwen3.7-text-embedding)
    │
    ▼
Milvus 向量存储
    │
    ▼
用户提问 → embedding → 相似度搜索 → 返回 top 4 相关片段
```

## 🚀 快速开始

### 1. 环境准备

- Python 3.10+
- Milvus 向量数据库（推荐 Docker 部署）

```bash
# 启动 Milvus（如果没有的话）
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

### 2. 克隆项目

```bash
git clone https://github.com/h4752/AgentDemo.git
cd langchain-agent-demo
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 配置 API Key

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY="你的 DeepSeek API Key"
DEEPSEEK_BASE_URL="你的 API 地址"
TAVILY_API_KEY="你的 Tavily API Key"
DASHSCOPE_API_KEY="你的 DashScope API Key"
MODEL_NAME="deepseek-v4-pro"
MILVUS_URL="http://localhost:19530"
```

| 服务 | 用途 | 获取地址 |
|---|---|---|
| DeepSeek | 大模型对话 | [platform.deepseek.com](https://platform.deepseek.com) |
| Tavily | 网页搜索 | [tavily.com](https://tavily.com) |
| DashScope | Embedding 向量化 | [dashscope.aliyun.com](https://dashscope.aliyun.com) |
| Milvus | 向量数据库 | 本地 Docker 部署 |

### 5. 运行

```bash
chainlit run app.py
```

浏览器访问 `http://localhost:8000`。

## 🛠️ 技术栈

| 类别 | 技术 |
|---|---|
| Agent 框架 | LangChain + LangGraph |
| Web 界面 | Chainlit |
| 大模型 | DeepSeek |
| Embedding | DashScope (qwen3.7-text-embedding) |
| 向量数据库 | Milvus |
| 文档切分 | LangChain RecursiveCharacterTextSplitter |
| 文件解析 | pypdf (PDF) + python-docx (Word) |
| 搜索工具 | Tavily Search API |
| 记忆存储 | SQLite (SqliteSaver) / InMemorySaver |
| 笔记存储 | 本地 Markdown 文件 |

## 🎯 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 向量数据库 | Milvus | 高性能开源方案，支持十亿级向量检索，pymilvus 3.0 API 简洁 |
| Embedding | DashScope | 中文语义理解优于多数开源模型，API 价格低 |
| 文档切分 | RecursiveCharacterTextSplitter | 中文句号 `。` 优先断句，chunk_size=500 + overlap=50，兼顾语义完整性和检索精度 |
| 记忆存储 | InMemorySaver (Web) / SqliteSaver (持久化) | Chainlit 会话级用内存，独立部署可切换 SQLite |
| 搜索工具 | Tavily | 专为 AI Agent 优化，返回结构化结果，减少 token 浪费 |
| UI 框架 | Chainlit | 原生支持流式渲染、文件上传、工具调用可视化 |
| 笔记存储 | 本地 Markdown | 人类可读，无需数据库依赖，便于版本管理 |

## 📂 项目结构

```
.
├── app.py                  # Chainlit UI 入口（文件上传 + Agent 对话）
├── src/
│   ├── config.py           # 配置管理（环境变量、常量）
│   ├── rag.py              # RAG 模块（KnowledgeBase 类、文件索引、向量检索）
│   ├── tools.py            # Agent 工具集（6 个工具）
│   ├── agent.py            # Agent 工厂函数
│   └── prompts.py          # 系统提示词
├── tests/
│   ├── test_rag.py         # RAG 模块单元测试
│   ├── test_tools.py       # 工具函数测试
│   ├── test_agent.py       # Agent 集成测试
│   └── conftest.py         # Pytest fixtures
├── resources/              # SQLite checkpoint、Chroma 持久化
├── pyproject.toml          # 项目依赖与元数据
└── .env                    # API Key 配置（不上传）
```

## 🔧 可配置环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_NAME` | `deepseek-v4-pro` | 大模型名称 |
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key |
| `TAVILY_API_KEY` | — | Tavily 搜索 API Key |
| `TAVILY_MAX_RESULTS` | `3` | 搜索结果返回数量 |
| `DASHSCOPE_API_KEY` | — | DashScope Embedding API Key |
| `EMBEDDING_MODEL` | `qwen3.7-text-embedding` | Embedding 模型 |
| `MILVUS_URL` | `http://localhost:19530` | Milvus 连接地址 |
| `COLLECTION_NAME` | `knowledge_base` | Milvus 集合名称 |
| `CHUNK_SIZE` | `500` | 文档切分块大小 |
| `CHUNK_OVERLAP` | `50` | 块间重叠字符数 |
| `TOP_K` | `4` | 检索返回条数 |
| `NOTES_PATH` | `./notes` | 笔记存储目录 |
| `CHECKPOINT_DB_PATH` | `./resources/checkpoint.db` | 对话记忆数据库路径 |

## 📝 License

MIT
