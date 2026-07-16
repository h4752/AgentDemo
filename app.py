import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain.messages import HumanMessage
from langchain_tavily import  TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
import chainlit as cl

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-pro")

tavily = TavilySearch(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def web_search(query: str) -> str:
    """这是一个网页搜索工具"""
    try:
        response = tavily.invoke(query)
        results = []
        for i, result in enumerate(response['results'], 1):
            results.append(
                f"{i}. {result['title']}\n"
                f"   URL: {result['url']}\n"
                f"   {result['content']}"
            )
        return "\n\n".join(results) if results else "未找到相关搜索结果"
    except Exception as e:
        return f"搜索失败: {type(e).__name__} - {str(e)}"


@tool
def save_research_note(title: str, content: str, path: str = "./notes") -> str:
    """将笔记保存到指定的路径文件夹内,文件名为标题名"""
    try:
        # 移除文件名中的非法字符
        safe_title = "".join(c for c in title if c not in r'\/:*?"<>|')
        Path(path).mkdir(parents=True, exist_ok=True)
        filepath = f"{path}/{safe_title}.md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"笔记已保存至: {filepath}"
    except Exception as e:
        return f"保存笔记失败: {type(e).__name__} - {str(e)}"

@tool
def list_notes(path: str = "./notes") -> str:
  """列出已有笔记"""
  try:
      Path(path).mkdir(parents=True, exist_ok=True)
      files = os.listdir(path)
      if not files:
          return "暂无笔记"
      return "\n".join(f"- {f}" for f in sorted(files))
  except Exception as e:
      return f"列出笔记失败: {type(e).__name__} - {str(e)}"


@tool
def read_notes(title: str, path: str = "./notes") -> str:
    """根据特定的标题检索笔记"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        files = os.listdir(path)
        matched = [f for f in files if f.startswith(title)]

        if not matched:
            return f"未找到 {title} 相关笔记"

        results = []
        for file in matched:
            file_path = f"{path}/{file}"
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                results.append(f"--- {file} ---\n{content}")

        return "\n\n".join(results)
    except Exception as e:
        return f"读取笔记失败: {type(e).__name__} - {str(e)}"

SYSTEM_PROMPT =  """你是一个专业的研究助手，善于运用各种工具来帮助用户高效地检索信息和整理知识。

  ## 你可以使用以下工具

  - web_search：从网页检索最新信息。当你需要事实核查、查找最新动态、或者回答你不知道的问题时使用
  - save_research_note：将重要内容保存为笔记（以标题为文件名创建 .md 文件）
  - list_notes：列出所有已保存的笔记标题
  - read_notes：根据关键词查找并阅读相关的笔记

  ## 行为规则

  - 当遇到需要最新信息的问题时，主动使用 web_search 搜索
  - 当搜索到重要信息时，主动建议用户保存笔记
  - 当用户提到之前讨论过的内容时，先用 list_notes 或 read_notes 找回历史笔记

  ## 输出要求

  - 用中文回答，结构清晰，分点列出关键信息
  - 引用搜索到的信息时，说明来源网址
  """


@cl.on_chat_start
async def start():
    conn = sqlite3.connect("resources/checkpoint.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    agent = create_agent(
        model=MODEL_NAME,
        tools=[web_search, save_research_note, list_notes, read_notes],
        system_prompt=SYSTEM_PROMPT,
        checkpointer = checkpointer,
    )

    cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(msg: cl.Message):
    agent = cl.user_session.get("agent")
    config = {"configurable": {"thread_id": msg.thread_id or "default"},
              "callbacks": [cl.AsyncLangchainCallbackHandler()]
              }

    response_msg = cl.Message(content="")
    await response_msg.send()

    for chunk, metadata in agent.stream(
            {"messages": [HumanMessage(content=msg.content)]},
            config=config,
            stream_mode="messages",
    ):

        if (
                isinstance(chunk, AIMessage)
                and chunk.content
                and not getattr(chunk, 'tool_calls', None)  # 跳过纯工具调用
        ):
            await response_msg.stream_token(chunk.content)

    await response_msg.update()


