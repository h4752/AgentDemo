import sqlite3
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import chainlit as cl
from src.agent import create_research_agent
from src.config import CHECKPOINT_DB_PATH

@cl.on_chat_start
async def start():
  conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
  checkpointer = SqliteSaver(conn)
  checkpointer.setup()

  agent = create_research_agent(checkpointer=checkpointer)
  cl.user_session.set("agent", agent)

@cl.on_message
async def on_message(msg: cl.Message):
  agent = cl.user_session.get("agent")
  config = {
      "configurable": {"thread_id": msg.thread_id or "default"},
      "callbacks": [cl.AsyncLangchainCallbackHandler()],
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
          and not getattr(chunk, 'tool_calls', None)
      ):
          await response_msg.stream_token(chunk.content)

  await response_msg.update()
