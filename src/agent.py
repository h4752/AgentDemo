from langchain.agents import create_agent
from src.config import MODEL_NAME
from src.tools import web_search, save_research_note, list_notes, read_notes
from src.prompts import SYSTEM_PROMPT


def create_research_agent(checkpointer=None):
    """创建研究助手 Agent，可选传入 checkpointer 以启用对话记忆"""
    return create_agent(
        model=MODEL_NAME,
        tools=[web_search, save_research_note, list_notes, read_notes],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )