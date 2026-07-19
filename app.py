import chainlit as cl
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from chainlit.element import Element
from src.agent import create_research_agent
from src.rag import index_file

WELCOME_MESSAGE = """# 智能研究助手 🧠

我可以帮你完成以下任务：

| 功能 | 说明 |
|---|---|
| 🔍 **网页搜索** | 检索互联网最新信息 |
| 📚 **知识库检索** | 搜索已索引的内部文档 |
| 📤 **文件上传** | 支持 `.txt` `.md` `.pdf` `.docx`，上传后自动索引到知识库 |
| 📝 **笔记管理** | 保存、列出、阅读研究笔记 |
| 💬 **多轮对话** | 记住上下文，连续深入讨论 |

**开始提问，或者直接上传文件吧！**
"""


@cl.on_chat_start
async def on_chat_start() -> None:
    """会话开始：初始化 Agent 并显示欢迎消息。"""
    checkpointer = InMemorySaver()
    agent = create_research_agent(checkpointer=checkpointer)
    cl.user_session.set("agent", agent)
    await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """处理用户消息：先处理上传文件，再进行 Agent 对话。"""
    agent = cl.user_session.get("agent")  # type: ignore[call-overload]

    # ---- 处理上传文件 ----
    if message.elements:
        await _handle_uploaded_files(message.elements)

    # ---- Agent 对话 ----
    user_content = message.content or ""
    if not user_content.strip():
        # 只上传了文件，没有文字内容
        return

    config = {"configurable": {"thread_id": cl.context.session.id}}

    # 使用流式模式输出
    msg = cl.Message(content="")
    await msg.send()

    try:
        async for chunk, metadata in agent.astream(
            {"messages": [HumanMessage(content=user_content)]},
            config=config,
            stream_mode="messages",
        ):
            # 在 stream_mode="messages" 下，chunk 是 (message_chunk, metadata) 元组
            # message_chunk 包含增量内容
            if hasattr(chunk, "content") and chunk.content:
                # 只追加文本内容，跳过 tool_calls 等非文本 chunk
                if isinstance(chunk.content, str):
                    await msg.stream_token(chunk.content)
    except Exception as e:
        await msg.update(content=f"处理消息时出错: {type(e).__name__} - {str(e)}")

    await msg.update()


async def _handle_uploaded_files(
    elements: list[Element],
) -> None:
    """将用户上传的文件索引到知识库。

    使用 index_file() 处理每个文件，支持 .txt .md .pdf .docx。
    """
    for element in elements:
        if not isinstance(element, cl.File):
            continue

        file: cl.File = element
        status_msg = cl.Message(content=f"正在索引 `{file.name}` ...")
        await status_msg.send()

        try:
            result = index_file(file.path, source_name=file.name)
            await status_msg.update(content=f"✅ {result}")
        except Exception as e:
            await status_msg.update(
                content=f"❌ 索引 `{file.name}` 失败: {type(e).__name__} - {str(e)}"
            )
