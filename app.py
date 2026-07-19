import asyncio
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
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

    # 流式输出，只显示 AI 文本回复，工具调用/结果用 Step 折叠展示
    msg = cl.Message(content="")
    await msg.send()

    # 跟踪工具调用：{tool_call_id: Step}
    tool_steps: dict[str, cl.Step] = {}

    try:
        async for chunk, metadata in agent.astream(
            {"messages": [HumanMessage(content=user_content)]},
            config=config,
            stream_mode="messages",
        ):
            # ---- 跳过工具结果（ToolMessage），不直接显示 ----
            if isinstance(chunk, ToolMessage):
                # 将工具输出写入对应的 Step
                tc_id = getattr(chunk, "tool_call_id", None)
                if tc_id and tc_id in tool_steps:
                    tool_steps[tc_id].output = (
                        str(chunk.content)[:2000]
                    )
                    await tool_steps[tc_id].update()
                continue

            # ---- 只处理 AI 消息 ----
            if not isinstance(chunk, AIMessageChunk):
                continue

            # ---- 工具调用：创建折叠 Step ----
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    tc_name = tc.get("name", "unknown")
                    tc_id = tc.get("id", "")
                    if tc_id and tc_id not in tool_steps:
                        step = cl.Step(
                            name=f"🔧 {tc_name}",
                            type="tool",
                            parent_id=msg.id,
                        )
                        step.input = str(tc.get("args", {}))[:1000]
                        tool_steps[tc_id] = step
                        await step.send()

            # ---- AI 文本回复：流式输出给用户 ----
            if chunk.content and isinstance(chunk.content, str):
                await msg.stream_token(chunk.content)

    except Exception as e:
        msg.content = (
            f"❌ 处理消息时出错: {type(e).__name__} - {str(e)}"
        )
        await msg.update()

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
            # 在线程池中运行同步 I/O 操作，避免阻塞事件循环
            result = await asyncio.to_thread(
                index_file, file.path, source_name=file.name
            )
            # 将技术性结果翻译为更友好的提示
            chunk_count = ""
            for word in result.split():
                if word.isdigit():
                    chunk_count = word
                    break
            status_msg.content = (
                f"✅ `{file.name}` 已索引完成，共 {chunk_count} 个文本片段。\n\n"
                f"你现在可以提问了，例如：「{file.name} 里讲了什么？」"
            )
            await status_msg.update()
        except Exception as e:
            status_msg.content = (
                f"❌ 索引 `{file.name}` 失败: {type(e).__name__} - {str(e)}"
            )
            await status_msg.update()
