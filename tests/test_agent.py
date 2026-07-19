import pytest
from langgraph.checkpoint.memory import InMemorySaver
from src.agent import create_research_agent


@pytest.fixture
def agent():
    """创建一个使用内存检查点的 Agent，适合测试"""
    checkpointer = InMemorySaver()
    return create_research_agent(checkpointer=checkpointer)


class TestAgentBasic:
    """Agent 基础集成测试"""

    def test_agent_invoke_returns_response(self, agent):
        """Agent 收到消息后能返回有效响应，不 crash"""
        config = {"configurable": {"thread_id": "test-invoke-1"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]},
            config=config,
        )

        assert "messages" in result
        # 最后一条消息应该是 AI 的回复
        last_msg = result["messages"][-1]
        assert hasattr(last_msg, "content")
        assert len(last_msg.content) > 0

    def test_agent_stream_yields_content(self, agent):
        """Agent 流式调用能产出内容块"""
        config = {"configurable": {"thread_id": "test-stream-1"}}
        chunks = list(
            agent.stream(
                {"messages": [{"role": "user", "content": "1+1等于几？"}]},
                stream_mode="messages",
                config=config,
            )
        )

        assert len(chunks) > 0
        # 应该有至少一个包含文本内容的 chunk
        content_chunks = []
        for chunk, metadata in chunks:
            if hasattr(chunk, "content") and chunk.content:
                content_chunks.append(chunk.content)

        assert len(content_chunks) > 0

    def test_agent_memory_across_turns(self, agent):
        """Agent 在同一 thread 内能记住上下文"""
        config = {"configurable": {"thread_id": "test-memory-1"}}

        # 第一轮：告诉 Agent 一个信息
        agent.invoke(
            {"messages": [{"role": "user", "content": "我最喜欢的颜色是深海蓝色。"}]},
            config=config,
        )

        # 第二轮：追问，验证它记住了
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "我最喜欢的颜色是什么？"}]},
            config=config,
        )

        last_msg = result["messages"][-1]
        assert "蓝" in last_msg.content or "blue" in last_msg.content.lower()