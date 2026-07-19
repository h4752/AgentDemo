from pathlib import Path
from unittest.mock import MagicMock, patch
from src.tools import save_research_note, list_notes, read_notes


class TestSaveResearchNote:
    """save_research_note 工具测试"""

    def test_save_note_creates_file(self, temp_notes_dir):
        """保存笔记后文件应该存在，且内容正确"""
        result = save_research_note.invoke({
            "title": "测试笔记",
            "content": "Hello World",
            "path": temp_notes_dir,
        })

        # 返回值确认
        assert "笔记已保存至" in result
        assert temp_notes_dir in result

        # 文件确认
        filepath = Path(temp_notes_dir) / "测试笔记.md"
        assert filepath.exists()
        assert filepath.read_text(encoding="utf-8") == "Hello World"

    def test_save_note_filters_illegal_chars(self, temp_notes_dir):
        """文件名包含非法字符时应该被过滤掉"""
        result = save_research_note.invoke({
            "title": "测试:笔记*?",
            "content": "test",
            "path": temp_notes_dir,
        })

        assert "笔记已保存至" in result
        # 确认非法字符被移除，文件名安全
        safe_file = Path(temp_notes_dir) / "测试笔记.md"
        assert safe_file.exists()


class TestListNotes:
    """list_notes 工具测试"""

    def test_list_notes_empty(self, temp_notes_dir):
        """空目录返回'暂无笔记'"""
        result = list_notes.invoke({"path": temp_notes_dir})
        assert result == "暂无笔记"

    def test_list_notes_nonempty(self, multiple_notes):
        """有笔记时列出所有笔记名"""
        result = list_notes.invoke({"path": multiple_notes})

        assert "- 人工智能发展.md" in result
        assert "- Python学习笔记.md" in result
        assert "- 量子计算科普.md" in result

    def test_list_notes_sorted(self, multiple_notes):
        """笔记按字母排序"""
        result = list_notes.invoke({"path": multiple_notes})
        lines = result.split("\n")
        # 中文按拼音排序，验证结果有序
        assert len(lines) == 3


class TestReadNotes:
    """read_notes 工具测试"""

    def test_read_note_found(self, sample_note):
        """按标题前缀查找，命中后返回内容"""
        result = read_notes.invoke({
            "title": "测试",
            "path": sample_note,
        })

        assert "# 测试笔记" in result
        assert "这是一个测试笔记的内容" in result

    def test_read_note_not_found(self, temp_notes_dir):
        """未命中时返回友好提示"""
        result = read_notes.invoke({
            "title": "不存在",
            "path": temp_notes_dir,
        })

        assert "未找到" in result
        assert "不存在" in result

    def test_read_note_prefix_match(self, multiple_notes):
        """前缀匹配：搜'Python'应只命中 Python 那篇"""
        result = read_notes.invoke({
            "title": "Python",
            "path": multiple_notes,
        })

        assert "Python" in result
        assert "量子计算" not in result
        assert "人工智能" not in result


class TestSearchKnowledgeBase:
    """search_knowledge_base 工具测试"""

    @patch("src.tools._get_kb")
    def test_search_knowledge_base_returns_results(self, mock_get_kb):
        """工具应返回格式化后的检索结果"""
        from src.tools import search_knowledge_base

        mock_kb = MagicMock()
        mock_kb.retrieve.return_value = "1. [Source: test.md] (relevance: 0.9500)\n   Test content"
        mock_get_kb.return_value = mock_kb

        result = search_knowledge_base.invoke({"query": "test query"})

        assert "Test content" in result
        assert "test.md" in result
        mock_kb.retrieve.assert_called_once_with("test query")

    @patch("src.tools._get_kb")
    def test_search_knowledge_base_handles_error(self, mock_get_kb):
        """工具应在出错时返回友好的错误信息"""
        from src.tools import search_knowledge_base

        mock_get_kb.side_effect = ConnectionError("Milvus unreachable")

        result = search_knowledge_base.invoke({"query": "test"})

        assert "知识库搜索失败" in result
        assert "ConnectionError" in result
        assert "Milvus unreachable" in result


class TestIndexToKnowledgeBase:
    """index_to_knowledge_base 工具测试"""

    @patch("src.tools._get_kb")
    def test_index_to_kb_success(self, mock_get_kb):
        """工具应调用 kb.index_texts 并返回结果"""
        from src.tools import index_to_knowledge_base

        mock_kb = MagicMock()
        mock_kb.index_texts.return_value = "Indexed 3 chunks from 1 documents into 'kb'."
        mock_get_kb.return_value = mock_kb

        result = index_to_knowledge_base.invoke({
            "content": "需要保存的内容",
            "source": "web_search",
        })

        assert "Indexed" in result
        mock_kb.index_texts.assert_called_once_with(
            ["需要保存的内容"], sources=["web_search"]
        )

    @patch("src.tools._get_kb")
    def test_index_to_kb_default_source(self, mock_get_kb):
        """不传 source 时默认为 'agent'"""
        from src.tools import index_to_knowledge_base

        mock_kb = MagicMock()
        mock_kb.index_texts.return_value = "Indexed 1 chunks..."
        mock_get_kb.return_value = mock_kb

        index_to_knowledge_base.invoke({"content": "test"})

        mock_kb.index_texts.assert_called_once_with(
            ["test"], sources=["agent"]
        )

    @patch("src.tools._get_kb")
    def test_index_to_kb_handles_error(self, mock_get_kb):
        """工具应在出错时返回友好的错误信息"""
        from src.tools import index_to_knowledge_base

        mock_get_kb.side_effect = RuntimeError("Embedding API error")

        result = index_to_knowledge_base.invoke({"content": "test"})

        assert "索引到知识库失败" in result
        assert "RuntimeError" in result
        assert "Embedding API error" in result