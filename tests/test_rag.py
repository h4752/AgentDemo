from unittest.mock import MagicMock, patch
import pytest
from src.rag import KnowledgeBase, retrieve_documents, index_file


class TestKnowledgeBaseInit:
    """KnowledgeBase 初始化测试"""

    def test_init_stores_config(self):
        """初始化后存储所有配置值，但不建立连接"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=300,
            chunk_overlap=30,
            top_k=5,
            embedding_model_name="test-model",
            dashscope_api_key="test-key",
        )
        assert kb.milvus_url == "http://test:19530"
        assert kb.collection_name == "test_col"
        assert kb.chunk_size == 300
        assert kb.chunk_overlap == 30
        assert kb.top_k == 5
        assert kb.embedding_model_name == "test-model"
        assert kb.dashscope_api_key == "test-key"
        # 懒初始化：尚未建立连接
        assert kb._client is None
        assert kb._embed_model is None

    def test_init_defaults(self):
        """使用默认参数初始化"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            top_k=4,
            embedding_model_name="qwen3.7-text-embedding",
            dashscope_api_key="key",
        )
        assert kb.top_k == 4
        assert kb.chunk_size == 500
        assert kb.chunk_overlap == 50

    @patch("src.rag.MilvusClient")
    def test_client_lazy_property(self, mock_milvus_client):
        """首次访问 client 属性时才创建 MilvusClient"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            top_k=4,
            embedding_model_name="test-model",
            dashscope_api_key="test-key",
        )
        assert kb._client is None
        client = kb.client
        assert client is not None
        # 再次访问返回同一实例
        assert kb.client is client
        mock_milvus_client.assert_called_once_with("http://test:19530")


class TestKnowledgeBaseRetrieve:
    """retrieve 方法测试（mock 掉 search）"""

    @pytest.fixture
    def kb(self):
        """返回一个 mock 了 search 的 KnowledgeBase 实例"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            top_k=4,
            embedding_model_name="test-model",
            dashscope_api_key="test-key",
        )
        return kb

    def test_retrieve_formats_results_correctly(self, kb):
        """检索结果应格式化为编号列表，含来源和相似度"""
        kb.search = MagicMock(return_value=[
            {
                "id": 0,
                "distance": 0.95,
                "entity": {"text": "文本内容一", "source": "doc1.md"},
            },
            {
                "id": 1,
                "distance": 0.85,
                "entity": {"text": "文本内容二", "source": "doc2.md"},
            },
        ])

        result = kb.retrieve("测试查询")

        assert "文本内容一" in result
        assert "doc1.md" in result
        assert "文本内容二" in result
        assert "doc2.md" in result
        assert result.startswith("1.")  # 编号列表
        assert "Source:" in result
        assert "relevance:" in result

    def test_retrieve_empty_results(self, kb):
        """无结果时返回友好提示"""
        kb.search = MagicMock(return_value=[])

        result = kb.retrieve("不存在的查询")

        assert "No relevant documents" in result

    def test_retrieve_handles_missing_entity_gracefully(self, kb):
        """结果缺少 entity 字段时不崩溃"""
        kb.search = MagicMock(return_value=[
            {"id": 0, "distance": 0.5},
        ])

        result = kb.retrieve("test")
        # 不应崩溃，使用默认值
        assert "unknown" in result

    def test_retrieve_passes_top_k_to_search(self, kb):
        """retrieve 将 top_k 参数传递给 search"""
        kb.search = MagicMock(return_value=[])

        kb.retrieve("query", top_k=10)

        kb.search.assert_called_once_with("query", 10)

    def test_retrieve_uses_default_top_k(self, kb):
        """不传 top_k 时使用实例默认值"""
        kb.search = MagicMock(return_value=[])

        kb.retrieve("query")

        kb.search.assert_called_once_with("query", None)


class TestKnowledgeBaseIndexTexts:
    """index_texts 方法测试"""

    @pytest.fixture
    def kb(self):
        """返回一个 mock 了 client 和 embed_model 的 KnowledgeBase 实例"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            top_k=4,
            embedding_model_name="test-model",
            dashscope_api_key="test-key",
        )
        # Mock 内部组件
        kb._client = MagicMock()
        kb._client.has_collection.return_value = True  # Collection 已存在
        kb._embed_model = MagicMock()
        kb._embed_model.embed_documents.return_value = [
            [0.1] * 1024,
            [0.2] * 1024,
        ]
        return kb

    def test_index_texts_returns_status_string(self, kb):
        """索引后返回状态字符串"""
        result = kb.index_texts(
            ["短文本一", "短文本二"],
            sources=["src1", "src2"],
        )

        assert "Indexed" in result
        assert "2 documents" in result
        assert "test_col" in result
        kb._client.insert.assert_called_once()

    def test_index_texts_empty_list(self, kb):
        """空文本列表返回提示"""
        result = kb.index_texts([])

        assert "No texts" in result
        # 不应调用 insert
        kb._client.insert.assert_not_called()

    def test_index_texts_default_sources(self, kb):
        """不传 sources 时默认为 'unknown'"""
        kb._embed_model.embed_documents.return_value = [[0.1] * 1024]

        result = kb.index_texts(["单文本"])

        assert "Indexed" in result
        # 验证 insert 数据中的 source 为 "unknown"
        call_args = kb._client.insert.call_args
        data = call_args[1]["data"]
        assert data[0]["source"] == "unknown"

    def test_index_texts_source_length_mismatch(self, kb):
        """sources 长度与 texts 不匹配时抛异常"""
        with pytest.raises(ValueError, match="sources length"):
            kb.index_texts(["a", "b"], sources=["only_one"])

    def test_index_texts_creates_collection_if_missing(self, kb):
        """Collection 不存在时自动创建"""
        kb._client.has_collection.return_value = False
        kb._embed_model.embed_query.return_value = [0.0] * 1536

        kb.index_texts(["test"], sources=["src"])

        kb._client.create_collection.assert_called_once()


class TestKnowledgeBaseIndexFile:
    """index_file 方法测试"""

    @pytest.fixture
    def kb(self):
        """返回一个 mock 了 client 和 embed_model 的 KnowledgeBase"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            top_k=4,
            embedding_model_name="test-model",
            dashscope_api_key="test-key",
        )
        kb._client = MagicMock()
        kb._client.has_collection.return_value = True
        kb._embed_model = MagicMock()
        kb._embed_model.embed_documents.return_value = [[0.1] * 1024]
        return kb

    def test_index_txt_file(self, kb, tmp_path):
        """应能读取 .txt 文件并索引"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("这是测试内容。", encoding="utf-8")

        result = kb.index_file(str(file_path))

        assert "Indexed" in result
        assert "test" in result  # source_name 默认为文件名
        kb._client.insert.assert_called_once()

    def test_index_md_file(self, kb, tmp_path):
        """.md 文件也按纯文本处理"""
        file_path = tmp_path / "doc.md"
        file_path.write_text("# 标题\n\n段落内容。", encoding="utf-8")

        result = kb.index_file(str(file_path))

        assert "Indexed" in result
        assert "doc" in result

    def test_index_file_uses_custom_source_name(self, kb, tmp_path):
        """可指定自定义 source_name"""
        file_path = tmp_path / "data.txt"
        file_path.write_text("内容", encoding="utf-8")

        result = kb.index_file(str(file_path), source_name="custom_src")

        assert "Indexed" in result
        # 验证 insert 时的 source 字段
        call_args = kb._client.insert.call_args
        data = call_args[1]["data"]
        assert data[0]["source"] == "custom_src"

    def test_index_file_not_found(self, kb):
        """文件不存在时返回错误信息"""
        result = kb.index_file("/nonexistent/path.txt")

        assert "File not found" in result
        kb._client.insert.assert_not_called()

    def test_index_unsupported_format(self, kb, tmp_path):
        """不支持的文件格式返回提示"""
        file_path = tmp_path / "image.png"
        file_path.write_text("fake png", encoding="utf-8")

        result = kb.index_file(str(file_path))

        assert "Unsupported file format" in result
        assert ".png" in result
        kb._client.insert.assert_not_called()


class TestIndexFileFunction:
    """模块级 index_file 便捷函数测试"""

    @patch("src.rag.KnowledgeBase")
    def test_index_file_delegates_to_kb(self, mock_kb_class):
        """验证 index_file 正确委托给 KnowledgeBase.index_file"""
        mock_kb = MagicMock()
        mock_kb.index_file.return_value = "Indexed 5 chunks..."
        mock_kb_class.return_value = mock_kb

        result = index_file("/path/to/doc.pdf", source_name="mydoc")

        assert result == "Indexed 5 chunks..."
        mock_kb.index_file.assert_called_once_with("/path/to/doc.pdf", "mydoc")


class TestSearchMethod:
    """search 方法测试"""

    def test_search_returns_results(self):
        """search 应返回原始结果列表"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            top_k=4,
            embedding_model_name="test-model",
            dashscope_api_key="test-key",
        )
        kb._client = MagicMock()
        kb._client.has_collection.return_value = True
        kb._client.search.return_value = [
            [
                {"id": 0, "distance": 0.9, "entity": {"text": "结果", "source": "s"}},
            ]
        ]
        kb._embed_model = MagicMock()
        kb._embed_model.embed_query.return_value = [0.1] * 1024

        results = kb.search("query")

        assert len(results) == 1
        assert results[0]["entity"]["text"] == "结果"

    def test_search_empty_results(self):
        """空结果时返回空列表"""
        kb = KnowledgeBase(
            milvus_url="http://test:19530",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
            top_k=4,
            embedding_model_name="test-model",
            dashscope_api_key="test-key",
        )
        kb._client = MagicMock()
        kb._client.has_collection.return_value = True
        kb._client.search.return_value = []  # 空
        kb._embed_model = MagicMock()
        kb._embed_model.embed_query.return_value = [0.1] * 1024

        results = kb.search("query")

        assert results == []


class TestRetrieveDocumentsFunction:
    """模块级 retrieve_documents 便捷函数测试"""

    @patch("src.rag.KnowledgeBase")
    def test_retrieve_documents_delegates_to_kb(self, mock_kb_class):
        """验证 retrieve_documents 正确委托给 KnowledgeBase.retrieve"""
        mock_kb = MagicMock()
        mock_kb.retrieve.return_value = "Mocked results"
        mock_kb_class.return_value = mock_kb

        result = retrieve_documents("test query")

        assert result == "Mocked results"
        mock_kb.retrieve.assert_called_once_with("test query", None)

    @patch("src.rag.KnowledgeBase")
    def test_retrieve_documents_passes_top_k(self, mock_kb_class):
        """验证 top_k 参数传递"""
        mock_kb = MagicMock()
        mock_kb.retrieve.return_value = "Mocked"
        mock_kb_class.return_value = mock_kb

        retrieve_documents("query", top_k=10)

        mock_kb.retrieve.assert_called_once_with("query", 10)
