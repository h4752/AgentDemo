import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-pro")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
NOTES_PATH = os.getenv("NOTES_PATH", "./notes")
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "./resources/checkpoint.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./resources/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
MILVUS_URL = os.getenv("MILVUS_URL", "http://localhost:19530")

# RAG / Knowledge Base settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")