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