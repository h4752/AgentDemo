import os
from pathlib import Path
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from src.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS, NOTES_PATH

tavily = TavilySearch(
    max_results=TAVILY_MAX_RESULTS,
    tavily_api_key=TAVILY_API_KEY,
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