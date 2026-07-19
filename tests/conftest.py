import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def temp_notes_dir():
  """创建一个临时笔记目录，测试结束后自动清理"""
  with tempfile.TemporaryDirectory() as tmpdir:
      yield tmpdir


@pytest.fixture
def sample_note(temp_notes_dir):
  """在临时目录里预存一篇笔记，返回目录路径"""
  note_content = "# 测试笔记\n\n这是一个测试笔记的内容。"
  filepath = Path(temp_notes_dir) / "测试笔记.md"
  filepath.write_text(note_content, encoding="utf-8")
  return temp_notes_dir


@pytest.fixture
def multiple_notes(temp_notes_dir):
  """在临时目录里预存多篇笔记"""
  notes = {
      "人工智能发展.md": "# AI 发展\n\n近年来 AI 发展迅速。",
      "Python学习笔记.md": "# Python\n\nPython 是一门流行的编程语言。",
      "量子计算科普.md": "# 量子计算\n\n量子计算利用量子比特。",
  }
  for filename, content in notes.items():
      filepath = Path(temp_notes_dir) / filename
      filepath.write_text(content, encoding="utf-8")
  return temp_notes_dir