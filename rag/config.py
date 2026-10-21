"""Every setting in one place. Swap the LLM for a cloud model by changing the three LLM_* env vars."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
INDEX_DIR = ROOT / "index"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge wants this on queries only
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")   # Ollama's OpenAI-compatible API
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")                        # Ollama ignores it, cloud APIs don't
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
