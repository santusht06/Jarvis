import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Smart Agent Project Maintainer"
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    SQLITE_DB_PATH: str = os.path.join(DATA_DIR, "maintainer.db")
    CHROMA_DB_PATH: str = os.path.join(DATA_DIR, "chroma")
    
    # Scanning & Maintainer limits
    LOCAL_SCAN_PATHS: List[str] = [
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Projects"),
        os.path.expanduser("~/Developer")
    ]
    DAILY_PROJECT_LIMIT: int = 1  # Exactly 1 project per 24 hours
    
    # LLM Settings (Supports Groq, OpenAI, Gemini, Ollama, or fallback)
    LLM_PROVIDER: str = "auto"  # 'groq', 'openai', 'gemini', 'ollama', or 'auto'
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # Safety & Precision Guardrails (99.999% accuracy enforcement)
    MAX_PATCH_CHANGE_PERCENT: float = 25.0  # Max percentage of lines altered
    REQUIRE_PR_CREATION: bool = False  # If True, opens PR instead of direct commit/push
    DEFAULT_BRANCH_PREFIX: str = "ai-readme-maint-"
    GIT_USER_EMAIL: str = "115890693+santusht06@users.noreply.github.com"
    GIT_USER_NAME: str = "Santusht Kotai"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)
