import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv(override=True)

# Конфигурация Mem0 для Self-Hosted + OpenRouter
MEM0_CONFIG = {
    "history_db_path": "history.db",
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "url": "http://localhost:6333",
            "api_key": os.getenv("QDRANT_API_KEY"),
            "collection_name": "mem0_memories",
            "embedding_model_dims": 384
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "openai_base_url": "https://openrouter.ai/api/v1"
        }
    },
    "embedder": {
        "provider": "huggingface",  # Локальная модель
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }
}
