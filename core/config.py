"""Central application settings for wiki-processor.

Composes the per-domain config loaders (LLM, embeddings) that already
follow the dataclass + load-from-env idiom. Cached once per process via
get_settings(); call get_settings.cache_clear() in tests to re-read env.

PROCESSOR_API_KEY is deliberately NOT part of Settings: it is read at
request time in api/dependencies.py so tests can toggle it per test.
"""

from dataclasses import dataclass
from functools import lru_cache

from services.embeddings import EmbeddingConfig, load_embedding_env
from services.llm import LLMConfig, load_from_env


@dataclass(frozen=True)
class Settings:
    llm: LLMConfig
    embeddings: EmbeddingConfig


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(llm=load_from_env(), embeddings=load_embedding_env())
