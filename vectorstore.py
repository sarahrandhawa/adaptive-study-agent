"""Pinecone vector store helpers — config from env, shared embeddings."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
DEFAULT_INDEX_METRIC = "cosine"


class VectorStoreConfigError(ValueError):
    """Raised when required vector-store environment variables are missing."""


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise VectorStoreConfigError(f"Missing required environment variable: {name}")
    return value


def get_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL).strip() or EMBEDDING_MODEL


def get_index_name() -> str:
    return _require_env("PINECONE_INDEX_NAME")


def _auto_create_index_enabled() -> bool:
    return os.getenv("PINECONE_AUTO_CREATE_INDEX", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _serverless_spec() -> ServerlessSpec:
    cloud = os.getenv("PINECONE_CLOUD", "aws").strip() or "aws"
    region = os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1"
    return ServerlessSpec(cloud=cloud, region=region)


@lru_cache
def get_pinecone_client() -> Pinecone:
    api_key = _require_env("PINECONE_API_KEY")
    return Pinecone(api_key=api_key)


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    # OPENAI_API_KEY is read by the OpenAI client from the environment.
    _require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=get_embedding_model())


def ensure_index_exists() -> None:
    """Create the configured Pinecone index when auto-create is enabled."""

    if not _auto_create_index_enabled():
        return

    client = get_pinecone_client()
    index_name = get_index_name()
    existing = {index_info["name"] for index_info in client.list_indexes()}

    if index_name in existing:
        return

    client.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric=DEFAULT_INDEX_METRIC,
        spec=_serverless_spec(),
    )

    deadline = time.time() + 60
    while time.time() < deadline:
        if client.describe_index(index_name).status["ready"]:
            return
        time.sleep(1)

    raise TimeoutError(f"Pinecone index '{index_name}' was not ready within 60 seconds")


def get_pinecone_index():
    ensure_index_exists()
    client = get_pinecone_client()
    return client.Index(get_index_name())


@lru_cache
def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore(index=get_pinecone_index(), embedding=get_embeddings())


def _jsonify(value: Any) -> Any:
    """Recursively convert Pinecone SDK objects into JSON-safe Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonify(value.to_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonify(value.model_dump())
    return str(value)


def check_pinecone() -> dict[str, Any]:
    """Return a debug payload confirming Pinecone config and connectivity."""

    result: dict[str, Any] = {
        "ok": False,
        "embedding_model": get_embedding_model(),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "config": {
            "pinecone_api_key_set": bool(os.getenv("PINECONE_API_KEY", "").strip()),
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY", "").strip()),
            "index_name": os.getenv("PINECONE_INDEX_NAME", "").strip() or None,
            "auto_create_index": _auto_create_index_enabled(),
            "cloud": os.getenv("PINECONE_CLOUD", "aws").strip() or "aws",
            "region": os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1",
        },
    }

    try:
        _require_env("PINECONE_API_KEY")
        _require_env("OPENAI_API_KEY")
        index_name = get_index_name()
    except VectorStoreConfigError as exc:
        result["error"] = str(exc)
        return result

    try:
        client = get_pinecone_client()
        indexes = client.list_indexes().names()
        result["indexes"] = indexes

        if index_name not in indexes:
            result["error"] = (
                f"Pinecone index '{index_name}' was not found. "
                "Create it in the Pinecone console or set PINECONE_AUTO_CREATE_INDEX=true."
            )
            return result

        description = _jsonify(client.describe_index(index_name))
        stats = _jsonify(client.Index(index_name).describe_index_stats())

        result.update(
            {
                "ok": True,
                "index": {
                    "name": index_name,
                    "ready": description.get("status", {}).get("ready"),
                    "dimension": description.get("dimension"),
                    "metric": description.get("metric"),
                    "host": description.get("host"),
                },
                "stats": {
                    "total_vector_count": stats.get("total_vector_count"),
                    "namespaces": stats.get("namespaces", {}),
                },
            }
        )
        return result
    except Exception as exc:  # noqa: BLE001 — surface connectivity issues in debug output
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
