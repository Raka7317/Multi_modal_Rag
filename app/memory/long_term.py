"""
Long-term memory = durable, cross-session facts about a user (preferences,
recurring context, prior decisions) that should survive beyond one
conversation thread. Stored in MongoDB, keyed by user_id, separate from the
LangGraph short-term checkpoint state.
"""
from datetime import datetime, timezone
from pymongo import MongoClient
from app.config import settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongodb_uri)
    return _client


def _collection():
    return get_client()[settings.mongodb_db][settings.long_term_collection]


def upsert_fact(user_id: str, key: str, value: str):
    _collection().update_one(
        {"user_id": user_id, "key": key},
        {"$set": {"value": value, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


def get_facts(user_id: str) -> dict[str, str]:
    docs = _collection().find({"user_id": user_id})
    return {d["key"]: d["value"] for d in docs}


def delete_fact(user_id: str, key: str):
    _collection().delete_one({"user_id": user_id, "key": key})
