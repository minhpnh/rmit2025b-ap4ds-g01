import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from flask import current_app


class ReviewRepo:
    def __init__(self, json_path: Optional[str] = None):
        # Default to configured path: instance/reviews.json
        self.json_path = json_path or current_app.config["REVIEWS_JSON"]
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        self._reviews = self._load()
        self._index = {r["id"]: r for r in self._reviews}

    def _load(self) -> List[Dict]:
        if not os.path.exists(self.json_path):
            return []
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception:
            return []

    def _save(self) -> None:
        tmp_path = self.json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._reviews, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.json_path)

    def list(self) -> List[Dict]:
        return list(self._reviews)

    def by_item(self, item_id: str) -> List[Dict]:
        return [r for r in self._reviews if str(r.get("item_id")) == str(item_id)]

    def get(self, review_id: str) -> Optional[Dict]:
        return self._index.get(str(review_id))

    def _next_id(self) -> str:
        # Simple incrementing string id based on length; sufficient for assignment
        return str(len(self._reviews) + 1)

    def add(self, review: Dict) -> Dict:
        if "id" not in review or not review["id"]:
            review["id"] = self._next_id()
        review.setdefault("created_at", datetime.utcnow().isoformat() + "Z")

        self._reviews.append(review)
        self._index[review["id"]] = review
        self._save()
        return review

