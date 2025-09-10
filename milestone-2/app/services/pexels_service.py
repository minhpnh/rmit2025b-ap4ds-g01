import os
from functools import lru_cache
from typing import Optional

import requests


PEXELS_API_URL = "https://api.pexels.com/v1/search"


def _get_api_key() -> Optional[str]:
    # Prefer explicit env var; optionally allow Flask config via env injection
    return os.getenv("PEXELS_API_KEY")


@lru_cache(maxsize=1024)
def image_url_for_title(title: str) -> Optional[str]:
    """Look up a representative image URL for a product title via Pexels.

    Returns a direct image URL (str) or None if unavailable/errors.
    Cached by title to avoid repeat network calls.
    """
    key = _get_api_key()
    if not key or not title:
        return None

    try:
        params = {
            "query": title,
            "per_page": 1,
            "orientation": "landscape",
        }
        headers = {"Authorization": key}
        resp = requests.get(PEXELS_API_URL, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        photos = data.get("photos") or []
        if not photos:
            return None
        src = photos[0].get("src") or {}
        # Prefer 'large' then 'medium' then 'original'
        return src.get("large") or src.get("medium") or src.get("original")
    except Exception:
        return None

