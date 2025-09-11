from typing import List, Dict, Optional
from flask import current_app


class ReviewRepo:
    def list(self) -> List[Dict]:
        res = current_app.supabase.table("reviews").select("*").execute()  # type: ignore
        return res.data

    def by_item(self, item_id: str) -> List[Dict]:
        res = (
            current_app.supabase.table("reviews")  # type: ignore
            .select("*")
            .eq("item_id", item_id)
            .execute()
        )  # type: ignore
        return res.data

    def get(self, review_id: str) -> Optional[Dict]:
        res = (
            current_app.supabase.table("reviews")  # type: ignore
            .select("*")
            .eq("id", review_id)
            .single()
            .execute()
        )
        return res.data if res.data else None

    def add(self, review: Dict) -> Dict:
        res = current_app.supabase.table("reviews").insert(review).execute()  # type: ignore
        return res.data[0]
