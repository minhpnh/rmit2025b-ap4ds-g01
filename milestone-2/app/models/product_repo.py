from typing import List, Dict, Optional, Tuple
from flask import current_app


class ProductRepo:
    def list(self) -> List[Dict]:
        res = current_app.supabase.table("products").select("*").execute()  # type: ignore
        return res.data

    def first_n(self, n: int = 24) -> List[Dict]:
        res = current_app.supabase.table("products").select("*").limit(n).execute()  # type: ignore
        return res.data

    def get(self, pid: str) -> Optional[Dict]:
        res = (
            current_app.supabase.table("products")  # type: ignore
            .select("*")
            .eq("id", pid)
            .single()
            .execute()
        )
        return res.data if res.data else None

    def count_recommended_reviews(self, product_id: str) -> int:
        res = (
            current_app.supabase.table("reviews")  # type: ignore
            .select("id", count="exact")  # count rows explicitly
            .eq("item_id", product_id)
            .eq("recommended", 1)
            .execute()
        )
        return res.count or 0

    def filter(self, division: str | None = None, department: str | None = None, class_: str | None = None) -> List[Dict]:
        q = current_app.supabase.table("products").select("*")  # type: ignore
        if division:
            q = q.eq("division", division)
        if department:
            q = q.eq("department", department)
        if class_:
            q = q.eq("class", class_)
        res = q.execute()
        return res.data

    def filter_ids(self, division: str | None = None, department: str | None = None, class_: str | None = None) -> List[str]:
        q = current_app.supabase.table("products").select("id")  # type: ignore
        if division:
            q = q.eq("division", division)
        if department:
            q = q.eq("department", department)
        if class_:
            q = q.eq("class", class_)
        res = q.execute()
        return [str(row["id"]) for row in (res.data or [])]

    def get_many(self, ids: List[str]) -> List[Dict]:
        if not ids:
            return []
        res = (
            current_app.supabase.table("products")  # type: ignore
            .select("*")
            .in_("id", ids)
            .execute()
        )
        return res.data or []

    def paginated(self, page: int = 1, per_page: int = 24,
                  division: str | None = None, department: str | None = None, class_: str | None = None) -> Tuple[List[Dict], int]:
        start = (page - 1) * per_page
        end = start + per_page - 1
        q = current_app.supabase.table("products").select("*", count="exact")  # type: ignore
        if division:
            q = q.eq("division", division)
        if department:
            q = q.eq("department", department)
        if class_:
            q = q.eq("class", class_)
        res = q.range(start, end).execute()
        total = res.count or 0
        return res.data or [], total
