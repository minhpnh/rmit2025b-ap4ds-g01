from typing import List, Dict, Optional
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
