import csv
from typing import List, Dict
from flask import current_app

class ProductRepo:
    def __init__(self, csv_path: str | None = None):
        self.csv_path = csv_path or current_app.config["PRODUCTS_CSV"]
        self._products = self._load_products()
        self._product_map = {p["id"]: p for p in self._products}

    def _load_products(self) -> List[Dict]:
        products = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                products.append({
                    "id": (row.get("id") or "").strip(),
                    "division": (row.get("division") or "").strip(),
                    "department": (row.get("department") or "").strip(),
                    "class": (row.get("class") or "").strip(),
                    "title": (row.get("title") or "").strip(),
                    "description": (row.get("description") or "").strip(),
                })
        return products

    def list(self): return self._products
    def first_n(self, n=24): return self._products[:n]
    def get(self, pid: str): return self._product_map.get(str(pid))
