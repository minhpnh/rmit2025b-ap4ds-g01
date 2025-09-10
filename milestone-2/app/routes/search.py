from flask import Blueprint, render_template, request
from ..models.product_repo import ProductRepo
from ..services.search_service import tokenize, match_query

bp = Blueprint("search", __name__)

@bp.get("/", endpoint="index")   # endpoint will be `search.index`
def index():
    q = request.args.get("q", "").strip()
    repo = ProductRepo()

    if not q:
        products = repo.first_n(24)
        return render_template("catalog/index.html", products=products, q=q, total=len(repo.list()))

    q_tokens = tokenize(q)
    results = [p for p in repo.list() if match_query(p, q_tokens)]
    return render_template("catalog/index.html", products=results, q=q, total=len(results))
