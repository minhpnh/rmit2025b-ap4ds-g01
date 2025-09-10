from flask import Blueprint, render_template, abort
from ..models.product_repo import ProductRepo

bp = Blueprint("catalog", __name__)

@bp.get("/")
def index():
    repo = ProductRepo()
    products = repo.first_n(24)
    return render_template("catalog/index.html", products=products)

@bp.get("/item/<id>")
def item(id):
    repo = ProductRepo()
    p = repo.get(id)
    if not p:
        return abort(404)
    return render_template("catalog/item_detail.html", p=p)
