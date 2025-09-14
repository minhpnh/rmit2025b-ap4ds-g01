from flask import Blueprint, request, redirect, url_for

bp = Blueprint("search", __name__)

@bp.get("/", endpoint="index")   # endpoint will be `search.index`
def index():
    # Normalize search to the catalog route which now handles
    # filtering, sorting, and dynamic search.
    params = {k: v for k, v in request.args.items()}
    return redirect(url_for("catalog.index", **params))
