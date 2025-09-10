from flask import Blueprint, render_template, request, redirect, url_for, abort
from ..models.product_repo import ProductRepo
from ..models.review_repo import ReviewRepo
from ..services.ml_service import predict_recommendation

bp = Blueprint("reviews", __name__)

@bp.get("/new")
def new_review():
    item_id = request.args.get("item_id", "").strip()
    if not item_id:
        return abort(400)
    p = ProductRepo().get(item_id)
    if not p:
        return abort(404)
    return render_template("reviews/new.html", p=p)


@bp.post("/new")
def predict_then_confirm():
    item_id = request.form.get("item_id", "").strip()
    p = ProductRepo().get(item_id)
    if not p:
        return abort(404)

    title = (request.form.get("title") or "").strip()
    body = (request.form.get("body") or "").strip()
    rating = int(request.form.get("rating") or 0)
    suggested = predict_recommendation(title, body)

    return render_template(
        "reviews/confirm.html",
        p=p,
        title=title,
        body=body,
        rating=rating,
        suggested=suggested,
    )


@bp.post("/create")
def create_review():
    item_id = request.form.get("item_id", "").strip()
    p = ProductRepo().get(item_id)
    if not p:
        return abort(404)

    title = (request.form.get("title") or "").strip()
    body = (request.form.get("body") or "").strip()
    rating = int(request.form.get("rating") or 0)
    suggested = int(request.form.get("suggested") or 0)
    recommended = int(request.form.get("recommended") or suggested)

    repo = ReviewRepo()
    review = repo.add({
        "item_id": p["id"],
        "title": title,
        "body": body,
        "rating": rating,
        "predicted": suggested,
        "recommended": recommended,
    })

    return redirect(url_for("reviews.show", id=review["id"]))


@bp.get("/<id>")
def show(id):
    repo = ReviewRepo()
    r = repo.get(id)
    if not r:
        return abort(404)
    p = ProductRepo().get(r.get("item_id"))
    return render_template("reviews/show.html", r=r, p=p)
