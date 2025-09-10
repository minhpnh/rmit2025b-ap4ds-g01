from flask import Blueprint, render_template

bp = Blueprint("reviews", __name__)

@bp.get("/new")
def new_review():
    return render_template("reviews/new.html")
