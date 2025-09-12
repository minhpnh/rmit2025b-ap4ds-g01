from flask import Blueprint, render_template

bp_home = Blueprint("home", __name__)

@bp_home.get("/")
def home():
    return render_template("home/homepage.html")

