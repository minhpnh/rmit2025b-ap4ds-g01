from flask import Flask
from .config import DevConfig
from .routes.catalog import bp as catalog_bp
from .routes.search import bp as search_bp
from .routes.reviews import bp as reviews_bp
from .services.search_service import highlight

def create_app(config_object=DevConfig):
    app = Flask(__name__, instance_relative_config=True, template_folder="templates", static_folder="static")
    app.config.from_object(config_object)
    app.config.from_pyfile("app.cfg", silent=True)  # optional per-machine overrides

    app.jinja_env.filters["hl"] = highlight
    app.register_blueprint(catalog_bp)
    app.register_blueprint(search_bp, url_prefix="/search")
    app.register_blueprint(reviews_bp, url_prefix="/reviews")
    return app