from flask import Flask
from dotenv import load_dotenv
from .config import DevConfig
from .routes.catalog import bp as catalog_bp
from .routes.search import bp as search_bp
from .routes.reviews import bp as reviews_bp
from .services.search_service import highlight

def create_app(config_object=DevConfig):
    # Load variables from .env for local/dev runs
    load_dotenv()
    app = Flask(__name__, instance_relative_config=True, template_folder="templates", static_folder="static")
    app.config.from_object(config_object)
    app.config.from_pyfile("app.cfg", silent=True)  # optional per-machine overrides

    app.jinja_env.filters["hl"] = highlight
    # Image helper from Pexels
    try:
        from .services.pexels_service import image_url_for_title
        app.jinja_env.filters["img"] = image_url_for_title
    except Exception:
        # If optional dependency/network errors occur at import time, skip filter registration
        pass
    app.register_blueprint(catalog_bp)
    app.register_blueprint(search_bp, url_prefix="/search")
    app.register_blueprint(reviews_bp, url_prefix="/reviews")
    return app
