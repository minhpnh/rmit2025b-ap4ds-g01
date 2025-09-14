from flask import Flask
from dotenv import load_dotenv
from .config import DevConfig
from .routes.catalog import bp as catalog_bp
from .routes.search import bp as search_bp
from .routes.reviews import bp as reviews_bp
from .routes.home import bp_home  # Import the home blueprint
from .ml_helpers.classification_model import load_model
from .ml_helpers.vectorizer import download_embedding_model, load_idf_weights
from .services.pexels_service import image_url_for_title
from .services.search_service import highlight
from .services.supabase_service import connect_to_db
from huggingface_hub import hf_hub_download
from pathlib import Path


def download_embedding_model(hf_token):
    # Download fasttext 300 dimensions english pretrained embedding model
    filename = "fasttext_thin.kv.vectors_ngrams.npy"
    data_dir = Path("./data")  # relative to root
    # data_dir.mkdir(exist_ok=True)  # create folder if it doesn't exist
    file_path = data_dir / filename

    if not file_path.exists():
        token = hf_token
        repo_id = "tsun2610/FastText-english-text-vectors"

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=data_dir,  # only the directory, not the full file path
        )

        print("Downloaded fasttext model sucessfully")
    else:
        print("Vector file already exists.")


def create_app(config_object=DevConfig):
    # Load variables from .env for local/dev runs
    load_dotenv()
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(config_object)
    app.config.from_pyfile("app.cfg", silent=True)  # optional per-machine overrides

    app.jinja_env.filters["hl"] = highlight
    # Image helper from Pexels
    try:
        app.jinja_env.filters["img"] = image_url_for_title
    except Exception:
        # If optional dependency/network errors occur at import time, skip filter registration
        pass
    app.register_blueprint(catalog_bp)
    app.register_blueprint(search_bp, url_prefix="/search")
    app.register_blueprint(reviews_bp, url_prefix="/reviews")
    app.register_blueprint(bp_home)  # Register the home blueprint

    # ML Helpers
    download_embedding_model(app.config["HUGGINGFACE_TOKEN"])
    setattr(
        app,
        "idf_dict",
        load_idf_weights("./data/trained_models/idf_weights.json"),
    )
    setattr(
        app,
        "bow_logreg_model",
        load_model("./data/trained_models/bow_title+text_logreg_bal.joblib"),
    )
    setattr(
        app,
        "bow_nb_model",
        load_model("./data/trained_models/bow_title+text_nb.joblib"),
    )
    setattr(
        app,
        "emb_logreg_bal_unweighted",
        load_model("./data/trained_models/emb_logreg_bal_unweighted.joblib"),
    )
    setattr(
        app,
        "emb_logreg_bal_weighted",
        load_model("./data/trained_models/emb_logreg_bal_weighted.joblib"),
    )

    # Create supabase Client as a Flask app attribute
    setattr(app, "supabase", connect_to_db(app.config["SUPABASE_KEY"]))

    download_embedding_model(app.config)

    return app
