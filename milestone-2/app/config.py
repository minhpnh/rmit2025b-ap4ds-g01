import os


class BaseConfig:
    # Base directory of the project
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    PRODUCTS_CSV = os.getenv("PRODUCTS_CSV", os.path.join("data", "webData.csv"))
    REVIEWS_JSON = os.getenv("REVIEWS_JSON", os.path.join("instance", "reviews.json"))
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    DATA_FOLDER = os.path.join(BASE_DIR, "data")


class DevConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False
