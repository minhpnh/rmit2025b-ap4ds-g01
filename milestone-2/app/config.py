import os


class BaseConfig:
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    PRODUCTS_CSV = os.getenv("PRODUCTS_CSV", os.path.join("data", "webData.csv"))
    REVIEWS_JSON = os.getenv("REVIEWS_JSON", os.path.join("instance", "reviews.json"))
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


class DevConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False
