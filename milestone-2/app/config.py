import os

class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    PRODUCTS_CSV = os.getenv("PRODUCTS_CSV", os.path.join("data", "webData.csv"))
    REVIEWS_JSON = os.getenv("REVIEWS_JSON", os.path.join("instance", "reviews.json"))

class DevConfig(BaseConfig):
    DEBUG = True

class ProdConfig(BaseConfig):
    DEBUG = False
