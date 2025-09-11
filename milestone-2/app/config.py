import os

class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    PRODUCTS_CSV = os.getenv("PRODUCTS_CSV", os.path.join("data", "webData.csv"))
    REVIEWS_JSON = os.getenv("REVIEWS_JSON", os.path.join("instance", "reviews.json"))
    # Text processing assets (so Milestone 2 is self-contained)
    STOPWORDS_PATH = os.getenv("STOPWORDS_PATH", os.path.join("data", "stopwords_en.txt"))
    VOCAB_PATH = os.getenv("VOCAB_PATH", os.path.join("data", "vocab.txt"))

class DevConfig(BaseConfig):
    DEBUG = True

class ProdConfig(BaseConfig):
    DEBUG = False
