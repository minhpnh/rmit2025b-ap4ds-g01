import joblib
from enum import Enum

from app.ml_helpers.vectorizer import FastTextVectorizer
from .tokenizer import Tokenizer

tokenizer = Tokenizer()


class VectorType(Enum):
    BOW = 1
    AVERAGE = 2
    TFIDF = 3


def load_model(model_path: str):
    """
    Load a trained model (.joblib)
    """
    # Load the trained sklearn model
    model = joblib.load(model_path)

    return model


def bow_process(review):
    tokens = tokenizer.process_review(review)
    return [" ".join(tokens)]


def vectorize_review(tokens, vectorizer: FastTextVectorizer, vector_type: VectorType):
    if vector_type == VectorType.BOW:
        raise Exception("Please use bow_process() for BoW models")

    elif vector_type == VectorType.AVERAGE:
        # average embedding
        return vectorizer.average_vector(tokens)

    elif vector_type == VectorType.TFIDF:
        return vectorizer.tfidf_weighted_vector(tokens)


def predict_review(
    review, model, vector_type: VectorType, vectorizer: FastTextVectorizer | None
):
    if vector_type == VectorType.BOW:
        processed_text = bow_process(review)
        prediction = model.predict(processed_text)
        return int(prediction[0])
    else:
        if vectorizer is None:
            raise Exception("No vectorizer detected for embedding review")
        tokens = tokenizer.process_review(review)
        vector = vectorize_review(tokens, vectorizer, vector_type)

        prediction = model.predict(vector.reshape(1, -1))
        return int(prediction[0])
