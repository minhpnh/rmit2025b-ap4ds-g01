from collections import Counter
from flask import current_app

from app.ml_helpers.vectorizer import FastTextVectorizer
from ..ml_helpers.classification_model import VectorType, predict_review


def predict_recommendation(title: str, body: str) -> int:
    full_review = title + " " + body

    predictions = []

    # --- Model 1: BoW Logistic Regression ---
    bow_logreg_model = current_app.bow_logreg_model  # type: ignore
    bow_logreg_label = predict_review(
        full_review, bow_logreg_model, vector_type=VectorType.BOW, vectorizer=None
    )
    predictions.append(bow_logreg_label)

    # --- Model 2: BoW Naive Bayes ---
    bow_nb_model = current_app.bow_nb_model  # type: ignore
    bow_nb_label = predict_review(
        full_review, bow_nb_model, vector_type=VectorType.BOW, vectorizer=None
    )
    predictions.append(bow_nb_label)

    ft_model = FastTextVectorizer(current_app.idf_dict)  # type: ignore
    ft_model.load_model(
        "./data/fasttext_thin.kv.vectors_ngrams.npy", "./data/vocab.txt"
    )

    # --- Model 3:  ---
    emb_logreg_bal_unweighted_model = current_app.emb_logreg_bal_unweighted  # type: ignore
    emb_logreg_bal_unweighted_label = predict_review(
        full_review,
        model=emb_logreg_bal_unweighted_model,
        vector_type=VectorType.AVERAGE,
        vectorizer=ft_model,
    )
    predictions.append(emb_logreg_bal_unweighted_label)

    # --- Model 4:  ---
    emb_logreg_bal_weighted_model = current_app.emb_logreg_bal_weighted  # type: ignore
    emb_logreg_bal_weighted_label = predict_review(
        full_review,
        model=emb_logreg_bal_weighted_model,
        vector_type=VectorType.TFIDF,
        vectorizer=ft_model,
    )
    predictions.append(emb_logreg_bal_weighted_label)

    # --- Fusion: majority vote ---
    counts = Counter(predictions)
    final_label = counts.most_common(1)[0][0]

    return final_label
