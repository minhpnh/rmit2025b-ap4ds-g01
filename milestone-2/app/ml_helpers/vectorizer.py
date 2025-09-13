import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BoWVectorizer:
    def __init__(self, vocab):
        self.vectorizer = CountVectorizer(vocabulary=vocab)

    def transform(self, text_list):
        return self.vectorizer.transform(text_list)


# TODO:: Implement FastText vectorizing with gensim

# class FastTextVectorizer:
#     def __init__(self, model_path, idf_dict=None):
#         # Load FastText model
#         self.model = self.load_vectors(model_path)
#         self.vector_size = self.model.get_dimension()
#         self.idf_dict = idf_dict or {}
#
#     def load_vectors(self, fname):
#         fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
#         n, d = map(int, fin.readline().split())
#         data = {}
#         for line in fin:
#             tokens = line.rstrip().split(" ")
#             data[tokens[0]] = map(float, tokens[1:])
#         return data
#
#     def _get_vector(self, token, weight=1.0):
#         vec = self.model.get_word_vector(token)  # always returns a vector
#         return vec * weight
#
#     def average_vector(self, tokens):
#         """Unweighted embeddings"""
#         if not tokens:
#             return np.zeros(self.vector_size)
#         vectors = [self._get_vector(t) for t in tokens]
#         return np.mean(vectors, axis=0)
#
#     def tfidf_weighted_vector(self, tokens):
#         """TF-IDF weighted embeddings"""
#         if not tokens:
#             return np.zeros(self.vector_size)
#         vectors = [self._get_vector(t, self.idf_dict.get(t, 1.0)) for t in tokens]
#         return np.mean(vectors, axis=0)
