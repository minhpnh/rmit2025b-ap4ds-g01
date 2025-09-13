import io
from huggingface_hub import hf_hub_download
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BoWVectorizer:
    def __init__(self, vocab):
        self.vectorizer = CountVectorizer(vocabulary=vocab)

    def transform(self, text_list):
        return self.vectorizer.transform(text_list)


def download_embedding_model(hf_token):
    # Download fasttext 300 dimensions english pretrained embedding model
    filename = "wiki-news-300d-1M-subword.vec"
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


class FastTextVectorizer:
    def __init__(self, idf_dict=None):
        self.idf_dict = idf_dict or {}

    def _load_vectors(self, fname):
        print("Loading FastText vectors... (this may take a while)")
        fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            vec = np.array(tokens[1:], dtype=np.float32)
            data[word] = vec
        print(f"Loaded {len(data)} word vectors of dimension {d}")
        return data, d

    def load_model(self, model_path):
        # Load FastText model
        self.model, self.vector_size = self._load_vectors(model_path)

    def average_vector(self, tokens):
        vecs = [self.model[t] for t in tokens if t in self.model]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def tfidf_weighted_vector(self, tokens):
        vecs = [
            self.model[t] * self.idf_dict.get(t, 1.0) for t in tokens if t in self.model
        ]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)
