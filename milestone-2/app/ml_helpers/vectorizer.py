from huggingface_hub import hf_hub_download
from pathlib import Path
import numpy as np
import json


def download_embedding_model(hf_token):
    # Download custom fasttext pretrained embedding model
    filename = "fasttext_thin.kv.vectors_ngrams.npy"
    data_dir = Path("./app/ml_models")  # store under app assets
    data_dir.mkdir(parents=True, exist_ok=True)
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


def load_idf_weights(file_path: str | Path) -> dict[str, float]:
    """
    Load IDF weights from a JSON file into a dictionary.

    Args:
        file_path (str | Path): Path to the idf_weights.json file.

    Returns:
        dict[str, float]: A dictionary mapping terms to their IDF values.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"IDF weights file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        idf_dict = json.load(f)

    # Ensure values are floats (sometimes JSON saves them as strings)
    return {term: float(value) for term, value in idf_dict.items()}


class FastTextVectorizer:
    def __init__(self, idf_dict=None):
        self.idf_dict = idf_dict or {}
        self.model = None
        self.vector_size = None
        self.vocab = None  # Add vocabulary mapping

    def _load_vectors(self, embeddings_path, vocab_path=None):
        """Load the embeddings and vocabulary mapping"""
        # Load the embeddings
        embeddings = np.load(embeddings_path)  # shape: (vocab_size, embedding_dim)
        vector_size = embeddings.shape[1]

        if vocab_path:
            vocab = {}
            with open(vocab_path, "r", encoding="utf-8") as f:
                for line in f:
                    word, idx = line.strip().split(":")
                    vocab[word] = int(idx)
        else:
            vocab = {}

        return embeddings, vector_size, vocab

    def load_model(self, embeddings_path, vocab_path=None):
        """Load FastText embeddings and vocabulary"""
        self.model, self.vector_size, self.vocab = self._load_vectors(
            embeddings_path, vocab_path
        )

    def get_vector(self, token):
        """Get vector for a single token"""

        if self.model is None or self.vocab is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.vector_size is None:
            raise ValueError(
                "Vector size is not set. Make sure to call load_model() first."
            )

        if token in self.vocab:
            idx = self.vocab[token]
            return self.model[idx]
        else:
            # Return zero vector for unknown tokens
            return np.zeros(self.vector_size)

    def average_vector(self, tokens):
        """Compute average vector for a list of tokens"""
        if self.model is None or self.vocab is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.vector_size is None:
            raise ValueError(
                "Vector size is not set. Make sure to call load_model() first."
            )

        vecs = []
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                vecs.append(self.model[idx])

        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def tfidf_weighted_vector(self, tokens):
        """Compute TF-IDF weighted average vector for a list of tokens"""

        if self.model is None or self.vocab is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.vector_size is None:
            raise ValueError(
                "Vector size is not set. Make sure to call load_model() first."
            )

        vecs = []
        for token in tokens:
            if token in self.vocab:
                idx = self.vocab[token]
                idf_weight = self.idf_dict.get(token, 1.0)
                weighted_vec = self.model[idx] * idf_weight
                vecs.append(weighted_vec)

        return np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size)

    def transform(self, token_lists, method="average"):
        """
        Transform multiple documents (lists of tokens) into embeddings

        Args:
            token_lists: List of token lists, e.g. [['word1', 'word2'], ['word3', 'word4']]
            method: 'average' or 'tfidf'

        Returns:
            np.array: Array of embeddings, shape (n_documents, vector_size)
        """
        embeddings = []

        for tokens in token_lists:
            if method == "tfidf":
                embedding = self.tfidf_weighted_vector(tokens)
            else:
                embedding = self.average_vector(tokens)
            embeddings.append(embedding)

        return np.array(embeddings)


# # Usage example:
# if __name__ == "__main__":
#     # Initialize vectorizer
#     vectorizer = FastTextVectorizer()
#
#     # Load your model and vocabulary
#     vectorizer.load_model(
#         embeddings_path="./data/fasttext_thin.kv.vectors_ngrams.npy",
#         vocab_path="./data/vocabulary.txt"  # Your txt file with token:index format
#     )
#
#     # Example usage
#     token_lists = [
#         ['hello', 'world'],
#         ['machine', 'learning', 'is', 'fun']
#     ]
#
#     # Get embeddings
#     embeddings = vectorizer.transform(token_lists, method='average')
#     print(f"Embeddings shape: {embeddings.shape}")
#
#     # Or get single vectors
#     single_embedding = vectorizer.average_vector(['hello', 'world'])
#     print(f"Single embedding shape: {single_embedding.shape}")
