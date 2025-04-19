import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseVectorStore

class FAISSVectorStore(BaseVectorStore):
    def __init__(self, faiss_config):
        base_dir = os.getcwd()
        vectorstore_dir = os.path.join(base_dir, faiss_config['folder_name'])
        os.makedirs(vectorstore_dir, exist_ok=True)
        print('Vector store_DIR -> ', vectorstore_dir)
        self.index_path = os.path.join(vectorstore_dir, faiss_config['index_file'])
        self.text_path = os.path.join(vectorstore_dir, faiss_config['texts_file'])

    def save(self, embeddings, texts):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        print('FAISS index saved to -> ', self.index_path)
        with open(self.text_path, "wb") as f:
            pickle.dump(texts, f)

    def load(self):
        index = faiss.read_index(self.index_path)

        with open(self.text_path, "rb") as f:
            texts = pickle.load(f)

        return index, texts

    def retrieve(self, query_embedding, k=3):
        index, texts = self.load()
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        D, I = index.search(query_embedding, k)
        return [texts[i] for i in I[0]]
