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

    def save(self, embeddings, texts, metadatas=None, ids=None):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        faiss.write_index(index, self.index_path)
        print('FAISS index saved to -> ', self.index_path)

        data_bundle = {
            "texts": texts,
            "metadatas": metadatas if metadatas else [None] * len(texts),
            "ids": ids if ids else list(range(len(texts))),
        }

        with open(self.text_path, "wb") as f:
            pickle.dump(data_bundle, f)

    def load(self):
        index = faiss.read_index(self.index_path)

        with open(self.text_path, "rb") as f:
            data_bundle  = pickle.load(f)

        return index, data_bundle 

def retrieve(self, query_embedding, k=3):
    index, data_bundle = self.load()
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

    D, I = index.search(query_embedding, k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({
            "text": data_bundle["texts"][idx],
            "metadata": data_bundle["metadatas"][idx],
            "id": data_bundle["ids"][idx],
            "score": float(dist)  # Lower is better in L2
        })
    return results

