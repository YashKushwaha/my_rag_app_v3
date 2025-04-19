class BaseVectorStore:
    def save(self, embeddings, texts):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def retrieve(self, query_embedding, k):
        raise NotImplementedError