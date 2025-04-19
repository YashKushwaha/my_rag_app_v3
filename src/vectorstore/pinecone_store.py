from pinecone import Pinecone, ServerlessSpec
import os
import numpy as np

class PineconeVectorStore:
    def __init__(self, pinecone_config):
        api_key = os.environ[pinecone_config['api_key_env']]
        self.index_name = pinecone_config['index_name']
        self.dimension = pinecone_config.get('dimension', 384)  # defaulting for MiniLM
        self.metric = pinecone_config.get('metric', 'cosine')
        self.cloud = pinecone_config.get('cloud', 'aws')
        self.region = pinecone_config.get('region', 'us-east-1')

        self.pc = Pinecone(api_key=api_key)

        # Create index if it doesn't exist
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )

        self.index = self.pc.Index(self.index_name)

    def save(self, embeddings, texts):
        to_upsert = [
            (str(i), embedding, {'text': text})
            for i, (embedding, text) in enumerate(zip(embeddings, texts))
        ]
        self.index.upsert(vectors=to_upsert)

    def load(self):
        # No-op for Pinecone; index is always live
        return self.index

    def retrieve(self, query_embedding, k=3):
        query_embedding = np.array(query_embedding).astype("float32").tolist()
        results = self.index.query(vector=query_embedding, top_k=k, include_metadata=True)
        return [match['metadata']['text'] for match in results['matches']]
