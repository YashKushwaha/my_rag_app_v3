#from .faiss_store import save_embeddings, load_embeddings  # or dynamically select based on config

from .faiss_store import FAISSVectorStore
from .pinecone_store import PineconeVectorStore

def load_vector_store(config):
    vector_store_to_use = config['vector_store_to_use']
    if vector_store_to_use == 'faiss':
        vector_store = FAISSVectorStore(config['faiss'])
    elif vector_store_to_use == 'pinecone':
        vector_store = PineconeVectorStore(config['pinecone'])
    else:
        raise ValueError(f"Unsupported vectorstore: {vector_store_to_use}")
    return vector_store