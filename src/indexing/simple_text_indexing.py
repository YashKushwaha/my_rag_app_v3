import os
import logging
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base_indexer import BaseIndexer

#from rag_indexing.utils.embedding_utils import load_embedder
#from rag_indexing.utils.vector_store_utils import load_vector_store
logger = logging.getLogger(__name__)

class SimpleTextIndexer(BaseIndexer):
    def __init__(self, config):
        pass
    def ingest(self):
        data_path = os.path.join(os.getcwd(), self.config['documents']['source_dir'])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory not found: {data_path}")

        loader = DirectoryLoader(data_path, glob='**/*.txt')
        documents = loader.load()

        chunk_size = self.config.get('chunk_size', 500)
        chunk_overlap = self.config.get('chunk_overlap', 100)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(documents)
        logger.info(f"Ingested and split {len(split_docs)} document chunks.")
        return split_docs

    def index(self):
        docs = self.ingest()
        texts = [doc.page_content for doc in docs]

        embedder_model = load_embedder(self.config['embedding'])
        embeddings = embedder_model.embed(texts)

        vector_store = load_vector_store(self.config['vector_store'])
        vector_store.save(embeddings, texts)

        logger.info(f"Saved {len(embeddings)} embeddings to vector store.")
