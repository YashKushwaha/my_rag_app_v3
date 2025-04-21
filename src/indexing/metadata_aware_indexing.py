import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.embedder import load_text_embedder
from src.vectorstore import load_vector_store
from src.logger_config import setup_logger
from .base_indexer import BaseIndexer

logger = setup_logger(__name__)
class MetadataAwareTextIndexer(BaseIndexer):
    def __init__(self, config):
        self.config = config
        self.docs = None

    def ingest(self):
        data_path = os.path.join(os.getcwd(), self.config['documents']['source_dir'])
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory not found: {data_path}")

        # Load with metadata (like file path)
        loader = DirectoryLoader(data_path, glob="**/*.txt", show_progress=True, use_multithreading=True)
        documents = loader.load()

        for doc in documents:
            doc.metadata['source_file'] = doc.metadata.get('source', 'unknown')

        chunk_size = self.config['chunking_strategy']['chunk_size']
        chunk_overlap = self.config['chunking_strategy']['chunk_overlap']
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(documents)

        self.docs = split_docs
        logger.info(f"Ingested and split {len(split_docs)} document chunks with metadata.")
        return split_docs

    def index(self):
        if self.docs is None:
            logger.debug("No preloaded docs found, ingesting now...")
            self.ingest()

        embedder = load_text_embedder(self.config['embedding'])
        texts = [doc.page_content for doc in self.docs]
        metadatas = [doc.metadata for doc in self.docs]
        embeddings = embedder.embed(texts)

        vector_store = load_vector_store(self.config['vector_store'])
        vector_store.save(embeddings, texts, metadatas)

        logger.info(f"Saved {len(embeddings)} metadata-aware embeddings to vector store.")
