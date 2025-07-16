from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages in-memory vector store operations using FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store manager.
        
        Args:
            model_name (str): HuggingFace embedding model name.
        """
        self.db = None
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Initialized vector store with embedding model: {model_name}")
    
    def is_initialized(self) -> bool:
        """
        Check if the vector store is initialized.
        
        Returns:
            bool: True if initialized, False otherwise.
        """
        return self.db is not None
    
    def add_documents(self, chunks: List[Any]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks (List[Any]): List of document chunks to add.
        """
        if not chunks:
            logger.warning("No document chunks to add")
            return
        
        try:
            if self.db is None:
                self.db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
                logger.info(f"Created new vector store with {len(chunks)} chunks")
            else:
                self.db.add_documents(documents=chunks)
                logger.info(f"Added {len(chunks)} chunks to existing vector store")
        
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def retrieve(self, query: str, k: int = 5) -> List[Any]:
        """
        Retrieve similar documents from the vector store.
        
        Args:
            query (str): Query string.
            k (int): Number of documents to retrieve.
            
        Returns:
            List[Any]: List of similar documents.
        """
        if not self.is_initialized():
            logger.error("Vector store not initialized")
            raise ValueError("Vector store has not been initialized. Please add documents first.")
        
        try:
            similar_documents = self.db.similarity_search(query=query, k=k)
            logger.info(f"Retrieved {len(similar_documents)} documents for query")
            return similar_documents
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def load_local(self, index_path: str = "faiss_index") -> None:
        """
        Load an existing vector store from disk.
        
        Args:
            index_path (str): Path to the FAISS index.
        """
        try:
            self.db = FAISS.load_local(index_path, embeddings=self.embeddings)
            logger.info(f"Loaded vector store from {index_path}")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise