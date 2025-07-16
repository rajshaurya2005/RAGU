from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from vector_store_manager import VectorStoreManager
from document_processor import DocumentProcessor
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RagPipeline:
    def __init__(self, provider: str, api_key: str, model: str, embedding_model: str = "all-MiniLM-L6-v2",
                temperature: float = 0.7, max_tokens: int = 1000,
                system_message: str = "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain enough information, say so."):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.vector_store = VectorStoreManager(model_name=embedding_model)
        self.document_processor = DocumentProcessor()
        self._initialize_llm()

    def _initialize_llm(self):
        if self.provider == "groq":
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model=self.model_name, api_key=self.api_key, temperature=self.temperature, max_tokens=self.max_tokens)
        elif self.provider == "openai":
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(model=self.model_name, api_key=self.api_key, temperature=self.temperature, max_tokens=self.max_tokens)
        elif self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model=self.model_name, api_key=self.api_key, temperature=self.temperature, max_tokens=self.max_tokens)
        # Add more providers as needed
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def load_documents(self, file_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Load and process documents into the vector store.
        
        Args:
            file_paths (List[str]): List of file paths to process.
            chunk_size (int): Size of text chunks.
            chunk_overlap (int): Overlap between consecutive chunks.
        """
        try:
            all_chunks = []
            for file_path in file_paths:
                documents = self.document_processor.load_document(file_path)
                if documents:
                    chunks = self.document_processor.split_text(documents, chunk_size, chunk_overlap)
                    all_chunks.extend(chunks)
            
            if all_chunks:
                self.vector_store.add_documents(all_chunks)
                logger.info(f"Loaded {len(all_chunks)} chunks into vector store")
            else:
                logger.warning("No documents were loaded")
        
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def load_existing_vector_store(self, index_path: str = "faiss_index") -> None:
        """
        Load an existing vector store from disk.
        
        Args:
            index_path (str): Path to the FAISS index.
        """
        try:
            self.vector_store.load_local(index_path)
            logger.info(f"Loaded vector store from {index_path}")
        except Exception as e:
            logger.error(f"Could not load vector store: {e}")
            raise
    
    def query(self, query: str, k: int = 5) -> Optional[str]:
        """
        Process a query using the RAG pipeline.
        
        Args:
            query (str): User query.
            k (int): Number of document chunks to retrieve.
            
        Returns:
            Optional[str]: Response from the LLM or error message.
        """
        try:
            if not self.vector_store.is_initialized():
                try:
                    self.load_existing_vector_store()
                except:
                    return "⚠️ No documents found. Please upload some files first and wait for them to be processed."
            
            context_docs = self.vector_store.retrieve(query, k)
            if not context_docs:
                return "No relevant documents found for your query."
            
            context = "\n\n".join(doc.page_content for doc in context_docs)
            system_prompt = SystemMessage(
                content="You are a helpful assistant. Answer questions based on the provided context. "
                        "If the context doesn't contain enough information, say so."
            )
            user_prompt = HumanMessage(
                content=f"""Based on the context below, answer the query to the best of your ability.

Context:
{context}

Query: {query}

Answer:"""
            )
            
            messages = [system_prompt, user_prompt]
            result = self.llm.invoke(messages)
            return result.content
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"