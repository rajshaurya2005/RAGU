from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import shutil
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading, splitting, and temporary file management."""
    
    def __init__(self):
        """Initialize document processor with text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Default, overridden by user settings
            chunk_overlap=50
        )
        self._file_loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader
        }
    
    def save_uploaded_files(self, uploaded_files: List) -> List[str]:
        """
        Save uploaded files to a temporary directory.
        
        Args:
            uploaded_files (List): List of uploaded file objects.
            
        Returns:
            List[str]: Paths to saved files.
        """
        try:
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
                logger.info(f"Saved file: {file_path}")
            
            return file_paths
        
        except Exception as e:
            logger.error(f"Error saving files: {e}")
            raise
    
    def cleanup_temp_files(self, temp_dir: str) -> None:
        """
        Clean up temporary directory and its contents.
        
        Args:
            temp_dir (str): Path to temporary directory.
        """
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory {temp_dir}: {e}")
    
    def load_document(self, file_path: str) -> Optional[List]:
        """
        Load a document from a file path.
        
        Args:
            file_path (str): Path to the document file.
            
        Returns:
            Optional[List]: Loaded documents or None if unsupported or error.
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            loader_class = self._file_loaders.get(file_ext)
            if not loader_class:
                logger.error(f"Unsupported file type: {file_ext}")
                return None
            
            loader = loader_class(file_path)
            documents = loader.load()
            logger.info(f"Loaded document: {file_path}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None
    
    def split_text(self, documents: List, chunk_size: int, chunk_overlap: int) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents (List): List of documents to split.
            chunk_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between chunks.
            
        Returns:
            List: List of document chunks.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []