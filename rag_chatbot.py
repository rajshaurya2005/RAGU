import streamlit as st
import logging
import uuid
from pathlib import Path
from rag_core import RagPipeline
from document_processor import DocumentProcessor
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RagChatbot:
    """Main application class for the RAG Chatbot using Streamlit."""
    
    def __init__(self):
        self._configure_page()
        self._initialize_session_state()
        self.document_processor = DocumentProcessor()
        
    def _configure_page(self) -> None:
        """Configure Streamlit page settings and custom CSS."""
        st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
        st.markdown(
            """
            <style>
                .main-header { text-align: center; padding: 1rem 0; }
                .chat-container { max-height: 500px; overflow-y: auto; }
                .file-info { background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.5rem; margin: 0.5rem 0; }
            </style>
            """,
            unsafe_allow_html=True
        )
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = []
        if "rag_pipeline" not in st.session_state:
            st.session_state.rag_pipeline = None
    
    def _render_header(self) -> None:
        """Render the main header."""
        st.markdown(
            '<div class="main-header"><h1>💬 RAG Chatbot</h1><p>Ask questions about your uploaded documents!</p></div>',
            unsafe_allow_html=True
        )
    
    def _render_sidebar(self) -> dict:
        """Render sidebar with settings and file uploader."""
        st.sidebar.title("🔧 Settings")

        provider = st.sidebar.selectbox(
            "Provider",
            ["groq", "openai", "anthropic", "azure_openai", "azure_ai", "google_vertexai", "google_genai", "bedrock", "bedrock_converse", "cohere", "fireworks", "together", "mistralai", "huggingface", "ollama", "google_anthropic_vertex", "deepseek", "ibm", "nvidia", "xai", "perplexity"]
        )
        temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
        max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, value=1000)
        system_message = st.sidebar.text_area(
            "System Message",
            value="You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain enough information, say so."
        )
   
        # API Key input
        api_key = st.sidebar.text_input(
            "API Key", type="password", key="api_key",
            help="Get your API key from https://console.groq.com/keys"
        )
        
        # Model selection
        model = st.sidebar.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile", "qwen-qwq-32b", "qwen/qwen3-32b",
                "deepseek-r1-distill-llama-70b", "gemma2-9b-it", "compound-beta",
                "compound-beta-mini", "llama-3.1-8b-instant", "llama3-70b-8192",
                "llama3-8b-8192", "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-12b",
                "meta-llama/llama-prompt-guard-2-22m", "meta-llama/llama-prompt-guard-2-86m"
            ],
            key="model"
        )
        
        # Embedding model input
        embedding_model = st.sidebar.text_input(
            "HuggingFace Embedding Model",
            value="all-MiniLM-L6-v2",
            help="Enter the name of the HuggingFace embedding model"
        )
        
        # File uploader
        uploaded_files = st.sidebar.file_uploader(
            "Upload files", accept_multiple_files=True,
            type=["pdf", "txt", "csv"], help="Upload PDF, TXT, or CSV files"
        )
        
        # Chunk settings
        st.sidebar.subheader("📄 Chunk and Retriever Settings")
        chunk_size = st.sidebar.slider(
            "Chunk Size", min_value=100, max_value=2000, value=500, step=100,
            help="Size of text chunks for processing"
        )
        chunk_overlap = st.sidebar.slider(
            "Chunk Overlap", min_value=0, max_value=500, value=50, step=10,
            help="Overlap between consecutive chunks"
        )
        k_retrievals = st.sidebar.slider(
            "Number of Retrievals", min_value=1, max_value=30, value=10, step=1,
            help="Number of document chunks to retrieve"
        )
        
        # Reset button
        if st.sidebar.button("🔄 Reset Session"):
            st.session_state.clear()
            st.rerun()
        
        return {
            "api_key": api_key,
            "model": model,
            "embedding_model": embedding_model,
            "uploaded_files": uploaded_files,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k_retrievals": k_retrievals,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_message": system_message
        }
    
    def _display_file_info(self, uploaded_files: List) -> None:
        """Display information about uploaded files."""
        if uploaded_files:
            st.sidebar.success(f"📁 {len(uploaded_files)} file(s) uploaded")
            with st.sidebar.expander("View uploaded files"):
                for file in uploaded_files:
                    file_size = len(file.getbuffer()) / 1024  # Size in KB
                    st.write(f"📄 {file.name} ({file_size:.1f} KB)")
    
    def _check_file_changes(self, uploaded_files: List) -> bool:
        """Check if uploaded files have changed."""
        current_files = [f.name for f in uploaded_files]
        if st.session_state.processed_files != current_files:
            st.session_state.processed_files = []
            st.sidebar.info("New files detected. They will be processed with your next query.")
            return True
        return False
    
    def _render_main_content(self, uploaded_files: List) -> None:
        """Render the main content area with chat interface and status."""
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 💭 Chat with your documents")
        with col2:
            if uploaded_files and st.session_state.processed_files:
                st.success("✅ Ready")
            elif uploaded_files:
                st.warning("⏳ Processing...")
            else:
                st.error("📄 No files")
        
        # Display chat history
        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
    def _process_query(self, query: str, settings: dict) -> Optional[str]:
        """Process user query using the RAG pipeline."""
        try:
            if not settings["api_key"]:
                raise ValueError("Please provide a valid Groq API key.")
            if not settings["uploaded_files"]:
                raise ValueError("Please upload at least one document.")
            
            st.session_state.rag_pipeline = RagPipeline(
                provider=settings["provider"],
                api_key=settings["api_key"],
                model=settings["model"],
                embedding_model=settings["embedding_model"],
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                system_message=settings["system_message"]
            )
            
            # Process new files if needed
            if settings["uploaded_files"] and not st.session_state.processed_files:
                with st.spinner("Processing uploaded documents..."):
                    file_paths = self.document_processor.save_uploaded_files(settings["uploaded_files"])
                    st.session_state.rag_pipeline.load_documents(
                        file_paths, settings["chunk_size"], settings["chunk_overlap"]
                    )
                    st.session_state.processed_files = [f.name for f in settings["uploaded_files"]]
                    logger.info(f"Processed {len(file_paths)} documents")
                    st.success("Documents processed successfully!")
            
            # Query the pipeline
            with st.spinner("Thinking..."):
                response = st.session_state.rag_pipeline.query(
                    query, k=settings["k_retrievals"]
                )
                return response
        
        except ValueError as ve:
            logger.error(f"Validation error: {ve}")
            return f"⚠️ {str(ve)}"
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"❌ Error: {str(e)}"
    
    def run(self) -> None:
        """Run the RAG Chatbot application."""
        self._render_header()
        settings = self._render_sidebar()
        self._display_file_info(settings["uploaded_files"])
        self._check_file_changes(settings["uploaded_files"])
        self._render_main_content(settings["uploaded_files"])
        
        # Handle user input
        if query := st.chat_input("Ask me about your documents...", key="chat_input"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            response = self._process_query(query, settings)
            if response:
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Render instructions
        with st.expander("ℹ️ How to use this RAG Chatbot"):
            st.markdown("""
            ### Getting Started:
            1. **🔑 Add your Groq API key** in the sidebar
            2. **📄 Upload your documents** (PDF, TXT, CSV supported)
            3. **⚙️ Adjust settings** (model, embedding, chunking)
            4. **💬 Ask questions** about your documents
            
            ### Features:
            - Customizable embedding models
            - Multiple file support
            - Configurable chunking
            - Multiple LLM models
            
            ### Tips:
            - Popular embedding models: all-MiniLM-L6-v2, BAAI/bge-small-en
            - Start with specific questions
            - Use reset button to clear session
            """)
        
        # Render footer
        st.markdown("---")
        st.markdown("**Built with** 🔗 Streamlit • 🦜 LangChain • ⚡ Groq • 🤗 HuggingFace")
        
        # Display status
        if st.session_state.processed_files:
            st.sidebar.markdown("### 📊 Status")
            st.sidebar.success(f"✅ {len(st.session_state.processed_files)} files processed")
            st.sidebar.info(f"🔧 Model: {settings['model']}")
            st.sidebar.info(f"📏 Chunk size: {settings['chunk_size']}")
            st.sidebar.info(f"🧠 Embedding: {settings['embedding_model']}")

if __name__ == "__main__":
    app = RagChatbot()
    app.run()