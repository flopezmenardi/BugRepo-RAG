import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the RAG system"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # API Keys and Credentials
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    
    # Pinecone Configuration
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "bugrepo")
    PINECONE_DIMENSION = 512  # Match your existing Pinecone index dimension
    
    # Embedding Configuration
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model that produces 512 dimensions
    EMBEDDING_BATCH_SIZE = 100  # Process embeddings in batches
    
    # Text Processing
    CHUNK_SIZE = 1000  # Characters per chunk
    CHUNK_OVERLAP = 200  # Overlap between chunks
    
    # Retrieval Configuration
    TOP_K_RESULTS = 5  # Number of similar bugs to retrieve
    
    # LLM Configuration
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.7
    MAX_TOKENS = 1500
    
    @classmethod
    def validate_config(cls):
        """Validate that required environment variables are set"""
        required_vars = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("PINECONE_API_KEY", cls.PINECONE_API_KEY),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        
        return True