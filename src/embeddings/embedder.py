from openai import OpenAI
from typing import List, Union
import numpy as np
import time
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugReportEmbedder:
    """
    A class to generate embeddings for bug reports using OpenAI's embedding models
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedder with OpenAI configuration
        
        Args:
            model_name (str): Name of the embedding model to use. 
                             Defaults to Config.EMBEDDING_MODEL
        """

        Config.validate_config()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.batch_size = Config.EMBEDDING_BATCH_SIZE
        
        logger.info(f"Initialized BugReportEmbedder with model: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:

            cleaned_text = self._clean_text(text)
            
            
            if "text-embedding-3-small" in self.model_name:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=cleaned_text,
                    dimensions=512  #  512 dimensions Pinecone compatibility
                )
            else:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=cleaned_text
                )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text of length {len(text)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
    
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            if i + self.batch_size < len(texts):
                time.sleep(0.1)
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return all_embeddings
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts (List[str]): Batch of texts to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            if "text-embedding-3-small" in self.model_name:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=cleaned_texts,
                    dimensions=512  # 512 dimensions Pinecone compatibility
                )
            else:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=cleaned_texts
                )
            
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"Generated embeddings for batch of {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            logger.info("Falling back to individual embedding generation")
            return [self.embed_text(text) for text in texts]
    
    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning for embedding generation
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text.strip()
        cleaned = " ".join(cleaned.split())
        
        max_chars = 8000 
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        return cleaned
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model
        
        Returns:
            int: Embedding dimension
        """
        if "ada-002" in self.model_name:
            return 1536
        elif "text-embedding-3-small" in self.model_name:
            return 512
        elif "text-embedding-3-large" in self.model_name:
            return 3072
        else:
            logger.warning(f"Unknown dimension for model {self.model_name}, assuming 512")
            return 512
    
    def test_connection(self) -> bool:
        """
        Test the connection to OpenAI by generating a simple embedding
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_embedding = self.embed_text("test connection")
            logger.info("OpenAI connection test successful")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {str(e)}")
            return False