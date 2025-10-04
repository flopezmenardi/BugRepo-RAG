import os
import json
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.embeddings.embedder import BugReportEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugReportIndexer:
    """
    Orchestrator for preparing raw bug report data and indexing it in Pinecone.
    
    This class handles the full pipeline:
    1. Load raw data from data directory
    2. Clean the data (via cleaner.py)
    3. Split into chunks (via splitter.py)
    4. Generate embeddings (via embedder.py)
    5. Store in Pinecone vector database
    """
    
    def __init__(self, index_name: str = None, test_limit: int = None):
        """
        Initialize the indexer
        
        Args:
            index_name (str): Name of the Pinecone index. Defaults to Config.PINECONE_INDEX_NAME
            test_limit (int): Limit number of bugs to process for testing. None for all bugs.
        """
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        self.embedder = BugReportEmbedder()
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self.test_limit = test_limit
        
        # Initialize Pinecone
        self._init_pinecone()
        
        # Placeholders for preprocessing components (to be implemented when we have data)
        # self.cleaner = BugReportCleaner()
        # self.splitter = BugReportSplitter()
        
        if self.test_limit:
            logger.info(f"Initialized BugReportIndexer for index: {self.index_name} (TEST MODE: {self.test_limit} bugs only)")
        else:
            logger.info(f"Initialized BugReportIndexer for index: {self.index_name}")
    
    def _init_pinecone(self):
        """Initialize Pinecone client and connect to existing index"""
        try:
            # Initialize Pinecone client with new API
            self.pc = Pinecone(
                api_key=Config.PINECONE_API_KEY
            )
            
            # Connect directly to the existing index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def index_bug_reports(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to index all bug reports from the data directory
        
        Args:
            data_path (str): Path to data file/directory. If None, uses Config.DATA_DIR
            
        Returns:
            Dict[str, Any]: Summary of indexing results
        """
        logger.info("Starting bug report indexing process...")
        
        # Use default data directory if no path provided
        if data_path is None:
            data_path = Config.DATA_DIR
        
        # Load raw data
        raw_data = self._load_raw_data(data_path)
        if not raw_data:
            logger.warning("No data found to index")
            return {"status": "no_data", "indexed_count": 0}
        
        logger.info(f"Loaded {len(raw_data)} raw bug reports")
        
        # Process data through the pipeline
        processed_chunks = self._process_data_pipeline(raw_data)
        
        if not processed_chunks:
            logger.warning("No chunks generated from data processing")
            return {"status": "no_chunks", "indexed_count": 0}
        
        logger.info(f"Generated {len(processed_chunks)} chunks for indexing")
        
        # Generate embeddings and store in Pinecone
        indexing_results = self._index_chunks(processed_chunks)
        
        logger.info("Bug report indexing completed successfully")
        return indexing_results
    
    def _load_raw_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load raw bug report data from the structured CSV file
        
        Args:
            data_path (str): Path to data file or directory
            
        Returns:
            List[Dict[str, Any]]: List of raw bug reports
        """
        data_path = Path(data_path)
        
        try:
            # If it's a directory, look for sample_bugs.csv
            if data_path.is_dir():
                csv_file = data_path / "sample_bugs.csv"
            else:
                csv_file = data_path
            
            if not csv_file.exists():
                logger.warning(f"CSV file does not exist: {csv_file}")
                return []
            
            # Load the structured CSV
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded CSV with columns: {list(df.columns)}")
            
            # Apply test limit if specified
            if self.test_limit:
                original_count = len(df)
                df = df.head(self.test_limit)
                logger.info(f"TEST MODE: Limited to {len(df)} bugs out of {original_count} total")
            
            # Convert to list of dictionaries
            bug_reports = df.to_dict('records')
            
            logger.info(f"Loaded {len(bug_reports)} bug reports from {csv_file}")
            return bug_reports
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            return []
    

    
    def _process_data_pipeline(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw data - simplified since data is already clean and structured
        
        Args:
            raw_data (List[Dict[str, Any]]): Raw bug report data from CSV
            
        Returns:
            List[Dict[str, Any]]: Processed chunks ready for embedding (one per bug)
        """
        processed_chunks = []
        
        logger.info("Processing structured bug data...")
        
        for i, bug_report in enumerate(tqdm(raw_data, desc="Processing bug reports")):
            try:
                # Extract the summary for embedding
                summary = bug_report.get('Summary', '')
                if not summary or summary.strip() == '':
                    logger.warning(f"Bug {bug_report.get('Bug ID', i)} has no summary, skipping")
                    continue
                
                # Clean the summary text (basic cleaning only)
                cleaned_summary = self._basic_clean_text(summary)
                
                # Check if summary needs splitting (fallback for very long summaries)
                chunks = self._split_if_needed(cleaned_summary)
                
                # Create chunk data for each piece (usually just one)
                for j, chunk_text in enumerate(chunks):
                    chunk_data = {
                        "id": f"{bug_report.get('Bug ID', i)}_{j}",
                        "text": chunk_text,
                        "original_bug_id": bug_report.get('Bug ID', i),
                        "chunk_index": j,
                        "metadata": {
                            "bug_id": bug_report.get('Bug ID', ''),
                            "type": bug_report.get('Type', ''),
                            "product": bug_report.get('Product', ''),
                            "component": bug_report.get('Component', ''),
                            "status": bug_report.get('Status', ''),
                            "resolution": bug_report.get('Resolution', ''),
                            "updated": bug_report.get('Updated', ''),
                        }
                    }
                    processed_chunks.append(chunk_data)
                    
            except Exception as e:
                logger.error(f"Error processing bug report {i}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(processed_chunks)} chunks from {len(raw_data)} bug reports")
        return processed_chunks
    
    def _basic_clean_text(self, text: str) -> str:
        """
        Basic text cleaning for embedding generation
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning - remove extra whitespace and newlines
        cleaned = text.strip()
        cleaned = " ".join(cleaned.split())  # Normalize whitespace
        
        return cleaned
    
    def _split_if_needed(self, text: str) -> List[str]:
        """
        Fallback splitter for very long summaries (rarely needed)
        
        Args:
            text (str): Text to potentially split
            
        Returns:
            List[str]: List of text chunks (usually just one)
        """
        # Most bug summaries should be short, but just in case...
        max_chars = 2000  # Conservative limit for embeddings
        
        if len(text) <= max_chars:
            return [text]
        
        logger.warning(f"Long summary detected ({len(text)} chars), splitting...")
        
        # Simple split at word boundaries
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chars
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to split at word boundary
            split_point = text.rfind(' ', start, end)
            if split_point == -1 or split_point <= start:
                split_point = end
            
            chunks.append(text[start:split_point])
            start = split_point + 1
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for chunks and store them in Pinecone
        
        Args:
            chunks (List[Dict[str, Any]]): Processed text chunks
            
        Returns:
            Dict[str, Any]: Indexing results
        """
        logger.info(f"Generating embeddings and indexing {len(chunks)} chunks...")
        
        # Prepare data for batch processing
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = self.embedder.embed_texts(texts)
        
        # Prepare vectors for Pinecone
        vectors_to_upsert = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            vector_data = {
                "id": chunk['id'],
                "values": embedding,
                "metadata": {
                    "text": chunk['text'][:1000],  # Truncate for metadata storage
                    "original_bug_id": chunk['original_bug_id'],
                    "chunk_index": chunk['chunk_index'],
                    **chunk['metadata']
                }
            }
            vectors_to_upsert.append(vector_data)
        
        # Batch upsert to Pinecone
        batch_size = 100
        total_upserted = 0
        
        for i in tqdm(range(0, len(vectors_to_upsert), batch_size), desc="Upserting to Pinecone"):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                upsert_response = self.index.upsert(vectors=batch)
                total_upserted += upsert_response.upserted_count
            except Exception as e:
                logger.error(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Get index stats
        stats = self.index.describe_index_stats()
        
        results = {
            "status": "success",
            "total_chunks": len(chunks),
            "indexed_count": total_upserted,
            "index_stats": stats
        }
        
        logger.info(f"Successfully indexed {total_upserted} vectors")
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get current index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {}
    
    def delete_all_vectors(self) -> bool:
        """Delete all vectors from the index (use with caution!)"""
        try:
            self.index.delete(delete_all=True)
            logger.info("All vectors deleted from index")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False


def main():
    """
    Main function to run the indexing process
    Can be called directly: python src/embeddings/indexer.py
    """
    try:
        # Initialize with test limit for safe testing
        # Change test_limit=5 to test_limit=None for full run
        indexer = BugReportIndexer(test_limit=5)
        
        # Test connection
        if not indexer.embedder.test_connection():
            logger.error("Failed to connect to OpenAI. Check your API key.")
            return
        
        # Run indexing
        results = indexer.index_bug_reports()
        
        print("\n" + "="*50)
        print("INDEXING RESULTS")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Total indexed: {results.get('indexed_count', 0)}")
        
        if 'index_stats' in results:
            stats = results['index_stats']
            print(f"Index total vectors: {stats.get('total_vector_count', 0)}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()