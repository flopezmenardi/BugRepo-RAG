#!/usr/bin/env python3
"""
Script to clear all vectors from the Pinecone database
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
import pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_pinecone_index():
    """Clear all vectors from the Pinecone index"""
    try:
        # Validate config
        Config.validate_config()
        
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Connect to index
        index_name = Config.PINECONE_INDEX_NAME
        logger.info(f"Connecting to Pinecone index: {index_name}")
        index = pc.Index(index_name)
        
        # Get index stats before clearing
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        logger.info(f"Index currently contains {total_vectors} vectors")
        
        if total_vectors == 0:
            logger.info("Index is already empty!")
            return
        
        # Confirm deletion
        response = input(f"Are you sure you want to delete all {total_vectors} vectors from '{index_name}'? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Operation cancelled")
            return
        
        # Delete all vectors (delete_all method)
        logger.info("Deleting all vectors from the index...")
        index.delete(delete_all=True)
        
        logger.info("✅ Successfully cleared all vectors from the Pinecone index")
        
        # Verify deletion
        import time
        time.sleep(2)  # Give it a moment to process
        stats_after = index.describe_index_stats()
        logger.info(f"Index now contains {stats_after.total_vector_count} vectors")
        
    except Exception as e:
        logger.error(f"❌ Error clearing Pinecone index: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    clear_pinecone_index()