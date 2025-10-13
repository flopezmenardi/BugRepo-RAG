import sys
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugReportRetriever:
    """
    Retrieves similar bug reports from Pinecone vector database based on embeddings and metadata filters
    """
    
    def __init__(self, index_name: str = None):
        """
        Initialize the retriever with Pinecone connection
        
        Args:
            index_name (str): Name of the Pinecone index. Defaults to Config.PINECONE_INDEX_NAME
        """
        # Validate configuration
        Config.validate_config()
        
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        
        # Initialize Pinecone
        self._init_pinecone()
        
        logger.info(f"Initialized BugReportRetriever for index: {self.index_name}")
    
    def _init_pinecone(self):
        """Initialize Pinecone client and connect to existing index"""
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def retrieve_similar_bugs(self, 
                            query_embedding: List[float], 
                            bug_type: str, 
                            product: str, 
                            component: str,
                            top_k: int = 10,
                            min_results: int = 3) -> List[str]:
        """
        Retrieve similar bug reports using cascading metadata filters
        
        Args:
            query_embedding (List[float]): The embedding vector of the enhanced bug query
            bug_type (str): Bug type filter (e.g., "defect", "enhancement")
            product (str): Product filter (e.g., "Firefox", "Thunderbird") - always required
            component (str): Component filter (e.g., "General", "Security")
            top_k (int): Maximum number of results to return
            min_results (int): Minimum results needed before dropping filters
            
        Returns:
            List[str]: List of bug IDs of similar bugs
        """
        logger.info(f"Searching for similar bugs with filters: type={bug_type}, product={product}, component={component}")
        
        # Strategy 1: Search with all three filters (type, product, component)
        bug_ids = self._search_with_filters(
            query_embedding, 
            {"type": bug_type, "product": product, "component": component},
            top_k
        )
        
        if len(bug_ids) >= min_results:
            logger.info(f"Found {len(bug_ids)} results with all filters")
            return bug_ids
        
        logger.info(f"Only found {len(bug_ids)} results with all filters, dropping 'type' filter")
        
        # Strategy 2: Drop 'type' filter, keep product and component
        bug_ids = self._search_with_filters(
            query_embedding,
            {"product": product, "component": component},
            top_k
        )
        
        if len(bug_ids) >= min_results:
            logger.info(f"Found {len(bug_ids)} results without 'type' filter")
            return bug_ids
        
        logger.info(f"Only found {len(bug_ids)} results without 'type' filter, dropping 'component' filter")
        
        # Strategy 3: Keep only 'product' filter (most important)
        bug_ids = self._search_with_filters(
            query_embedding,
            {"product": product},
            top_k
        )
        
        logger.info(f"Found {len(bug_ids)} results with only 'product' filter")
        return bug_ids
    
    def _search_with_filters(self, 
                           query_embedding: List[float], 
                           filters: Dict[str, str], 
                           top_k: int) -> List[str]:
        """
        Execute a single search with given metadata filters
        
        Args:
            query_embedding (List[float]): The query embedding vector
            filters (Dict[str, str]): Metadata filters to apply
            top_k (int): Number of results to return
            
        Returns:
            List[str]: List of bug IDs
        """
        try:
            # Prepare filter for Pinecone query
            # Pinecone expects filters in this format: {"field": {"$eq": "value"}}
            pinecone_filter = {}
            for field, value in filters.items():
                if value:  # Only add non-empty filters
                    pinecone_filter[field] = {"$eq": value}
            
            # Execute the query
            if pinecone_filter:
                logger.debug(f"Querying with filters: {pinecone_filter}")
                response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=pinecone_filter
                )
            else:
                logger.debug("Querying without filters")
                response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
            
            # Extract bug IDs from the response
            bug_ids = []
            for match in response.matches:
                # The bug_id should be in the metadata
                bug_id = match.metadata.get('bug_id')
                if bug_id:
                    bug_ids.append(str(bug_id))
                    logger.debug(f"Found bug {bug_id} with score {match.score}")
            
            return bug_ids
            
        except Exception as e:
            logger.error(f"Error during Pinecone search: {str(e)}")
            return []
    
    def get_bug_details(self, bug_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve full metadata for given bug IDs (optional utility method)
        
        Args:
            bug_ids (List[str]): List of bug IDs to fetch
            
        Returns:
            List[Dict[str, Any]]: List of bug metadata
        """
        if not bug_ids:
            return []
        
        try:
            # Fetch vectors by their IDs to get full metadata
            response = self.index.fetch(ids=[f"{bug_id}_0" for bug_id in bug_ids])  # Assuming chunk index 0
            
            bug_details = []
            for bug_id in bug_ids:
                vector_id = f"{bug_id}_0"
                if vector_id in response.vectors:
                    metadata = response.vectors[vector_id].metadata
                    bug_details.append(metadata)
            
            return bug_details
            
        except Exception as e:
            logger.error(f"Error fetching bug details: {str(e)}")
            return []


def main():
    """
    Test function for the retriever
    """
    try:
        retriever = BugReportRetriever()
        
        # Example test query (you would get this from embedder in real use)
        test_embedding = [0.1] * 512  # Dummy 512-dimensional vector
        
        # Test the retrieval
        similar_bugs = retriever.retrieve_similar_bugs(
            query_embedding=test_embedding,
            bug_type="defect",
            product="Firefox", 
            component="General"
        )
        
        print(f"Found {len(similar_bugs)} similar bugs: {similar_bugs}")
        
    except Exception as e:
        logger.error(f"Retriever test failed: {str(e)}")


if __name__ == "__main__":
    main()
