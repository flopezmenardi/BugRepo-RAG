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
        logger.info(f"🔍 Starting similarity search with embedding dimension: {len(query_embedding)}")
        logger.info(f"📊 Search filters: type='{bug_type}', product='{product}', component='{component}'")
        logger.info(f"🎯 Search parameters: top_k={top_k}, min_results={min_results}")
        
        # First, let's check index stats
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"📈 Index stats: {stats}")
        except Exception as e:
            logger.error(f"❌ Failed to get index stats: {str(e)}")
        
        # Strategy 1: Search with all three filters (type, product, component)
        logger.info("🔍 Strategy 1: Searching with ALL filters (type + product + component)")
        bug_ids = self._search_with_filters(
            query_embedding, 
            {"type": bug_type, "product": product, "component": component},
            top_k
        )
        
        if len(bug_ids) >= min_results:
            logger.info(f"✅ Strategy 1 SUCCESS: Found {len(bug_ids)} results with all filters")
            return bug_ids
        
        logger.warning(f"⚠️ Strategy 1 INSUFFICIENT: Only found {len(bug_ids)} results with all filters, dropping 'type' filter")
        
        # Strategy 2: Drop 'type' filter, keep product and component
        logger.info("🔍 Strategy 2: Searching without 'type' filter (product + component only)")
        bug_ids = self._search_with_filters(
            query_embedding,
            {"product": product, "component": component},
            top_k
        )
        
        if len(bug_ids) >= min_results:
            logger.info(f"✅ Strategy 2 SUCCESS: Found {len(bug_ids)} results without 'type' filter")
            return bug_ids
        
        logger.warning(f"⚠️ Strategy 2 INSUFFICIENT: Only found {len(bug_ids)} results without 'type' filter, dropping 'component' filter")
        
        # Strategy 3: Keep only 'product' filter (most important)
        logger.info("🔍 Strategy 3: Searching with ONLY 'product' filter")
        bug_ids = self._search_with_filters(
            query_embedding,
            {"product": product},
            top_k
        )
        
        if len(bug_ids) > 0:
            logger.info(f"✅ Strategy 3 SUCCESS: Found {len(bug_ids)} results with only 'product' filter")
        else:
            logger.error(f"❌ Strategy 3 FAILED: Found {len(bug_ids)} results even with minimal filtering!")
            
            # Emergency Strategy 4: Search with NO filters at all
            logger.info("🚨 Strategy 4: EMERGENCY search with NO filters")
            bug_ids = self._search_with_filters(query_embedding, {}, top_k)
            if len(bug_ids) > 0:
                logger.warning(f"⚠️ Strategy 4: Found {len(bug_ids)} results with NO filters - metadata filtering issue!")
            else:
                logger.error(f"💀 Strategy 4 TOTAL FAILURE: No results even without filters - check embedding or index!")
        
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
            active_filters = {}
            
            for field, value in filters.items():
                if value:  # Only add non-empty filters
                    pinecone_filter[field] = {"$eq": value}
                    active_filters[field] = value
            
            logger.info(f"🔍 Executing Pinecone query with {len(active_filters)} active filters: {active_filters}")
            logger.debug(f"🔧 Pinecone filter format: {pinecone_filter}")
            logger.debug(f"📏 Query embedding sample: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ..., {query_embedding[-1]:.4f}]")
            
            # Execute the query
            if pinecone_filter:
                logger.info(f"🎯 Querying WITH metadata filters")
                response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter=pinecone_filter
                )
            else:
                logger.info(f"🎯 Querying WITHOUT metadata filters")
                response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
            
            logger.info(f"📊 Pinecone returned {len(response.matches)} matches")
            
            # Extract bug IDs from the response
            bug_ids = []
            for i, match in enumerate(response.matches):
                # The bug_id should be in the metadata
                bug_id = match.metadata.get('bug_id')
                score = match.score
                
                logger.info(f"  📋 Match {i+1}: bug_id='{bug_id}', score={score:.4f}")
                logger.debug(f"     📂 Full metadata: {match.metadata}")
                
                if bug_id:
                    # Convert float bug IDs to integers for consistency
                    if isinstance(bug_id, (int, float)):
                        bug_id = str(int(float(bug_id)))
                    bug_ids.append(bug_id)
                else:
                    logger.warning(f"  ⚠️ Match {i+1} missing bug_id in metadata: {match.metadata}")
            
            if not bug_ids:
                logger.warning(f"❌ No valid bug_ids extracted from {len(response.matches)} matches!")
                if response.matches:
                    logger.info("🔍 Sample match metadata for debugging:")
                    for i, match in enumerate(response.matches[:2]):  # Show first 2 matches
                        logger.info(f"  Match {i+1} metadata keys: {list(match.metadata.keys())}")
                        logger.info(f"  Match {i+1} metadata: {match.metadata}")
            
            return bug_ids
            
        except Exception as e:
            logger.error(f"❌ Error during Pinecone search: {str(e)}")
            logger.error(f"🔧 Query details - filters: {filters}, top_k: {top_k}")
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
            # Convert bug IDs to integers to ensure consistent formatting
            normalized_bug_ids = []
            for bug_id in bug_ids:
                if isinstance(bug_id, (int, float)):
                    normalized_id = str(int(float(bug_id)))
                else:
                    # Try to convert string to int (removes .0 if present)
                    try:
                        normalized_id = str(int(float(bug_id)))
                    except (ValueError, TypeError):
                        normalized_id = str(bug_id)
                normalized_bug_ids.append(normalized_id)
            
            logger.info(f"📋 Fetching bug details for {len(normalized_bug_ids)} bugs: {normalized_bug_ids}")
            
            # Fetch vectors by their IDs to get full metadata
            vector_ids = [f"{bug_id}_0" for bug_id in normalized_bug_ids]  # Assuming chunk index 0
            logger.debug(f"🔍 Looking for vector IDs: {vector_ids}")
            
            response = self.index.fetch(ids=vector_ids)
            logger.info(f"📊 Pinecone fetch returned {len(response.vectors)} vectors out of {len(vector_ids)} requested")
            
            bug_details = []
            for bug_id in normalized_bug_ids:
                vector_id = f"{bug_id}_0"
                if vector_id in response.vectors:
                    metadata = response.vectors[vector_id].metadata
                    bug_details.append(metadata)
                    logger.debug(f"✅ Found metadata for bug {bug_id}")
                else:
                    logger.warning(f"❌ No vector found for bug ID {bug_id} (vector_id: {vector_id})")
            
            logger.info(f"📋 Successfully fetched details for {len(bug_details)} bugs")
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
