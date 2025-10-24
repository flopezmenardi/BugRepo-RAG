import sys
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config

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
 
        Config.validate_config()
        
        self.index_name = index_name or Config.PINECONE_INDEX_NAME
        self._init_pinecone()
        
        logger.info(f"Initialized BugReportRetriever for index: {self.index_name}")
    
    def _init_pinecone(self):
        """Initialize Pinecone client and connect to existing index"""
        try:
            self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)

            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def retrieve_similar_bugs(self, 
                            query_embedding: List[float], 
                            classification: str, 
                            product: str, 
                            component: str,
                            top_k: int = 5,
                            score_threshold: float = 0.75,
                            return_scores: bool = False) -> List[Any]:
        """
        Retrieve similar bug reports using cascading metadata filters with score-based fallback
        
        Args:
            query_embedding (List[float]): The embedding vector of the enhanced bug query
            classification (str): Classification filter (e.g., "Components", "Client Software")
            product (str): Product filter (e.g., "Firefox", "Core") - always required
            component (str): Component filter (e.g., "General", "Security")
            top_k (int): Maximum number of results to return (default 5)
            score_threshold (float): Minimum score threshold for keeping results (default 0.75)
            return_scores (bool): If True, include similarity scores alongside bug IDs
            
        Returns:
            List[str] or List[Dict[str, Any]]: Similar bug identifiers (optionally with scores)
        """
        logger.info(f"🔍 Starting similarity search with embedding dimension: {len(query_embedding)}")
        logger.info(f"📊 Search filters: classification='{classification}', product='{product}', component='{component}'")
        logger.info(f"🎯 Search parameters: top_k={top_k}, score_threshold={score_threshold}")
        
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"📈 Index stats: {stats}")
        except Exception as e:
            logger.error(f"❌ Failed to get index stats: {str(e)}")
        

        logger.info("🔍 Strategy 1: Searching with ALL filters (classification + product + component)")
        strategy1_candidates = self._search_with_filters(
            query_embedding, 
            {"classification": classification, "product": product, "component": component},
            top_k
        )

        final_candidates = []
        high_quality_from_s1 = [c for c in strategy1_candidates if c["score"] >= score_threshold]
        low_quality_from_s1 = [c for c in strategy1_candidates if c["score"] < score_threshold]
        
        logger.info(f"📊 Strategy 1 results: {len(strategy1_candidates)} total, {len(high_quality_from_s1)} above threshold")
        
        if len(high_quality_from_s1) >= top_k:
            final_candidates = high_quality_from_s1[:top_k]
            logger.info(f"✅ Strategy 1 SUCCESS: Found {len(final_candidates)} high-quality results with all filters")
        else:
            final_candidates.extend(high_quality_from_s1)
            needed = top_k - len(final_candidates)
            logger.info(f"⚠️ Strategy 1 PARTIAL: Only {len(high_quality_from_s1)} high-quality results, need {needed} more")
            
            if needed > 0:
                logger.info("🔍 Strategy 2: Searching with classification + product only")
                strategy2_candidates = self._search_with_filters(
                    query_embedding,
                    {"classification": classification, "product": product},
                    top_k * 2  
                )

                existing_ids = {c["bug_id"] for c in final_candidates}
                new_high_quality = [c for c in strategy2_candidates 
                                  if c["bug_id"] not in existing_ids and c["score"] >= score_threshold]

                strategy2_added = new_high_quality[:needed]
                final_candidates.extend(strategy2_added)
                needed -= len(strategy2_added)
                
                logger.info(f"📊 Strategy 2: Added {len(strategy2_added)} high-quality results")
                
                if needed > 0:
                    logger.info("🔍 Strategy 3: Searching with product only")
                    strategy3_candidates = self._search_with_filters(
                        query_embedding,
                        {"product": product},
                        top_k * 2 
                    )

                    existing_ids = {c["bug_id"] for c in final_candidates}
                    new_high_quality = [c for c in strategy3_candidates 
                                      if c["bug_id"] not in existing_ids and c["score"] >= score_threshold]

                    strategy3_added = new_high_quality[:needed]
                    final_candidates.extend(strategy3_added)
                    needed -= len(strategy3_added)
                    
                    logger.info(f"📊 Strategy 3: Added {len(strategy3_added)} high-quality results")

                    if needed > 0:
                        logger.warning(f"⚠️ Only found {len(final_candidates)} high-quality results, filling with lower-scoring ones")
                        all_low_quality = []
                        existing_ids = {c["bug_id"] for c in final_candidates}

                        all_low_quality.extend([c for c in low_quality_from_s1 if c["bug_id"] not in existing_ids])

                        strategy2_low = [c for c in strategy2_candidates 
                                       if c["bug_id"] not in existing_ids and c["score"] < score_threshold]
                        all_low_quality.extend(strategy2_low)

                        strategy3_low = [c for c in strategy3_candidates 
                                       if c["bug_id"] not in existing_ids and c["score"] < score_threshold]
                        all_low_quality.extend(strategy3_low)

                        all_low_quality.sort(key=lambda x: x["score"], reverse=True)
                        final_candidates.extend(all_low_quality[:needed])

        final_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"🎯 Final results: {len(final_candidates)} bugs retrieved")
        for i, candidate in enumerate(final_candidates, 1):
            logger.info(f"  📋 Result {i}: bug_id='{candidate['bug_id']}', score={candidate['score']:.4f}")
        
        return self._maybe_return_candidates(final_candidates, return_scores)
    
    def _search_with_filters(self, 
                           query_embedding: List[float], 
                           filters: Dict[str, str], 
                           top_k: int) -> List[Dict[str, Any]]:
        """
        Execute a single search with given metadata filters
        
        Args:
            query_embedding (List[float]): The query embedding vector
            filters (Dict[str, str]): Metadata filters to apply
            top_k (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of bug IDs with similarity scores
        """
        try:
            # Pinecone expects filters in this format: {"field": {"$eq": "value"}}
            pinecone_filter = {}
            active_filters = {}
            
            for field, value in filters.items():
                if value:  
                    pinecone_filter[field] = {"$eq": value}
                    active_filters[field] = value
            
            logger.info(f"🔍 Executing Pinecone query with {len(active_filters)} active filters: {active_filters}")
            logger.debug(f"🔧 Pinecone filter format: {pinecone_filter}")
            logger.debug(f"📏 Query embedding sample: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ..., {query_embedding[-1]:.4f}]")

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

            candidates: List[Dict[str, Any]] = []
            for i, match in enumerate(response.matches):
                bug_id = match.metadata.get('bug_id')
                score = match.score
                
                logger.info(f"  📋 Match {i+1}: bug_id='{bug_id}', score={score:.4f}")
                logger.debug(f"     📂 Full metadata: {match.metadata}")
                
                if bug_id:
                    if isinstance(bug_id, (int, float)):
                        bug_id = str(int(float(bug_id)))
                    candidates.append({"bug_id": bug_id, "score": score})
                else:
                    logger.warning(f"  ⚠️ Match {i+1} missing bug_id in metadata: {match.metadata}")
            
            if not candidates:
                logger.warning(f"❌ No valid bug_ids extracted from {len(response.matches)} matches!")
                if response.matches:
                    logger.info("🔍 Sample match metadata for debugging:")
                    for i, match in enumerate(response.matches[:2]): 
                        logger.info(f"  Match {i+1} metadata keys: {list(match.metadata.keys())}")
                        logger.info(f"  Match {i+1} metadata: {match.metadata}")
            
            return candidates
            
        except Exception as e:
            logger.error(f"❌ Error during Pinecone search: {str(e)}")
            logger.error(f"🔧 Query details - filters: {filters}, top_k: {top_k}")
            return []

    @staticmethod
    def _maybe_return_candidates(candidates: List[Dict[str, Any]], return_scores: bool) -> List[Any]:
        if return_scores:
            return candidates
        return [item["bug_id"] for item in candidates]
    
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
            normalized_bug_ids = []
            for bug_id in bug_ids:
                if isinstance(bug_id, (int, float)):
                    normalized_id = str(int(float(bug_id)))
                else:
                    try:
                        normalized_id = str(int(float(bug_id)))
                    except (ValueError, TypeError):
                        normalized_id = str(bug_id)
                normalized_bug_ids.append(normalized_id)
            
            logger.info(f"📋 Fetching bug details for {len(normalized_bug_ids)} bugs: {normalized_bug_ids}")

            vector_ids = [f"{bug_id}_0" for bug_id in normalized_bug_ids] 
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

        test_embedding = [0.1] * 512 
        
        # Test the retrieval
        similar_bugs = retriever.retrieve_similar_bugs(
            query_embedding=test_embedding,
            classification="Components",
            product="Firefox", 
            component="General"
        )
        
        print(f"Found {len(similar_bugs)} similar bugs: {similar_bugs}")
        
    except Exception as e:
        logger.error(f"Retriever test failed: {str(e)}")


if __name__ == "__main__":
    main()
