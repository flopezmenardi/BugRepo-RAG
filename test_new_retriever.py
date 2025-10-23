#!/usr/bin/env python3
"""
Test script to verify the updated retriever logic with score-based fallback
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.retrieval.retriever import BugReportRetriever
from src.embeddings.embedder import BugReportEmbedder

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_new_retriever_logic():
    """Test the updated retriever with classification filters and score-based fallback"""
    try:
        logger.info("🧪 Testing updated retriever logic...")
        
        # Initialize components
        embedder = BugReportEmbedder()
        retriever = BugReportRetriever()
        
        # Test with a real query
        test_query = "browser crash defect occurs when opening large files with numerous images and complex JavaScript interactions"
        
        logger.info(f"🔍 Generating embedding for test query: '{test_query[:80]}...'")
        query_embedding = embedder.embed_text(test_query)
        
        # Test the retrieval with different scenarios
        test_cases = [
            {
                "name": "High Specificity Test",
                "classification": "Client Software",
                "product": "Firefox", 
                "component": "General",
                "expected": "Should find specific Firefox/General/Components bugs"
            },
            {
                "name": "Broad Search Test", 
                "classification": "Client Software",
                "product": "Firefox for Android",
                "component": "General",
                "expected": "Should demonstrate fallback logic if specific matches are low-scoring"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n🧪 TEST CASE {i}: {test_case['name']}")
            logger.info(f"📋 Classification: {test_case['classification']}")
            logger.info(f"📋 Product: {test_case['product']}")
            logger.info(f"📋 Component: {test_case['component']}")
            logger.info(f"💡 Expected: {test_case['expected']}")
            
            # Run retrieval with scores
            results = retriever.retrieve_similar_bugs(
                query_embedding=query_embedding,
                classification=test_case['classification'],
                product=test_case['product'],
                component=test_case['component'],
                top_k=5,
                score_threshold=0.75,
                return_scores=True
            )
            
            logger.info(f"📊 RESULTS for {test_case['name']}:")
            if results:
                high_quality = [r for r in results if r['score'] >= 0.75]
                low_quality = [r for r in results if r['score'] < 0.75]
                
                logger.info(f"  ✅ Total results: {len(results)}")
                logger.info(f"  🎯 High quality (≥0.75): {len(high_quality)}")
                logger.info(f"  ⚠️ Low quality (<0.75): {len(low_quality)}")
                
                for j, result in enumerate(results, 1):
                    quality = "🎯" if result['score'] >= 0.75 else "⚠️"
                    logger.info(f"    {quality} Result {j}: ID={result['bug_id']}, Score={result['score']:.4f}")
            else:
                logger.warning(f"  ❌ No results found!")
        
        logger.info("\n✅ Retriever testing completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Retriever test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_new_retriever_logic()