#!/usr/bin/env python3
"""
Alternative approach: Generate vector IDs based on CSV data and update metadata
This approach is more efficient since we know the ID pattern: bug_id_0, bug_id_1, etc.
"""

import logging
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.embeddings.indexer import BugReportIndexer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_classification_metadata_efficiently():
    """
    Efficiently add classification metadata by using known vector ID patterns
    """
    try:
        logger.info("🚀 Starting efficient classification metadata update...")
        
        # Initialize indexer
        indexer = BugReportIndexer(test_limit=None)
        index = indexer.index
        
        # Step 1: Load CSV data
        csv_path = Config.PROJECT_ROOT / "data" / "bugs_since.csv"
        logger.info(f"📖 Loading data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loaded {len(df)} bugs from CSV")
        
        # Clean classification data
        df['classification'] = df['classification'].fillna('')
        df['classification'] = df['classification'].apply(indexer._clean_metadata_value)
        
        # Step 2: Generate vector IDs based on bug_ids (assuming chunk index 0 for most)
        vector_updates = []
        missing_vectors = []
        
        logger.info("🔍 Preparing vector ID list and classification data...")
        
        batch_size = 100
        total_processed = 0
        total_updated = 0
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing bug batches"):
            batch_df = df.iloc[i:i + batch_size]
            
            # Create vector IDs for this batch (assuming _0 chunk index)
            vector_ids = []
            bug_classifications = {}
            
            for _, row in batch_df.iterrows():
                bug_id = str(int(float(row['bug_id'])))
                classification = row['classification']
                vector_id = f"{bug_id}_0"  # Most bugs have chunk index 0
                
                vector_ids.append(vector_id)
                bug_classifications[vector_id] = classification
            
            # Fetch existing vectors to get their current metadata and embeddings
            try:
                fetch_response = index.fetch(ids=vector_ids)
                
                vectors_to_update = []
                
                for vector_id in vector_ids:
                    if vector_id in fetch_response.vectors:
                        vector_data = fetch_response.vectors[vector_id]
                        classification = bug_classifications[vector_id]
                        
                        # Preserve existing metadata and add classification
                        updated_metadata = dict(vector_data.metadata) if vector_data.metadata else {}
                        updated_metadata['classification'] = classification
                        
                        # Create update vector (same ID, same embedding, enhanced metadata)
                        update_vector = {
                            'id': vector_id,
                            'values': vector_data.values,
                            'metadata': updated_metadata
                        }
                        vectors_to_update.append(update_vector)
                    else:
                        missing_vectors.append(vector_id)
                
                # Upsert batch with updated metadata
                if vectors_to_update:
                    upsert_response = index.upsert(vectors=vectors_to_update)
                    batch_updated = upsert_response.upserted_count
                    total_updated += batch_updated
                    
                    logger.debug(f"✅ Batch {i//batch_size + 1}: Updated {batch_updated} vectors")
                
                total_processed += len(vector_ids)
                
            except Exception as e:
                logger.error(f"❌ Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Report results
        logger.info("📊 METADATA UPDATE RESULTS:")
        logger.info(f"  📋 Total bugs processed: {len(df)}")
        logger.info(f"  ✅ Vectors updated: {total_updated}")
        logger.info(f"  ❌ Missing vectors: {len(missing_vectors)}")
        
        if missing_vectors:
            logger.info(f"⚠️ Sample missing vector IDs: {missing_vectors[:10]}")
            logger.info("💡 These might use different chunk indices or weren't indexed")
        
        # Step 3: Verification
        logger.info("🔍 Verifying updates...")
        
        # Sample a few vectors to verify classification was added
        dummy_vector = [0.0] * 512
        sample_response = index.query(
            vector=dummy_vector,
            top_k=5,
            include_metadata=True
        )
        
        logger.info("📋 VERIFICATION SAMPLE:")
        for i, match in enumerate(sample_response.matches, 1):
            metadata = match.metadata
            classification = metadata.get('classification', 'MISSING')
            bug_id = metadata.get('bug_id', 'UNKNOWN')
            
            logger.info(f"  Sample {i}: {match.id}")
            logger.info(f"    Bug ID: {bug_id}")
            logger.info(f"    Classification: '{classification}'")
            logger.info(f"    Has classification: {'✅' if 'classification' in metadata else '❌'}")
        
        logger.info("✅ Classification metadata update completed!")
        return total_updated, len(missing_vectors)
        
    except Exception as e:
        logger.error(f"❌ Metadata update failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    add_classification_metadata_efficiently()