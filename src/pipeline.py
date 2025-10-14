"""
Main RAG Pipeline Orchestrator

This is the main entry point for the RAG system. It orchestrates the entire pipeline:
1. Load user bug report from JSON file
2. Enhance the bug text using LLM prompts
3. Convert enhanced text to embeddings
4. Retrieve similar bugs using vector search
5. Generate final report using LLM

Usage:
    python src/pipeline.py path/to/bug_report.json
"""

import sys
import json
import argparse
from pathlib import Path
import logging
from typing import Dict, Any, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.embeddings.embedder import BugReportEmbedder
from src.retrieval.retriever import BugReportRetriever
# from src.llm.prompts import enhance_bug_text  # Not implemented yet
from src.llm.generator import generate_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Main RAG Pipeline orchestrator for bug report analysis
    """
    
    def __init__(self):
        """Initialize all pipeline components"""
        logger.info("Initializing RAG Pipeline components...")
        
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        self.embedder = BugReportEmbedder()
        self.retriever = BugReportRetriever()
        logger.info("RAG Pipeline initialized successfully")
    
    def process_bug_report(self, bug_report_path: str) -> Dict[str, Any]:
        """
        Main pipeline method that processes a bug report through the entire RAG flow
        
        Args:
            bug_report_path (str): Path to JSON file containing bug report
            
        Returns:
            Dict[str, Any]: Results including similar bugs and generated report
        """
        logger.info(f"Processing bug report: {bug_report_path}")
        
        try:
            # Step 1: Load bug report from JSON file
            bug_data = self._load_bug_report(bug_report_path)
            if not bug_data:
                return {"error": "Failed to load bug report"}
            
            # Step 2: Enhance bug text using LLM prompts (placeholder for now)
            enhanced_text = self._enhance_bug_text(bug_data.get('summary', ''))
            
            # Step 3: Convert enhanced text to embeddings
            query_embedding = self._generate_embedding(enhanced_text)
            if not query_embedding:
                return {"error": "Failed to generate embedding"}
            
            # Step 4: Retrieve similar bugs using vector search
            similar_bug_ids = self._retrieve_similar_bugs(query_embedding, bug_data)
            
            # Step 5: Generate final report (placeholder for now)
            report_path = self._generate_report(bug_data, similar_bug_ids)
            
            # Return results
            results = {
                "status": "success",
                "input_bug": bug_data,
                "enhanced_text": enhanced_text,
                "similar_bugs_count": len(similar_bug_ids),
                "similar_bug_ids": similar_bug_ids,
                "report_path": report_path
            }
            
            logger.info("Pipeline processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            return {"error": f"Pipeline failed: {str(e)}"}
    
    def _load_bug_report(self, file_path: str) -> Dict[str, Any]:
        """
        Load bug report from JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            Dict[str, Any]: Bug report data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                bug_data = json.load(f)
            
            # Validate required fields
            required_fields = ["bug_id", "type", "product", "component", "status", "resolution", "updated"]
            missing_fields = [field for field in required_fields if field not in bug_data]
            
            if missing_fields:
                logger.warning(f"Missing fields in bug report: {missing_fields}")
            
            # Ensure we have some text to work with
            if not bug_data.get('summary') and not bug_data.get('description'):
                logger.warning("Bug report has no summary or description text")
                bug_data['summary'] = f"Bug {bug_data.get('bug_id', 'unknown')}"
            
            logger.info(f"Loaded bug report: ID={bug_data.get('bug_id')}, Type={bug_data.get('type')}, Product={bug_data.get('product')}")
            return bug_data
            
        except FileNotFoundError:
            logger.error(f"Bug report file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in bug report file: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error loading bug report: {str(e)}")
            return {}
    
    def _enhance_bug_text(self, original_text: str) -> str:
        """
        Enhance bug text using LLM prompts for better retrieval
        
        Args:
            original_text (str): Original bug summary/description
            
        Returns:
            str: Enhanced text optimized for retrieval
        """
        # TODO: Implement when prompts.py is ready
        # enhanced = enhance_bug_text(original_text)
        # return enhanced
        
        # Placeholder: return original text for now
        logger.info("Using original text (enhancement not implemented yet)")
        return original_text
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Convert text to embedding vector
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            embedding = self.embedder.embed_text(text)
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return []
    
    def _retrieve_similar_bugs(self, query_embedding: List[float], bug_data: Dict[str, Any]) -> List[str]:
        """
        Retrieve similar bugs using vector search with metadata filters
        
        Args:
            query_embedding (List[float]): Query embedding vector
            bug_data (Dict[str, Any]): Bug metadata for filtering
            
        Returns:
            List[str]: List of similar bug IDs
        """
        try:
            similar_bugs = self.retriever.retrieve_similar_bugs(
                query_embedding=query_embedding,
                bug_type=bug_data.get('type', ''),
                product=bug_data.get('product', ''),
                component=bug_data.get('component', ''),
                top_k=Config.TOP_K_RESULTS
            )
            
            logger.info(f"Retrieved {len(similar_bugs)} similar bugs")
            return similar_bugs
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar bugs: {str(e)}")
            return []
    
    def _generate_report(self, bug_data: Dict[str, Any], similar_bug_ids: List[str]) -> str:
        """
        Generate final report using LLM
        
        Args:
            bug_data (Dict[str, Any]): Original bug data
            similar_bug_ids (List[str]): IDs of similar bugs
            
        Returns:
            str: Path to generated report file
        """
        try:
            report_path = generate_report(bug_data, similar_bug_ids)
            logger.info("Generated LLM-backed report: %s", report_path)
            return report_path
        except Exception as e:
            logger.error(f"Failed to generate report: {str(e)}")
            return ""


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(description="Process a bug report through the RAG pipeline")
    parser.add_argument("bug_report", help="Path to JSON file containing bug report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run pipeline
    try:
        pipeline = RAGPipeline()
        results = pipeline.process_bug_report(args.bug_report)
        
        # Print results
        print("\n" + "="*50)
        print("RAG PIPELINE RESULTS")
        print("="*50)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✅ Status: {results['status']}")
            print(f"📝 Input Bug ID: {results['input_bug'].get('bug_id')}")
            print(f"🔍 Similar Bugs Found: {results['similar_bugs_count']}")
            print(f"📋 Similar Bug IDs: {', '.join(results['similar_bug_ids'][:5])}{'...' if len(results['similar_bug_ids']) > 5 else ''}")
            print(f"📄 Report Generated: {results['report_path']}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
