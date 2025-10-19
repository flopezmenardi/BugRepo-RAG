#!/usr/bin/env python3
"""
Script to analyze unique values in the bugs_since.csv file
Specifically extracts unique values for classification and platform columns
"""

import logging
import pandas as pd
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_csv_values():
    """Analyze unique values in classification and platform columns"""
    try:
        # Load the CSV file
        csv_path = Path("data/bugs_since.csv")
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        logger.info(f"Loading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info(f"Total rows in CSV: {len(df)}")
        logger.info(f"Columns available: {list(df.columns)}")
        
        print("\n" + "="*60)
        print("UNIQUE VALUES ANALYSIS")
        print("="*60)
        
        # Analyze Classification column
        if 'classification' in df.columns:
            print(f"\n📊 CLASSIFICATION VALUES:")
            print("-" * 40)
            classification_counts = df['classification'].value_counts()
            for value, count in classification_counts.items():
                if pd.notna(value):  # Skip NaN values
                    print(f"  '{value}': {count} bugs")
            
            unique_classifications = df['classification'].dropna().unique()
            print(f"\n✅ Total unique classifications: {len(unique_classifications)}")
            print(f"📝 For LLM prompt: {sorted(unique_classifications)}")
        
        # Analyze Platform column
        if 'platform' in df.columns:
            print(f"\n🖥️  PLATFORM VALUES:")
            print("-" * 40)
            platform_counts = df['platform'].value_counts()
            for value, count in platform_counts.items():
                if pd.notna(value):  # Skip NaN values
                    print(f"  '{value}': {count} bugs")
            
            unique_platforms = df['platform'].dropna().unique()
            print(f"\n✅ Total unique platforms: {len(unique_platforms)}")
            print(f"📝 For LLM prompt: {sorted(unique_platforms)}")
        
        # Bonus: Analyze other relevant columns
        other_columns = ['priority', 'severity', 'product', 'component', 'status']
        for col in other_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                if len(unique_values) <= 20:  # Only show if not too many values
                    print(f"\n🔍 {col.upper()} VALUES:")
                    print("-" * 40)
                    value_counts = df[col].value_counts()
                    for value, count in value_counts.head(10).items():  # Top 10
                        if pd.notna(value):
                            print(f"  '{value}': {count} bugs")
                    print(f"📝 For LLM prompt: {sorted(unique_values)}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"❌ Error analyzing CSV: {str(e)}")


if __name__ == "__main__":
    analyze_csv_values()