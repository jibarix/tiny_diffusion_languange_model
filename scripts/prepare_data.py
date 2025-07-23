#!/usr/bin/env python3
"""
Data Preparation Script
Process text file into curriculum-ready dataset
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pipeline import TextDataPipeline


def main():
    parser = argparse.ArgumentParser(description="Prepare text data for curriculum training")
    parser.add_argument("--book", required=True, help="Path to text file")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=25000, help="Target vocabulary size")
    parser.add_argument("--clusters", type=int, default=8, help="Number of thematic clusters")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", 
                       help="Sentence embedding model")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.book).exists():
        print(f"❌ File not found: {args.book}")
        sys.exit(1)
    
    print(f"📚 Processing: {args.book}")
    print(f"📁 Output: {args.output}")
    
    # Initialize pipeline
    pipeline = TextDataPipeline(
        embedding_model=args.embedding_model,
        n_clusters=args.clusters,
        target_vocab_size=args.vocab_size,
        enable_argument_mining=True,
        enable_vocab_curriculum=False  # Explicitly disable problematic feature
    )
    
    # Process text file
    try:
        segments = pipeline.process_text_file(args.book)
        print(f"✅ Processed {len(segments)} text segments")
        
        # Save results
        pipeline.save_data(args.output)
        
        # Print summary statistics
        difficulties = [s.combined_difficulty for s in segments]
        print(f"\n📊 Statistics:")
        print(f"   Segments: {len(segments)}")
        print(f"   Avg difficulty: {sum(difficulties)/len(difficulties):.3f}")
        print(f"   Vocab size: {pipeline.tokenizer.vocab_size if pipeline.tokenizer else 'N/A'}")
        print(f"   Clusters: {args.clusters}")
        
        print(f"\n🎯 Data ready for curriculum training!")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()