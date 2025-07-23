"""
Complete Data Processing Pipeline - FIXED VERSION
Text → sentences → difficulty scores → curriculum → dataset
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

import nltk
import spacy
import textstat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class TextSegment:
    """A text segment with difficulty scores"""
    text: str
    index: int
    
    # Difficulty scores
    lexical_rarity: float = 0.0
    syntactic_complexity: float = 0.0
    thematic_centrality: float = 0.0
    combined_difficulty: float = 0.0
    
    # Metadata
    length: int = 0
    cluster_id: int = -1


class TextDataPipeline:
    """Complete pipeline for processing text and creating curriculum"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 n_clusters: int = 8,
                 target_vocab_size: int = 25000):
        
        self.embedding_model_name = embedding_model
        self.n_clusters = n_clusters
        self.target_vocab_size = target_vocab_size
        
        # Initialize models
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tokenizer = None  # Will be created from data
        
        # Data storage
        self.segments: List[TextSegment] = []
        self.embeddings: np.ndarray = None
        self.cluster_model = None
        
    def load_text(self, file_path: str) -> str:
        """Load and clean text from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Basic cleaning
        text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
        text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """Segment text into sentences"""
        # Use NLTK for sentence segmentation
        sentences = nltk.sent_tokenize(text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences
    
    def create_tokenizer(self, texts: List[str]) -> AutoTokenizer:
        """Create tokenizer with special tokens"""
        # Use base GPT-2 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Add special tokens if they don't exist
        special_tokens = {
            "pad_token": "<pad>",
            "mask_token": "<mask>", 
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }
        
        # Only add tokens that don't exist
        tokens_to_add = {}
        if tokenizer.pad_token is None:
            tokens_to_add["pad_token"] = special_tokens["pad_token"]
        if tokenizer.mask_token is None:
            tokens_to_add["mask_token"] = special_tokens["mask_token"] 
        if tokenizer.bos_token is None:
            tokens_to_add["bos_token"] = special_tokens["bos_token"]
        if tokenizer.eos_token is None:
            tokens_to_add["eos_token"] = special_tokens["eos_token"]
        
        if tokens_to_add:
            tokenizer.add_special_tokens(tokens_to_add)
        
        return tokenizer
    
    def calculate_lexical_difficulty(self, segments: List[str]) -> List[float]:
        """Calculate lexical rarity scores"""
        if self.tokenizer is None:
            self.tokenizer = self.create_tokenizer(segments)
        
        scores = []
        for text in segments:
            # Simple approach: use average word length and readability
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            avg_token_length = len(tokens) / max(1, len(text.split()))
            scores.append(avg_token_length)
        
        return scores
    
    def calculate_syntactic_complexity(self, segments: List[str]) -> List[float]:
        """Calculate syntactic complexity using readability metrics"""
        scores = []
        for text in segments:
            try:
                # Flesch Reading Ease (inverted so higher = harder)
                flesch = textstat.flesch_reading_ease(text)
                complexity = max(0, (100 - flesch) / 100)
            except:
                complexity = 0.5  # Default to medium complexity
            scores.append(complexity)
        
        return scores
    
    def calculate_thematic_centrality(self, segments: List[str]) -> List[float]:
        """Calculate thematic centrality using sentence embeddings"""
        print("Calculating sentence embeddings...")
        embeddings = self.embedding_model.encode(segments, show_progress_bar=True)
        self.embeddings = embeddings
        
        # Cluster sentences
        print(f"Clustering into {self.n_clusters} groups...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(embeddings_scaled)
        
        # Calculate centrality within each cluster
        centrality_scores = []
        for i, embedding in enumerate(embeddings):
            cluster_id = cluster_labels[i]
            cluster_center = self.cluster_model.cluster_centers_[cluster_id]
            
            # Distance to cluster center (inverted so closer = higher centrality)
            distance = np.linalg.norm(embeddings_scaled[i] - cluster_center)
            centrality = max(0, 1 - distance / 2)  # Normalize
            centrality_scores.append(centrality)
        
        return centrality_scores, cluster_labels
    
    def create_curriculum_splits(self) -> Dict[str, List[TextSegment]]:
        """Create curriculum splits based on difficulty"""
        # Sort by combined difficulty
        sorted_segments = sorted(self.segments, key=lambda x: x.combined_difficulty)
        
        n_segments = len(sorted_segments)
        easy_end = n_segments // 3
        medium_end = 2 * n_segments // 3
        
        splits = {
            'easy': sorted_segments[:easy_end],
            'medium': sorted_segments[easy_end:medium_end], 
            'hard': sorted_segments[medium_end:],
            'all': sorted_segments
        }
        
        return splits
    
    def process_text_file(self, file_path: str) -> List[TextSegment]:
        """Process text file and create segments with difficulty scores"""
        print(f"Processing: {file_path}")
        
        # Load and segment text
        text = self.load_text(file_path)
        sentences = self.segment_text(text)
        print(f"Found {len(sentences)} sentences")
        
        # Calculate difficulty scores
        print("Calculating lexical difficulty...")
        lexical_scores = self.calculate_lexical_difficulty(sentences)
        
        print("Calculating syntactic complexity...")
        syntactic_scores = self.calculate_syntactic_complexity(sentences)
        
        print("Calculating thematic centrality...")
        thematic_scores, cluster_labels = self.calculate_thematic_centrality(sentences)
        
        # Create segments
        self.segments = []
        for i, sentence in enumerate(sentences):
            # Combine difficulty scores (you can adjust weights)
            combined = (
                0.3 * lexical_scores[i] + 
                0.4 * syntactic_scores[i] + 
                0.3 * (1 - thematic_scores[i])  # Invert centrality for difficulty
            )
            
            segment = TextSegment(
                text=sentence,
                index=i,
                lexical_rarity=lexical_scores[i],
                syntactic_complexity=syntactic_scores[i],
                thematic_centrality=thematic_scores[i],
                combined_difficulty=combined,
                length=len(sentence.split()),
                cluster_id=cluster_labels[i]
            )
            self.segments.append(segment)
        
        return self.segments
    
    def save_data(self, output_dir: str):
        """Save processed data to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save segments
        with open(output_path / "segments.pkl", "wb") as f:
            pickle.dump(self.segments, f)
        
        # Save curriculum splits
        splits = self.create_curriculum_splits()
        with open(output_path / "curriculum_splits.pkl", "wb") as f:
            pickle.dump(splits, f)
        
        # Save tokenizer
        if self.tokenizer:
            tokenizer_path = output_path / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(output_path / "embeddings.npy", self.embeddings)
        
        # Save cluster model
        if self.cluster_model:
            with open(output_path / "cluster_model.pkl", "wb") as f:
                pickle.dump(self.cluster_model, f)
        
        print(f"Data saved to: {output_path}")
    
    def load_data(self, data_dir: str):
        """Load previously processed data"""
        data_path = Path(data_dir)
        
        # Load segments
        with open(data_path / "segments.pkl", "rb") as f:
            self.segments = pickle.load(f)
        
        # Load embeddings
        if (data_path / "embeddings.npy").exists():
            self.embeddings = np.load(data_path / "embeddings.npy")
        
        # Load tokenizer
        if (data_path / "tokenizer").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(data_path / "tokenizer")
        
        # Load cluster model
        if (data_path / "cluster_model.pkl").exists():
            with open(data_path / "cluster_model.pkl", "rb") as f:
                self.cluster_model = pickle.load(f)
        
        print(f"Data loaded from: {data_path}")


def main():
    """Example usage"""
    # Initialize pipeline
    pipeline = TextDataPipeline(n_clusters=8, target_vocab_size=25000)
    
    # Process text file
    segments = pipeline.process_text_file("data/raw/complete_works_shakespeare.txt")
    
    # Save results
    pipeline.save_data("data/processed")
    
    # Print statistics
    difficulties = [s.combined_difficulty for s in segments]
    print(f"\nDifficulty Statistics:")
    print(f"Mean: {np.mean(difficulties):.3f}")
    print(f"Std: {np.std(difficulties):.3f}")
    print(f"Range: {np.min(difficulties):.3f} - {np.max(difficulties):.3f}")


if __name__ == "__main__":
    main()