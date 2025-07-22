"""
Complete Data Processing Pipeline
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
        """Create custom tokenizer from text corpus"""
        # Start with GPT-2 tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Get vocabulary frequency
        word_freq = {}
        for text in texts:
            tokens = base_tokenizer.tokenize(text.lower())
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and take top tokens
        sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_vocab = [token for token, freq in sorted_vocab[:self.target_vocab_size-4]]
        
        # Add special tokens
        special_tokens = ["<pad>", "<mask>", "<bos>", "<eos>"]
        final_vocab = special_tokens + top_vocab
        
        # Create new tokenizer with compressed vocab
        tokenizer = base_tokenizer
        tokenizer.vocab = {token: i for i, token in enumerate(final_vocab)}
        tokenizer.vocab_size = len(final_vocab)
        
        return tokenizer
    
    def calculate_lexical_difficulty(self, segments: List[str]) -> List[float]:
        """Calculate lexical rarity scores"""
        if self.tokenizer is None:
            self.tokenizer = self.create_tokenizer(segments)
        
        scores = []
        
        # Calculate token frequencies across all segments
        all_tokens = []
        for segment in segments:
            tokens = self.tokenizer.tokenize(segment.lower())
            all_tokens.extend(tokens)
        
        token_freq = {}
        for token in all_tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        total_tokens = len(all_tokens)
        
        # Score each segment
        for segment in segments:
            tokens = self.tokenizer.tokenize(segment.lower())
            if not tokens:
                scores.append(0.0)
                continue
            
            # Calculate average inverse frequency
            rarities = []
            for token in tokens:
                freq = token_freq.get(token, 1)
                rarity = np.log(total_tokens / freq)  # IDF-like score
                rarities.append(rarity)
            
            scores.append(np.mean(rarities))
        
        # Normalize to 0-1
        scores = np.array(scores)
        if scores.std() > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores.tolist()
    
    def calculate_syntactic_difficulty(self, segments: List[str]) -> List[float]:
        """Calculate syntactic complexity scores"""
        scores = []
        
        for segment in segments:
            # Use multiple readability metrics
            flesch_kincaid = textstat.flesch_kincaid_grade(segment)
            gunning_fog = textstat.gunning_fog(segment)
            
            # Sentence length and clause count
            doc = self.nlp(segment)
            sent_length = len([token for token in doc if not token.is_space])
            
            # Count subordinate clauses (approximate)
            clause_markers = len([token for token in doc 
                                if token.dep_ in ['mark', 'advcl', 'acl', 'relcl']])
            
            # Combine metrics
            complexity = (
                flesch_kincaid * 0.4 + 
                gunning_fog * 0.3 + 
                np.log(sent_length) * 0.2 + 
                clause_markers * 0.1
            )
            
            scores.append(max(0, complexity))
        
        # Normalize to 0-1
        scores = np.array(scores)
        if scores.std() > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores.tolist()
    
    def calculate_thematic_centrality(self, segments: List[str]) -> List[float]:
        """Calculate thematic centrality using clustering"""
        # Generate embeddings
        print("Generating sentence embeddings...")
        embeddings = self.embedding_model.encode(segments, show_progress_bar=True)
        self.embeddings = embeddings
        
        # Perform clustering
        print(f"Clustering into {self.n_clusters} themes...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        self.cluster_model = kmeans
        
        # Calculate centrality (inverse distance to cluster center)
        centralities = []
        for i, embedding in enumerate(embeddings):
            cluster_id = cluster_labels[i]
            center = kmeans.cluster_centers_[cluster_id]
            distance = np.linalg.norm(embedding - center)
            centrality = 1.0 / (1.0 + distance)  # Inverse distance
            centralities.append(centrality)
        
        # Normalize to 0-1 (higher = more central)
        centralities = np.array(centralities)
        if centralities.std() > 0:
            centralities = (centralities - centralities.min()) / (centralities.max() - centralities.min())
        
        return centralities.tolist(), cluster_labels.tolist()
    
    def process_text_file(self, file_path: str) -> List[TextSegment]:
        """Complete pipeline: text file → processed segments"""
        print(f"Processing: {file_path}")
        
        # Load and segment
        text = self.load_text(file_path)
        sentences = self.segment_text(text)
        print(f"Found {len(sentences)} sentences")
        
        # Calculate difficulty scores
        print("Calculating lexical difficulty...")
        lexical_scores = self.calculate_lexical_difficulty(sentences)
        
        print("Calculating syntactic difficulty...")
        syntactic_scores = self.calculate_syntactic_difficulty(sentences)
        
        print("Calculating thematic centrality...")
        centrality_scores, cluster_labels = self.calculate_thematic_centrality(sentences)
        
        # Create segments with scores
        segments = []
        for i, sentence in enumerate(sentences):
            segment = TextSegment(
                text=sentence,
                index=i,
                lexical_rarity=lexical_scores[i],
                syntactic_complexity=syntactic_scores[i],
                thematic_centrality=centrality_scores[i],
                length=len(sentence),
                cluster_id=cluster_labels[i]
            )
            
            # Combined difficulty score (weighted)
            segment.combined_difficulty = (
                0.3 * segment.lexical_rarity +
                0.3 * segment.syntactic_complexity +
                0.4 * (1.0 - segment.thematic_centrality)  # Invert centrality
            )
            
            segments.append(segment)
        
        self.segments = segments
        return segments
    
    def create_curriculum_splits(self, 
                                easy_threshold: float = 33.0,
                                hard_threshold: float = 67.0) -> Dict[str, List[TextSegment]]:
        """Split segments into curriculum stages"""
        if not self.segments:
            raise ValueError("No segments found. Run process_text_file first.")
        
        # Sort by combined difficulty
        sorted_segments = sorted(self.segments, key=lambda x: x.combined_difficulty)
        
        n_segments = len(sorted_segments)
        easy_cutoff = int(n_segments * easy_threshold / 100)
        hard_cutoff = int(n_segments * hard_threshold / 100)
        
        splits = {
            'easy': sorted_segments[:easy_cutoff],
            'medium': sorted_segments[easy_cutoff:hard_cutoff],
            'hard': sorted_segments[hard_cutoff:],
            'all': sorted_segments
        }
        
        print(f"Curriculum splits: Easy({len(splits['easy'])}), "
              f"Medium({len(splits['medium'])}), Hard({len(splits['hard'])})")
        
        return splits
    
    def save_data(self, output_dir: str):
        """Save processed data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save segments
        with open(output_path / "segments.pkl", "wb") as f:
            pickle.dump(self.segments, f)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(output_path / "embeddings.npy", self.embeddings)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path / "tokenizer")
        
        # Save cluster model
        if self.cluster_model is not None:
            with open(output_path / "cluster_model.pkl", "wb") as f:
                pickle.dump(self.cluster_model, f)
        
        # Save curriculum splits
        splits = self.create_curriculum_splits()
        with open(output_path / "curriculum_splits.pkl", "wb") as f:
            pickle.dump(splits, f)
        
        print(f"Data saved to: {output_path}")
    
    def load_data(self, data_dir: str):
        """Load processed data"""
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
    segments = pipeline.process_text_file("data/raw/origin_of_species.txt")
    
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