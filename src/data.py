"""
Data Pipeline with Curriculum Construction

Complete data processing pipeline for tiny text diffusion model:
- Text preprocessing and segmentation
- Multi-dimensional difficulty scoring (lexical, syntactic, centrality)
- Curriculum construction for 3-stage training
- Compressed tokenizer creation
- Dataset formatting for each curriculum stage

Based on 2025 research on curriculum learning and data efficacy.
"""

import os
import re
import json
import pickle
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# Core dependencies
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

# NLP processing
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import textstat

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


@dataclass
class TextSegment:
    """Individual text segment with metadata"""
    text: str
    index: int
    start_pos: int
    end_pos: int
    length: int
    
    # Difficulty scores
    lexical_difficulty: float = 0.0
    syntactic_difficulty: float = 0.0
    centrality_score: float = 0.0
    composite_difficulty: float = 0.0
    
    # Optional metadata
    cluster_id: Optional[int] = None
    argument_role: Optional[str] = None
    has_dialogue: bool = False
    sentence_count: int = 1


class DifficultyScorer:
    """Multi-dimensional difficulty scoring for text segments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scoring_config = config.get('difficulty_scoring', {})
        
        # Initialize components
        self.nlp = None
        self.embedding_model = None
        self.idf_vectorizer = None
        self.reference_idf_scores = None
        
        self._setup_models()
    
    def _setup_models(self):
        """Initialize NLP models and tools"""
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        
        # Load sentence embedding model
        embedding_model_name = self.scoring_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize IDF vectorizer for lexical difficulty
        self.idf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=50000,
            ngram_range=(1, 1)
        )
    
    def compute_lexical_difficulty(self, segments: List[TextSegment], reference_corpus: Optional[List[str]] = None) -> List[float]:
        """
        Compute lexical difficulty based on word rarity (IDF scores).
        Higher scores = more rare/difficult vocabulary.
        """
        # Use provided corpus or create from segments
        if reference_corpus is None:
            corpus = [seg.text for seg in segments]
        else:
            corpus = reference_corpus + [seg.text for seg in segments]
        
        # Fit IDF vectorizer
        self.idf_vectorizer.fit(corpus)
        vocabulary = self.idf_vectorizer.vocabulary_
        idf_scores = self.idf_vectorizer.idf_
        
        # Create IDF lookup
        word_to_idf = {word: idf_scores[idx] for word, idx in vocabulary.items()}
        
        difficulties = []
        for segment in segments:
            doc = self.nlp(segment.text.lower())
            content_words = [token.text for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
            
            if not content_words:
                difficulties.append(0.0)
                continue
            
            # Average IDF score of content words
            idf_scores_segment = [word_to_idf.get(word, max(word_to_idf.values())) for word in content_words]
            avg_idf = np.mean(idf_scores_segment)
            difficulties.append(avg_idf)
        
        # Normalize to 0-1 range
        min_diff, max_diff = min(difficulties), max(difficulties)
        if max_diff > min_diff:
            difficulties = [(d - min_diff) / (max_diff - min_diff) for d in difficulties]
        
        return difficulties
    
    def compute_syntactic_difficulty(self, segments: List[TextSegment]) -> List[float]:
        """
        Compute syntactic difficulty using multiple linguistic features.
        """
        difficulties = []
        
        for segment in segments:
            doc = self.nlp(segment.text)
            
            # Feature extraction
            features = {
                'sentence_length': len(segment.text.split()),
                'num_sentences': len(list(doc.sents)),
                'avg_sentence_length': len(segment.text.split()) / max(len(list(doc.sents)), 1),
                'num_clauses': len([token for token in doc if token.dep_ in ['csubj', 'ccomp', 'advcl', 'acl', 'relcl']]),
                'flesch_kincaid': textstat.flesch_kincaid_grade(segment.text),
                'parse_tree_depth': self._get_max_depth(doc),
                'num_complex_words': len([token for token in doc if len(token.text) > 6 and token.is_alpha]),
            }
            
            # Weighted composite score (from config)
            feature_weights = self.scoring_config.get('syntactic_difficulty', {}).get('features', {
                'sentence_length': {'weight': 0.2},
                'num_clauses': {'weight': 0.3},
                'flesch_kincaid': {'weight': 0.3},
                'parse_tree_depth': {'weight': 0.2},
            })
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for feature_name, feature_config in feature_weights.items():
                if feature_name in features:
                    weight = feature_config['weight']
                    value = features[feature_name]
                    weighted_score += weight * value
                    total_weight += weight
            
            if total_weight > 0:
                weighted_score /= total_weight
            
            difficulties.append(max(0.0, weighted_score))
        
        # Normalize to 0-1 range
        if difficulties:
            min_diff, max_diff = min(difficulties), max(difficulties)
            if max_diff > min_diff:
                difficulties = [(d - min_diff) / (max_diff - min_diff) for d in difficulties]
        
        return difficulties
    
    def _get_max_depth(self, doc) -> int:
        """Calculate maximum parse tree depth"""
        def get_depth(token):
            if not list(token.children):
                return 1
            return 1 + max(get_depth(child) for child in token.children)
        
        depths = [get_depth(sent.root) for sent in doc.sents]
        return max(depths) if depths else 1
    
    def compute_centrality_scores(self, segments: List[TextSegment]) -> List[float]:
        """
        Compute thematic centrality using clustering.
        Higher scores = more prototypical/central examples.
        """
        if len(segments) < 2:
            return [1.0] * len(segments)
        
        # Get embeddings
        texts = [seg.text for seg in segments]
        embeddings = self.embedding_model.encode(texts)
        
        # Determine number of clusters
        num_clusters = min(
            self.scoring_config.get('num_clusters', 10),
            max(2, len(segments) // 5)  # At least 5 samples per cluster
        )
        
        # Cluster embeddings
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
        
        # Compute centrality scores (inverse of distance to cluster center)
        centrality_scores = []
        for i, (embedding, cluster_id) in enumerate(zip(embeddings, cluster_labels)):
            center = cluster_centers[cluster_id]
            distance = euclidean_distances([embedding], [center])[0][0]
            
            # Convert distance to centrality (higher = more central)
            # Using exponential decay for smoother distribution
            centrality = np.exp(-distance)
            centrality_scores.append(centrality)
            
            # Store cluster ID in segment
            segments[i].cluster_id = cluster_id
        
        # Normalize to 0-1 range
        min_cent, max_cent = min(centrality_scores), max(centrality_scores)
        if max_cent > min_cent:
            centrality_scores = [(c - min_cent) / (max_cent - min_cent) for c in centrality_scores]
        
        return centrality_scores
    
    def compute_composite_scores(self, segments: List[TextSegment]) -> List[float]:
        """
        Combine all difficulty dimensions into composite scores.
        """
        # Get weights from config
        weights = {
            'lexical_difficulty': self.scoring_config.get('lexical_difficulty', {}).get('weight', 0.3),
            'syntactic_difficulty': self.scoring_config.get('syntactic_difficulty', {}).get('weight', 0.4),
            'thematic_centrality': self.scoring_config.get('thematic_centrality', {}).get('weight', 0.3),
        }
        
        composite_scores = []
        for segment in segments:
            # Centrality is inverted for difficulty (high centrality = low difficulty)
            centrality_difficulty = 1.0 - segment.centrality_score
            
            composite = (
                weights['lexical_difficulty'] * segment.lexical_difficulty +
                weights['syntactic_difficulty'] * segment.syntactic_difficulty +
                weights['thematic_centrality'] * centrality_difficulty
            )
            composite_scores.append(composite)
        
        return composite_scores
    
    def score_segments(self, segments: List[TextSegment], reference_corpus: Optional[List[str]] = None) -> List[TextSegment]:
        """
        Apply all difficulty scoring methods to segments.
        """
        print(f"Computing difficulty scores for {len(segments)} segments...")
        
        # Lexical difficulty
        lexical_scores = self.compute_lexical_difficulty(segments, reference_corpus)
        for seg, score in zip(segments, lexical_scores):
            seg.lexical_difficulty = score
        
        # Syntactic difficulty
        syntactic_scores = self.compute_syntactic_difficulty(segments)
        for seg, score in zip(segments, syntactic_scores):
            seg.syntactic_difficulty = score
        
        # Centrality scores
        centrality_scores = self.compute_centrality_scores(segments)
        for seg, score in zip(segments, centrality_scores):
            seg.centrality_score = score
        
        # Composite scores
        composite_scores = self.compute_composite_scores(segments)
        for seg, score in zip(segments, composite_scores):
            seg.composite_difficulty = score
        
        print("Difficulty scoring complete.")
        return segments


class TextPreprocessor:
    """Text preprocessing and segmentation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config.get('data', {})
        
        # Segmentation parameters
        self.min_length = self.data_config.get('min_sentence_length', 10)
        self.max_length = self.data_config.get('max_sentence_length', 200)
        
        # Initialize NLTK
        self.sent_tokenizer = nltk.sent_tokenize
    
    def load_text(self, filepath: str) -> str:
        """Load and basic clean text from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Basic cleaning
        text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str) -> List[TextSegment]:
        """
        Segment text into meaningful units (sentences or short paragraphs).
        """
        # Split into sentences
        sentences = self.sent_tokenizer(text)
        
        segments = []
        current_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            # Filter by length
            word_count = len(sentence.split())
            if self.min_length <= word_count <= self.max_length:
                # Find position in original text
                start_pos = text.find(sentence, current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                end_pos = start_pos + len(sentence)
                
                segment = TextSegment(
                    text=sentence,
                    index=len(segments),
                    start_pos=start_pos,
                    end_pos=end_pos,
                    length=word_count
                )
                segments.append(segment)
                
                current_pos = end_pos
        
        print(f"Segmented text into {len(segments)} segments")
        return segments


class CompressedTokenizer:
    """Create compressed tokenizer optimized for single corpus"""
    
    def __init__(self, base_model_name: str = "gpt2"):
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        # Add special tokens for diffusion
        special_tokens = {"mask_token": "[MASK]"}
        self.base_tokenizer.add_special_tokens(special_tokens)
        
        self.compressed_vocab = None
        self.token_mapping = None
        self.inverse_mapping = None
    
    def create_compressed_vocab(self, texts: List[str], target_coverage: float = 0.9, max_vocab_size: int = 25000) -> Dict[str, Any]:
        """
        Create compressed vocabulary covering target percentage of corpus.
        FIXED: Ensures proper token ID mapping with PAD != 0 to avoid collapse.
        """
        print(f"Creating compressed vocabulary (target coverage: {target_coverage:.1%})...")
        
        # Tokenize all texts with base tokenizer
        all_tokens = []
        for text in texts:
            tokens = self.base_tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Sort by frequency
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize selected tokens list
        selected_tokens = []
        cumulative_count = 0
        
        # CRITICAL FIX: Place special tokens in specific order to match GPT-2 expectations
        # GPT-2 expects <|endoftext|> at position 0
        special_tokens_ordered = []
        
        # Add <|endoftext|> first (position 0) to match GPT-2
        if self.base_tokenizer.eos_token:
            special_tokens_ordered.append(self.base_tokenizer.eos_token)
        
        # Add MASK token next (position 1)
        special_tokens_ordered.append("[MASK]")
        
        # Add PAD token at position 2 (NOT 0!)
        if self.base_tokenizer.pad_token and self.base_tokenizer.pad_token not in special_tokens_ordered:
            special_tokens_ordered.append(self.base_tokenizer.pad_token)
        
        # Add BOS if different from EOS
        if self.base_tokenizer.bos_token and self.base_tokenizer.bos_token not in special_tokens_ordered:
            special_tokens_ordered.append(self.base_tokenizer.bos_token)
        
        # Add special tokens to selected_tokens and count their frequency
        for token in special_tokens_ordered:
            selected_tokens.append(token)
            # Count special token occurrences in corpus (if any)
            if token in token_counts:
                cumulative_count += token_counts[token]
        
        # Add frequent tokens to reach target coverage
        for token, count in sorted_tokens:
            if token not in selected_tokens:  # Avoid duplicates
                selected_tokens.append(token)
                cumulative_count += count
                
                coverage = cumulative_count / total_tokens
                if coverage >= target_coverage or len(selected_tokens) >= max_vocab_size:
                    break
        
        # Create CONTIGUOUS mappings
        self.compressed_vocab = {}
        self.inverse_mapping = {}
        
        for i, token in enumerate(selected_tokens):
            self.compressed_vocab[token] = i
            self.inverse_mapping[i] = token
        
        # Verify mappings are complete and contiguous
        expected_size = len(selected_tokens)
        if len(self.compressed_vocab) != expected_size:
            raise ValueError(f"Vocab mapping incomplete: expected {expected_size}, got {len(self.compressed_vocab)}")
        
        # Also store token mapping for compatibility
        self.token_mapping = self.compressed_vocab
        
        final_coverage = cumulative_count / total_tokens
        print(f"Compressed vocabulary: {len(selected_tokens)} tokens, {final_coverage:.1%} coverage")
        print(f"FIXED TOKEN MAPPING:")
        print(f"  Token 0 (EOS): {self.inverse_mapping[0]}")
        print(f"  Token 1 (MASK): {self.inverse_mapping[1]}")
        print(f"  Token 2 (PAD): {self.inverse_mapping.get(2, 'N/A')}")
        print(f"Token ID range: [0, {len(selected_tokens)-1}]")
        
        return {
            'vocab': self.compressed_vocab,
            'size': len(selected_tokens),
            'coverage': final_coverage,
            'special_tokens': special_tokens_ordered
        }
    
    def encode(self, text: str) -> List[int]:
        """Encode text using compressed vocabulary"""
        if self.compressed_vocab is None:
            raise ValueError("Must create compressed vocabulary first")
        
        # Tokenize with base tokenizer
        tokens = self.base_tokenizer.tokenize(text)
        
        # Map to compressed vocab with bounds checking
        token_ids = []
        unk_id = self.compressed_vocab.get('[UNK]', 0)  # Fallback to 0 if no UNK
        
        for token in tokens:
            if token in self.compressed_vocab:
                token_id = self.compressed_vocab[token]
                # Ensure token_id is within vocab bounds
                if 0 <= token_id < len(self.compressed_vocab):
                    token_ids.append(token_id)
                else:
                    token_ids.append(unk_id)
            else:
                token_ids.append(unk_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        if self.inverse_mapping is None:
            raise ValueError("Must create compressed vocabulary first")
        
        tokens = [self.inverse_mapping.get(token_id, '[UNK]') for token_id in token_ids]
        return self.base_tokenizer.convert_tokens_to_string(tokens)
    
    def save(self, filepath: str):
        """Save compressed tokenizer"""
        data = {
            'vocab': self.compressed_vocab,
            'token_mapping': self.token_mapping,
            'inverse_mapping': self.inverse_mapping,
            'base_model_name': self.base_tokenizer.name_or_path,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'CompressedTokenizer':
        """Load compressed tokenizer"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(data['base_model_name'])
        tokenizer.compressed_vocab = data['vocab']
        tokenizer.token_mapping = data['token_mapping']
        tokenizer.inverse_mapping = {int(k): v for k, v in data['inverse_mapping'].items()}
        
        return tokenizer


class CurriculumConstructor:
    """Construct 3-stage curriculum from scored segments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.curriculum_config = config.get('curriculum', {})
        self.stages = self.curriculum_config.get('stages', [])
    
    def construct_curriculum(self, segments: List[TextSegment]) -> Dict[str, List[TextSegment]]:
        """
        Construct 3-stage curriculum datasets from scored segments.
        """
        print("Constructing curriculum datasets...")
        
        curriculum_datasets = {}
        
        for stage_config in self.stages:
            stage_name = stage_config['name']
            criteria = stage_config['data_selection_criteria']
            
            # Select segments based on criteria
            selected_segments = self._select_segments_for_stage(segments, criteria)
            
            print(f"Stage {stage_name}: {len(selected_segments)} segments")
            curriculum_datasets[stage_name] = selected_segments
        
        return curriculum_datasets
    
    def _select_segments_for_stage(self, segments: List[TextSegment], criteria: Dict[str, Any]) -> List[TextSegment]:
        """Select segments based on stage criteria"""
        
        # Handle special cases first
        if criteria.get('use_full_corpus', False):
            return segments.copy()
        
        if 'use_first_n_sentences' in criteria:
            n = criteria['use_first_n_sentences']
            return segments[:n]
        
        # Filter by difficulty percentiles
        filtered_segments = segments.copy()
        
        # Syntactic complexity filtering
        if 'syntactic_complexity' in criteria:
            complexity_filter = criteria['syntactic_complexity']
            if complexity_filter == 'bottom_33_percent':
                filtered_segments = self._filter_by_percentile(filtered_segments, 'syntactic_difficulty', 0, 0.33)
            elif complexity_filter == 'bottom_66_percent':
                filtered_segments = self._filter_by_percentile(filtered_segments, 'syntactic_difficulty', 0, 0.66)
            elif complexity_filter == 'bottom_50_percent':
                filtered_segments = self._filter_by_percentile(filtered_segments, 'syntactic_difficulty', 0, 0.50)
            elif complexity_filter == 'bottom_75_percent':
                filtered_segments = self._filter_by_percentile(filtered_segments, 'syntactic_difficulty', 0, 0.75)
        
        # Lexical rarity filtering
        if 'lexical_rarity' in criteria:
            rarity_filter = criteria['lexical_rarity']
            if rarity_filter == 'bottom_33_percent':
                filtered_segments = self._filter_by_percentile(filtered_segments, 'lexical_difficulty', 0, 0.33)
            elif rarity_filter == 'bottom_66_percent':
                filtered_segments = self._filter_by_percentile(filtered_segments, 'lexical_difficulty', 0, 0.66)
        
        # Thematic centrality filtering
        if 'thematic_centrality' in criteria:
            centrality_filter = criteria['thematic_centrality']
            if centrality_filter == 'top_33_percent':
                filtered_segments = self._filter_by_percentile(filtered_segments, 'centrality_score', 0.67, 1.0)
        
        # Length filtering
        if 'min_sentence_length' in criteria:
            min_len = criteria['min_sentence_length']
            filtered_segments = [seg for seg in filtered_segments if seg.length >= min_len]
        
        if 'max_sentence_length' in criteria:
            max_len = criteria['max_sentence_length']
            filtered_segments = [seg for seg in filtered_segments if seg.length <= max_len]
        
        return filtered_segments
    
    def _filter_by_percentile(self, segments: List[TextSegment], score_attr: str, min_percentile: float, max_percentile: float) -> List[TextSegment]:
        """Filter segments by score percentiles"""
        scores = [getattr(seg, score_attr) for seg in segments]
        min_threshold = np.percentile(scores, min_percentile * 100)
        max_threshold = np.percentile(scores, max_percentile * 100)
        
        return [seg for seg in segments if min_threshold <= getattr(seg, score_attr) <= max_threshold]


class DiffusionDataset(Dataset):
    """PyTorch dataset for masked diffusion training"""
    
    def __init__(self, segments: List[TextSegment], tokenizer: CompressedTokenizer, config: Dict[str, Any], stage_config: Dict[str, Any]):
        self.segments = segments
        self.tokenizer = tokenizer
        self.config = config
        self.stage_config = stage_config
        
        self.sequence_length = config.get('data', {}).get('sequence_length', 512)
        self.masking_rate_range = stage_config.get('masking_rate_range', (0.15, 0.15))
        self.training_format = stage_config.get('training_format', 'sentences')
        
        # Prepare formatted data
        self.formatted_data = self._format_data()
    
    def _format_data(self) -> List[Dict[str, Any]]:
        """Format segments according to stage requirements"""
        formatted_data = []
        
        if self.training_format == 'sentences':
            # Individual sentences
            for segment in self.segments:
                token_ids = self.tokenizer.encode(segment.text)
                if len(token_ids) <= self.sequence_length:
                    formatted_data.append({
                        'input_ids': token_ids,
                        'text': segment.text,
                        'segment_idx': segment.index
                    })
        
        elif self.training_format == 'pairs':
            # Create logical pairs (simplified - could be enhanced with actual argument mining)
            for i in range(0, len(self.segments) - 1, 2):
                if i + 1 < len(self.segments):
                    seg1, seg2 = self.segments[i], self.segments[i + 1]
                    combined_text = f"{seg1.text} [SEP] {seg2.text}"
                    token_ids = self.tokenizer.encode(combined_text)
                    
                    if len(token_ids) <= self.sequence_length:
                        formatted_data.append({
                            'input_ids': token_ids,
                            'text': combined_text,
                            'segment_idx': [seg1.index, seg2.index]
                        })
        
        elif self.training_format == 'paragraphs':
            # Combine segments into paragraphs
            current_paragraph = []
            current_length = 0
            
            for segment in self.segments:
                segment_tokens = self.tokenizer.encode(segment.text)
                
                if current_length + len(segment_tokens) <= self.sequence_length:
                    current_paragraph.append(segment)
                    current_length += len(segment_tokens)
                else:
                    # Finalize current paragraph
                    if current_paragraph:
                        paragraph_text = ' '.join([seg.text for seg in current_paragraph])
                        token_ids = self.tokenizer.encode(paragraph_text)
                        
                        formatted_data.append({
                            'input_ids': token_ids,
                            'text': paragraph_text,
                            'segment_idx': [seg.index for seg in current_paragraph]
                        })
                    
                    # Start new paragraph
                    current_paragraph = [segment]
                    current_length = len(segment_tokens)
            
            # Add final paragraph
            if current_paragraph:
                paragraph_text = ' '.join([seg.text for seg in current_paragraph])
                token_ids = self.tokenizer.encode(paragraph_text)
                
                formatted_data.append({
                    'input_ids': token_ids,
                    'text': paragraph_text,
                    'segment_idx': [seg.index for seg in current_paragraph]
                })
        
        return formatted_data
    
    def __len__(self):
        return len(self.formatted_data)
    
    def __getitem__(self, idx):
        item = self.formatted_data[idx]
        input_ids = item['input_ids']
        
        # Get vocab bounds for safe token IDs
        vocab_size = len(self.tokenizer.compressed_vocab)
        
        # CRITICAL FIX: Use correct PAD token ID (position 2, not 0!)
        pad_token_id = self.tokenizer.token_mapping.get('[PAD]', 2)
        if pad_token_id == 0:
            # Fallback if pad token is still at 0 - this should not happen with fixed tokenizer
            print("WARNING: PAD token still at position 0! Using position 2 as fallback.")
            pad_token_id = 2
        
        # Pad to sequence length
        if len(input_ids) < self.sequence_length:
            input_ids = input_ids + [pad_token_id] * (self.sequence_length - len(input_ids))
        else:
            input_ids = input_ids[:self.sequence_length]
        
        # Ensure all token IDs are within vocab bounds
        input_ids = [min(max(0, tid), vocab_size - 1) for tid in input_ids]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token_id != pad_token_id else 0 for token_id in input_ids]
        
        # Dynamic masking with proper bounds
        min_mask, max_mask = self.masking_rate_range
        base_masking_rate = random.uniform(min_mask, max_mask)
        
        # Apply dynamic difficulty adjustment if available
        if hasattr(self, 'attention_difficulty'):
            adaptive_masking_rate = base_masking_rate * self.attention_difficulty
            adaptive_masking_rate = max(0.05, min(0.95, adaptive_masking_rate))
        else:
            adaptive_masking_rate = base_masking_rate
        
        # Create masked version - ONLY mask non-padded tokens
        mask_token_id = self.tokenizer.token_mapping.get('[MASK]', 1)
        masked_input_ids = input_ids.copy()
        labels = [-100] * len(input_ids)  # -100 means ignore in loss
        
        for i in range(len(input_ids)):
            # CRITICAL FIX: Only process non-padded tokens
            if attention_mask[i] == 1 and random.random() < adaptive_masking_rate:
                labels[i] = input_ids[i]  # Store original for loss
                masked_input_ids[i] = mask_token_id  # Replace with mask
            # Padded positions remain -100 in labels (ignored in loss)
        
        # Verify no pad tokens in labels (they should all be -100)
        pad_in_labels = sum(1 for label in labels if label == pad_token_id)
        if pad_in_labels > 0:
            print(f"WARNING: {pad_in_labels} pad tokens found in labels - this will bias training!")
        
        return {
            'input_ids': torch.tensor(masked_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'original_text': item['text']
        }

    def update_attention_difficulty(self, attention_entropy, difficulty_multiplier):
        """Update masking difficulty based on model's attention patterns"""
        self.attention_difficulty = attention_entropy * difficulty_multiplier

class DataPipeline:
    """Complete data processing pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.preprocessor = TextPreprocessor(config)
        self.difficulty_scorer = DifficultyScorer(config)
        self.curriculum_constructor = CurriculumConstructor(config)
        self.tokenizer = None
        
        # Data storage
        self.raw_text = None
        self.segments = None
        self.curriculum_datasets = None
    
    def process_book(self, book_path: str, save_dir: str = "data/processed") -> Dict[str, Any]:
        """
        Complete pipeline: load book -> segment -> score -> construct curriculum.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Processing book: {book_path}")
        
        # 1. Load and preprocess text
        self.raw_text = self.preprocessor.load_text(book_path)
        print(f"Loaded text: {len(self.raw_text):,} characters")
        
        # 2. Segment text
        self.segments = self.preprocessor.segment_text(self.raw_text)
        
        # 3. Create compressed tokenizer
        print("Creating compressed tokenizer...")
        self.tokenizer = CompressedTokenizer()
        vocab_info = self.tokenizer.create_compressed_vocab(
            [seg.text for seg in self.segments],
            target_coverage=self.config.get('data', {}).get('vocab_compression_target', 0.9),
            max_vocab_size=self.config.get('model', {}).get('vocab_size', 25000)
        )
        
        # Save tokenizer
        tokenizer_path = os.path.join(save_dir, "compressed_tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        
        # 4. Score difficulty
        self.segments = self.difficulty_scorer.score_segments(self.segments)
        
        # 5. Construct curriculum
        self.curriculum_datasets = self.curriculum_constructor.construct_curriculum(self.segments)
        
        # 6. Save processed data
        processed_data = {
            'segments': self.segments,
            'curriculum_datasets': self.curriculum_datasets,
            'vocab_info': vocab_info,
            'statistics': self._compute_statistics()
        }
        
        # Save segments
        segments_path = os.path.join(save_dir, "segments.pkl")
        with open(segments_path, 'wb') as f:
            pickle.dump(self.segments, f)
        
        # Save curriculum splits
        curriculum_path = os.path.join(save_dir, "curriculum_splits.pkl")
        with open(curriculum_path, 'wb') as f:
            pickle.dump(self.curriculum_datasets, f)
        
        # Save statistics
        stats_path = os.path.join(save_dir, "statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(processed_data['statistics'], f, indent=2)
        
        print(f"Data processing complete. Saved to: {save_dir}")
        return processed_data
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics"""
        if not self.segments:
            return {}
        
        # Overall statistics
        total_segments = len(self.segments)
        total_words = sum(seg.length for seg in self.segments)
        
        # Difficulty distribution
        lexical_scores = [seg.lexical_difficulty for seg in self.segments]
        syntactic_scores = [seg.syntactic_difficulty for seg in self.segments]
        centrality_scores = [seg.centrality_score for seg in self.segments]
        composite_scores = [seg.composite_difficulty for seg in self.segments]
        
        stats = {
            'total_segments': total_segments,
            'total_words': total_words,
            'avg_segment_length': total_words / total_segments if total_segments > 0 else 0,
            
            'difficulty_distributions': {
                'lexical': {
                    'mean': float(np.mean(lexical_scores)),
                    'std': float(np.std(lexical_scores)),
                    'min': float(np.min(lexical_scores)),
                    'max': float(np.max(lexical_scores))
                },
                'syntactic': {
                    'mean': float(np.mean(syntactic_scores)),
                    'std': float(np.std(syntactic_scores)),
                    'min': float(np.min(syntactic_scores)),
                    'max': float(np.max(syntactic_scores))
                },
                'centrality': {
                    'mean': float(np.mean(centrality_scores)),
                    'std': float(np.std(centrality_scores)),
                    'min': float(np.min(centrality_scores)),
                    'max': float(np.max(centrality_scores))
                },
                'composite': {
                    'mean': float(np.mean(composite_scores)),
                    'std': float(np.std(composite_scores)),
                    'min': float(np.min(composite_scores)),
                    'max': float(np.max(composite_scores))
                }
            },
            
            'curriculum_breakdown': {}
        }
        
        # Curriculum stage statistics
        if self.curriculum_datasets:
            for stage_name, stage_segments in self.curriculum_datasets.items():
                stage_words = sum(seg.length for seg in stage_segments)
                stage_composite_scores = [seg.composite_difficulty for seg in stage_segments]
                
                stats['curriculum_breakdown'][stage_name] = {
                    'num_segments': len(stage_segments),
                    'total_words': stage_words,
                    'avg_segment_length': stage_words / len(stage_segments) if stage_segments else 0,
                    'avg_difficulty': float(np.mean(stage_composite_scores)) if stage_composite_scores else 0,
                    'difficulty_range': [float(np.min(stage_composite_scores)), float(np.max(stage_composite_scores))] if stage_composite_scores else [0, 0]
                }
        
        return stats
    
    def create_dataloaders(self, stage_name: str, batch_size: int = 32, split_ratio: float = 0.9) -> Tuple[DataLoader, DataLoader]:
        """
        Create train/validation dataloaders for a specific curriculum stage.
        """
        if not self.curriculum_datasets or stage_name not in self.curriculum_datasets:
            raise ValueError(f"Curriculum stage '{stage_name}' not found")
        
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Run process_book() first.")
        
        # Get segments for this stage
        stage_segments = self.curriculum_datasets[stage_name]
        
        # Split into train/validation
        random.shuffle(stage_segments)
        split_idx = int(len(stage_segments) * split_ratio)
        train_segments = stage_segments[:split_idx]
        val_segments = stage_segments[split_idx:]
        
        # Get stage configuration
        stage_config = None
        for stage in self.config.get('curriculum', {}).get('stages', []):
            if stage['name'] == stage_name:
                stage_config = stage
                break
        
        if not stage_config:
            raise ValueError(f"Configuration for stage '{stage_name}' not found")
        
        # Create datasets
        train_dataset = DiffusionDataset(train_segments, self.tokenizer, self.config, stage_config)
        val_dataset = DiffusionDataset(val_segments, self.tokenizer, self.config, stage_config)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('system', {}).get('num_workers', 4),
            pin_memory=self.config.get('system', {}).get('pin_memory', True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('system', {}).get('num_workers', 4),
            pin_memory=self.config.get('system', {}).get('pin_memory', True)
        )
        
        return train_loader, val_loader
    
    @classmethod
    def load_processed_data(cls, config: Dict[str, Any], data_dir: str = "data/processed") -> 'DataPipeline':
        """
        Load previously processed data.
        """
        pipeline = cls(config)
        
        # Load segments
        segments_path = os.path.join(data_dir, "segments.pkl")
        with open(segments_path, 'rb') as f:
            pipeline.segments = pickle.load(f)
        
        # Load curriculum datasets
        curriculum_path = os.path.join(data_dir, "curriculum_splits.pkl")
        with open(curriculum_path, 'rb') as f:
            pipeline.curriculum_datasets = pickle.load(f)
        
        # Load tokenizer
        tokenizer_path = os.path.join(data_dir, "compressed_tokenizer.json")
        pipeline.tokenizer = CompressedTokenizer.load(tokenizer_path)
        
        print(f"Loaded processed data from: {data_dir}")
        return pipeline
    
    def print_curriculum_summary(self):
        """Print detailed curriculum summary"""
        if not self.curriculum_datasets:
            print("No curriculum data available")
            return
        
        print("=" * 60)
        print("CURRICULUM SUMMARY")
        print("=" * 60)
        
        total_segments = len(self.segments) if self.segments else 0
        print(f"Total segments: {total_segments:,}")
        
        if self.tokenizer:
            print(f"Vocabulary size: {len(self.tokenizer.compressed_vocab):,}")
        
        print(f"\nStage breakdown:")
        for stage_name, stage_segments in self.curriculum_datasets.items():
            stage_words = sum(seg.length for seg in stage_segments)
            avg_difficulty = np.mean([seg.composite_difficulty for seg in stage_segments]) if stage_segments else 0
            
            print(f"\n  {stage_name.title()}:")
            print(f"    Segments: {len(stage_segments):,} ({len(stage_segments)/total_segments:.1%} of total)")
            print(f"    Words: {stage_words:,}")
            print(f"    Avg difficulty: {avg_difficulty:.3f}")
            
            # Show difficulty distribution
            if stage_segments:
                difficulties = [seg.composite_difficulty for seg in stage_segments]
                print(f"    Difficulty range: {min(difficulties):.3f} - {max(difficulties):.3f}")
        
        print("=" * 60)
    
    def get_sample_texts(self, stage_name: str, num_samples: int = 3) -> List[str]:
        """Get sample texts from a curriculum stage"""
        if stage_name not in self.curriculum_datasets:
            return []
        
        stage_segments = self.curriculum_datasets[stage_name]
        if not stage_segments:
            return []
        
        # Sample segments with different difficulty levels
        difficulties = [seg.composite_difficulty for seg in stage_segments]
        
        samples = []
        if len(stage_segments) >= num_samples:
            # Get low, medium, high difficulty samples
            sorted_segments = sorted(stage_segments, key=lambda x: x.composite_difficulty)
            indices = [0, len(sorted_segments)//2, len(sorted_segments)-1]
            
            for i in indices[:num_samples]:
                samples.append(sorted_segments[i].text)
        else:
            samples = [seg.text for seg in stage_segments[:num_samples]]
        
        return samples


def create_debug_data_pipeline(config: Dict[str, Any], debug_text: str = None) -> DataPipeline:
    """
    Create a minimal data pipeline for debugging/testing.
    Ensures all curriculum stages have at least 1 sample.
    """
    if debug_text is None:
        debug_text = """
        This is a simple sentence for testing. The quick brown fox jumps over the lazy dog.
        Here is another sentence with more complex vocabulary and intricate grammatical structures.
        Scientific research demonstrates the efficacy of iterative methodologies in computational linguistics.
        Machine learning algorithms require substantial computational resources for optimal performance.
        The fundamental principles of natural language processing encompass multiple interdisciplinary domains.
        Complex theoretical frameworks underpin advanced artificial intelligence research paradigms.
        Sophisticated neural network architectures enable unprecedented natural language understanding capabilities.
        Comprehensive evaluation methodologies validate computational linguistics research findings systematically.
        """
    
    # Create temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(debug_text)
        temp_path = f.name
    
    try:
        # Process the debug text
        pipeline = DataPipeline(config)
        pipeline.process_book(temp_path, save_dir="data/debug")
        
        # Fix curriculum distribution to ensure all stages have samples
        _fix_debug_curriculum_distribution(pipeline)
        
        return pipeline
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def _fix_debug_curriculum_distribution(pipeline: DataPipeline):
    """
    Fix curriculum distribution to ensure all stages have at least 1 sample.
    Redistributes segments evenly across all stages.
    """
    if not pipeline.segments or not pipeline.curriculum_datasets:
        return
    
    # Get all segments
    all_segments = pipeline.segments
    num_segments = len(all_segments)
    
    # Get stage names from config
    stages = pipeline.config.get('curriculum', {}).get('stages', [])
    stage_names = [stage['name'] for stage in stages]
    
    if not stage_names:
        return
    
    # Distribute segments evenly across stages
    segments_per_stage = max(1, num_segments // len(stage_names))
    remainder = num_segments % len(stage_names)
    
    # Clear existing curriculum
    pipeline.curriculum_datasets = {}
    
    start_idx = 0
    for i, stage_name in enumerate(stage_names):
        # Calculate how many segments this stage gets
        current_stage_size = segments_per_stage
        if i < remainder:  # Distribute remainder segments to first stages
            current_stage_size += 1
        
        end_idx = start_idx + current_stage_size
        stage_segments = all_segments[start_idx:end_idx]
        
        pipeline.curriculum_datasets[stage_name] = stage_segments
        start_idx = end_idx
    
    # Log the new distribution
    print("Fixed curriculum distribution:")
    for stage_name, segments in pipeline.curriculum_datasets.items():
        print(f"  {stage_name}: {len(segments)} segments")


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'model': {'vocab_size': 5000},
        'data': {
            'sequence_length': 128,
            'min_sentence_length': 5,
            'max_sentence_length': 50,
            'vocab_compression_target': 0.9
        },
        'curriculum': {
            'stages': [
                {
                    'name': 'foundational',
                    'epochs': 1,
                    'masking_rate_range': (0.7, 0.8),
                    'data_selection_criteria': {
                        'syntactic_complexity': 'bottom_33_percent',
                        'lexical_rarity': 'bottom_33_percent',
                        'thematic_centrality': 'top_33_percent'
                    },
                    'training_format': 'sentences'
                },
                {
                    'name': 'structural',
                    'epochs': 1,
                    'masking_rate_range': (0.4, 0.5),
                    'data_selection_criteria': {
                        'syntactic_complexity': 'bottom_66_percent'
                    },
                    'training_format': 'pairs'
                },
                {
                    'name': 'refinement',
                    'epochs': 1,
                    'masking_rate_range': (0.1, 0.2),
                    'data_selection_criteria': {
                        'use_full_corpus': True
                    },
                    'training_format': 'paragraphs'
                }
            ]
        },
        'difficulty_scoring': {
            'num_clusters': 3,
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'system': {
            'num_workers': 2,
            'pin_memory': True
        }
    }
    
    print("Testing data pipeline...")
    
    # Create debug pipeline
    pipeline = create_debug_data_pipeline(test_config)
    
    # Print summary
    pipeline.print_curriculum_summary()
    
    # Test dataloader creation
    for stage_name in ['foundational', 'structural', 'refinement']:
        try:
            train_loader, val_loader = pipeline.create_dataloaders(stage_name, batch_size=2)
            print(f"\n{stage_name} stage:")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
            
            # Test a batch
            for batch in train_loader:
                print(f"  Batch shape: {batch['input_ids'].shape}")
                print(f"  Sample text: {batch['original_text'][0][:100]}...")
                break
        except Exception as e:
            print(f"Error creating dataloaders for {stage_name}: {e}")
    
    print("\nData pipeline test complete!")