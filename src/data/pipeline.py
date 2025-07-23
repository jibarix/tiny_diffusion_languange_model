"""
Enhanced Data Processing Pipeline - Configuration-Driven Version
Text → sentences → multi-dimensional difficulty → curriculum → dataset
All hardcoded values moved to PipelineConfig
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
import math
from collections import defaultdict, Counter

import torch
import nltk
import spacy
import textstat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Import the new configuration
from config import PipelineConfig

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class TextSegment:
    """A text segment with comprehensive difficulty scores"""
    text: str
    index: int
    
    # Static difficulty scores
    lexical_rarity: float = 0.0
    syntactic_complexity: float = 0.0
    thematic_centrality: float = 0.0
    
    # Dynamic difficulty scores (updated during training)
    model_difficulty: float = 0.0  # Based on model loss
    gradient_magnitude: float = 0.0  # Based on gradient norms
    
    # Argumentative structure
    argumentative_role: str = "unknown"  # claim, evidence, warrant, backing, rebuttal
    argumentative_confidence: float = 0.0
    
    # Combined scores
    combined_difficulty: float = 0.0
    stage_assignment: int = 1  # Which curriculum stage
    
    # Metadata
    length: int = 0
    cluster_id: int = -1
    vocabulary_level: int = 1  # For vocabulary curriculum


@dataclass 
class VocabularyCurriculum:
    """Manages progressive vocabulary expansion"""
    core_vocab: List[str] = field(default_factory=list)
    level_vocabs: Dict[int, List[str]] = field(default_factory=dict)
    max_level: int = 5
    
    def get_vocab_for_level(self, level: int) -> List[str]:
        """Get vocabulary tokens up to specified level"""
        vocab = self.core_vocab.copy()
        for i in range(1, min(level + 1, self.max_level + 1)):
            vocab.extend(self.level_vocabs.get(i, []))
        return vocab


class ArgumentMiner:
    """Rule-based argument mining for scientific text"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Patterns for identifying argumentative roles
        self.claim_patterns = [
            r'\b(i argue|i contend|i propose|it follows|therefore|thus|hence)\b',
            r'\b(this shows|this proves|this demonstrates|the evidence shows)\b',
            r'\b(in conclusion|to conclude|we can conclude)\b'
        ]
        
        self.evidence_patterns = [
            r'\b(for example|for instance|such as|namely|specifically)\b',
            r'\b(observations? show|experiments? reveal|data indicates?)\b',
            r'\b(we observe|we find|we see|we note)\b',
            r'\b(in the case of|consider the|examination of)\b'
        ]
        
        self.warrant_patterns = [
            r'\b(because|since|as|given that|considering)\b',
            r'\b(this is due to|this results from|this follows from)\b',
            r'\b(the reason|the cause|the explanation)\b'
        ]
        
        self.backing_patterns = [
            r'\b(according to|research shows|studies indicate)\b',
            r'\b(it is well known|it is established|it is accepted)\b',
            r'\b(previous work|prior research|earlier studies)\b'
        ]
        
        self.rebuttal_patterns = [
            r'\b(however|nevertheless|nonetheless|yet|but)\b',
            r'\b(on the contrary|in contrast|conversely)\b',
            r'\b(some might argue|critics might say|objections include)\b',
            r'\b(although|though|while|whereas)\b'
        ]
    
    def classify_sentence(self, sentence: str) -> Tuple[str, float]:
        """Classify sentence by argumentative role"""
        sentence_lower = sentence.lower()
        
        # Score each role
        scores = {
            'claim': self._count_patterns(sentence_lower, self.claim_patterns),
            'evidence': self._count_patterns(sentence_lower, self.evidence_patterns),
            'warrant': self._count_patterns(sentence_lower, self.warrant_patterns),
            'backing': self._count_patterns(sentence_lower, self.backing_patterns),
            'rebuttal': self._count_patterns(sentence_lower, self.rebuttal_patterns)
        }
        
        # Add heuristics using config values
        if sentence.endswith('?'):
            scores['claim'] += self.config.question_claim_bonus
        if any(word in sentence_lower for word in ['because', 'since']):
            scores['warrant'] += self.config.argument_pattern_bonus
        if any(word in sentence_lower for word in ['data', 'observation', 'experiment']):
            scores['evidence'] += self.config.argument_pattern_bonus
            
        # Find best role
        best_role = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_role] / max(sum(scores.values()), 1.0)
        
        return best_role, max(confidence, self.config.min_argument_confidence)
    
    def _count_patterns(self, text: str, patterns: List[str]) -> float:
        """Count pattern matches in text"""
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text))
        return count


class DynamicDifficultyScorer:
    """Updates difficulty scores based on model training dynamics"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loss_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
    
    def update_segment_difficulty(self, segment_id: int, loss: float, 
                                 gradient_norm: float = None):
        """Update difficulty based on model performance"""
        self.loss_history[segment_id].append(loss)
        if gradient_norm is not None:
            self.gradient_history[segment_id].append(gradient_norm)
    
    def get_model_difficulty(self, segment_id: int) -> float:
        """Get current model-based difficulty score"""
        if segment_id not in self.loss_history:
            return 0.5  # Default neutral difficulty
        
        losses = self.loss_history[segment_id]
        if len(losses) == 0:
            return 0.5
        
        # Use exponential moving average of recent losses
        weights = [self.config.ema_alpha ** (len(losses) - i - 1) for i in range(len(losses))]
        weighted_loss = sum(w * l for w, l in zip(weights, losses)) / sum(weights)
        
        # Normalize using config value
        return min(1.0, weighted_loss / self.config.loss_normalization_factor)
    
    def get_gradient_difficulty(self, segment_id: int) -> float:
        """Get gradient-based difficulty score"""
        if segment_id not in self.gradient_history:
            return 0.5
        
        grads = self.gradient_history[segment_id]
        if len(grads) == 0:
            return 0.5
        
        # Higher gradient norm = model learning more = higher difficulty
        avg_grad = np.mean(grads[-5:])  # Use recent gradient norms
        return min(1.0, avg_grad / self.config.gradient_normalization_factor)


class TextDataPipeline:
    """Enhanced pipeline with configuration-driven processing"""
    
    def __init__(self, 
                 pipeline_config: PipelineConfig = None,
                 embedding_model: str = None,
                 n_clusters: int = None,
                 target_vocab_size: int = 25000,
                 enable_argument_mining: bool = True,
                 enable_vocab_curriculum: bool = True):
        
        # Use provided config or create default
        self.config = pipeline_config or PipelineConfig.default()
        
        # Override config with explicit parameters if provided
        if embedding_model:
            self.embedding_model_name = embedding_model
        else:
            self.embedding_model_name = self.config.embedding_model_default
            
        if n_clusters:
            self.n_clusters = n_clusters
        else:
            self.n_clusters = self.config.default_n_clusters
            
        self.target_vocab_size = target_vocab_size
        self.enable_argument_mining = enable_argument_mining
        self.enable_vocab_curriculum = enable_vocab_curriculum
        
        # Initialize models
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.tokenizer = None
        
        # Advanced components (now config-driven)
        self.argument_miner = ArgumentMiner(self.config) if enable_argument_mining else None
        self.dynamic_scorer = DynamicDifficultyScorer(self.config)
        self.vocab_curriculum = VocabularyCurriculum() if enable_vocab_curriculum else None
        
        # Data storage
        self.segments: List[TextSegment] = []
        self.embeddings: np.ndarray = None
        self.cluster_model = None
        
        # Curriculum weights (adjustable via config in future versions)
        self.lexical_weight = 0.25
        self.syntactic_weight = 0.25
        self.centrality_weight = 0.25
        self.argument_weight = 0.15
        self.dynamic_weight = 0.10
    
    def load_text(self, file_path: str) -> str:
        """Load and clean text from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Basic cleaning
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def segment_text(self, text: str) -> List[str]:
        """Segment text into sentences using config thresholds"""
        sentences = nltk.sent_tokenize(text)
        
        # Filter and clean using config values
        cleaned_sentences = []
        for s in sentences:
            s = s.strip()
            if (len(s) > self.config.min_sentence_length and 
                len(s.split()) >= self.config.min_sentence_words):
                cleaned_sentences.append(s)
        
        return cleaned_sentences
    
    def create_vocabulary_curriculum(self, texts: List[str]) -> VocabularyCurriculum:
        """Create progressive vocabulary curriculum"""
        if not self.enable_vocab_curriculum:
            return None
        
        # Create base tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Analyze vocabulary frequency in the corpus
        all_tokens = []
        for text in texts:
            tokens = base_tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend([base_tokenizer.decode([t]) for t in tokens])
        
        # Count frequencies
        token_freq = Counter(all_tokens) 
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Create curriculum levels
        vocab_curriculum = VocabularyCurriculum()
        
        # Core vocabulary using config fraction
        core_size = int(len(sorted_tokens) * self.config.vocab_core_fraction)
        vocab_curriculum.core_vocab = [token for token, _ in sorted_tokens[:core_size]]
        
        # Progressive levels
        remaining_tokens = sorted_tokens[core_size:]
        level_size = len(remaining_tokens) // vocab_curriculum.max_level
        
        for level in range(1, vocab_curriculum.max_level + 1):
            start_idx = (level - 1) * level_size
            end_idx = start_idx + level_size if level < vocab_curriculum.max_level else len(remaining_tokens)
            vocab_curriculum.level_vocabs[level] = [
                token for token, _ in remaining_tokens[start_idx:end_idx]
            ]
        
        return vocab_curriculum
    
    def create_adaptive_tokenizer(self, texts: List[str], vocab_level: int = 1) -> AutoTokenizer:
        """Create tokenizer with vocabulary filtered by level"""
        base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        if not self.vocab_curriculum:
            return self._add_special_tokens(base_tokenizer)
        
        # Get allowed vocabulary for this level
        allowed_vocab = set(self.vocab_curriculum.get_vocab_for_level(vocab_level))
        
        # Calculate actual vocabulary size for this level
        core_size = len(self.vocab_curriculum.core_vocab)
        level_additions = sum(len(self.vocab_curriculum.level_vocabs.get(i, [])) 
                            for i in range(1, vocab_level + 1))
        target_vocab_size = min(core_size + level_additions, self.target_vocab_size)
        
        # Analyze token frequencies in corpus
        all_tokens = []
        for text in texts:
            tokens = base_tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        token_freq = Counter(all_tokens)
        
        # Get most frequent tokens up to target size
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Always include special tokens
        special_token_ids = set()
        special_tokens_dict = {}
        
        if base_tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<pad>"
        if base_tokenizer.mask_token is None:
            special_tokens_dict["mask_token"] = "<mask>"
        if base_tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<bos>"
        if base_tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "<eos>"
        
        # Add special tokens first
        if special_tokens_dict:
            base_tokenizer.add_special_tokens(special_tokens_dict)
        
        # Get special token IDs
        special_token_ids.update([
            base_tokenizer.pad_token_id,
            base_tokenizer.mask_token_id,
            base_tokenizer.bos_token_id,
            base_tokenizer.eos_token_id
        ])
        special_token_ids.discard(None)
        
        # Select tokens for this vocabulary level
        selected_tokens = set(special_token_ids)
        
        # Add most frequent tokens up to target size
        for token_id, freq in sorted_tokens:
            if len(selected_tokens) >= target_vocab_size:
                break
            
            # Skip if already added
            if token_id in selected_tokens:
                continue
                
            # For vocabulary curriculum: check if token is allowed at this level
            if vocab_level < self.vocab_curriculum.max_level:
                token_text = base_tokenizer.decode([token_id])
                if token_text not in allowed_vocab and len(selected_tokens) > core_size:
                    continue
            
            selected_tokens.add(token_id)
        
        # Create mapping: special tokens get fixed positions, then frequent tokens
        old_to_new_mapping = {}
        new_to_old_mapping = {}
        special_token_mapping = {}
        
        # Map special tokens to positions 0, 1, 2, etc.
        next_id = 0
        for attr_name in ['pad_token_id', 'mask_token_id', 'bos_token_id', 'eos_token_id', 'unk_token_id']:
            original_id = getattr(base_tokenizer, attr_name, None)
            if original_id is not None and original_id in selected_tokens:
                special_token_mapping[original_id] = next_id
                old_to_new_mapping[original_id] = next_id
                new_to_old_mapping[next_id] = original_id
                next_id += 1
        
        # Then assign remaining tokens
        current_new_id = len(special_token_mapping)
        remaining_tokens = sorted([t for t in selected_tokens if t not in special_token_mapping])
        
        for old_id in remaining_tokens:
            old_to_new_mapping[old_id] = current_new_id
            new_to_old_mapping[current_new_id] = old_id
            current_new_id += 1
        
        # Create new vocabulary
        new_vocab = {}
        for new_id, old_id in new_to_old_mapping.items():
            token_text = base_tokenizer.decode([old_id])
            new_vocab[token_text] = new_id
        
        # Create FilteredTokenizer class
        class FilteredTokenizer:
            def __init__(self, base_tokenizer, old_to_new, new_to_old, new_vocab, special_mapping):
                self.base_tokenizer = base_tokenizer
                self.old_to_new = old_to_new
                self.new_to_old = new_to_old
                self.new_vocab = new_vocab
                self.vocab_size = len(new_vocab)
                
                # Set special token IDs using guaranteed mappings
                original_pad_id = base_tokenizer.pad_token_id
                original_mask_id = base_tokenizer.mask_token_id
                original_bos_id = base_tokenizer.bos_token_id
                original_eos_id = base_tokenizer.eos_token_id
                original_unk_id = base_tokenizer.unk_token_id
                
                self.pad_token_id = special_mapping.get(original_pad_id, 0)
                self.mask_token_id = special_mapping.get(original_mask_id, 1)
                self.bos_token_id = special_mapping.get(original_bos_id) if original_bos_id else None
                self.eos_token_id = special_mapping.get(original_eos_id) if original_eos_id else None
                self.unk_token_id = special_mapping.get(original_unk_id, 0) if original_unk_id else 0
                
                # Validate all special token IDs are within vocabulary
                for attr_name in ['pad_token_id', 'mask_token_id', 'bos_token_id', 'eos_token_id', 'unk_token_id']:
                    token_id = getattr(self, attr_name)
                    if token_id is not None and token_id >= self.vocab_size:
                        raise ValueError(f"Special token {attr_name} is outside vocabulary range")
                
                # Set special token strings
                self.pad_token = base_tokenizer.pad_token
                self.mask_token = base_tokenizer.mask_token
                self.bos_token = base_tokenizer.bos_token
                self.eos_token = base_tokenizer.eos_token
                self.unk_token = base_tokenizer.unk_token
            
            def encode(self, text, add_special_tokens=True, **kwargs):
                # Encode with base tokenizer then map to filtered vocab
                base_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
                filtered_ids = []
                
                for token_id in base_ids:
                    if token_id in self.old_to_new:
                        filtered_ids.append(self.old_to_new[token_id])
                    else:
                        # Map unknown tokens to UNK
                        filtered_ids.append(self.unk_token_id)
                
                return filtered_ids
            
            def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                # Map back to base vocabulary for decoding
                base_ids = []
                for token_id in token_ids:
                    if token_id in self.new_to_old:
                        base_ids.append(self.new_to_old[token_id])
                    else:
                        # Handle out-of-vocab tokens
                        base_ids.append(self.base_tokenizer.unk_token_id or 0)
                
                return self.base_tokenizer.decode(base_ids, skip_special_tokens=skip_special_tokens, **kwargs)
            
            def save_pretrained(self, save_directory):
                """Save filtered tokenizer"""
                from pathlib import Path
                save_path = Path(save_directory)
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Save base tokenizer
                self.base_tokenizer.save_pretrained(save_path)
                
                # Save filtering information
                import json
                filter_info = {
                    'vocab_level': vocab_level,
                    'vocab_size': self.vocab_size,
                    'old_to_new_mapping': {str(k): v for k, v in self.old_to_new.items()},
                    'new_to_old_mapping': {str(k): v for k, v in self.new_to_old.items()},
                    'filtered_vocab': self.new_vocab,
                    'target_vocab_size': target_vocab_size,
                    'special_token_mapping': {str(k): v for k, v in special_mapping.items()}
                }
                
                with open(save_path / 'vocab_filter.json', 'w') as f:
                    json.dump(filter_info, f, indent=2)
                
                print(f"Filtered tokenizer saved to: {save_path}")
            
            def __len__(self):
                return self.vocab_size
        
        # Create and return filtered tokenizer
        filtered_tokenizer = FilteredTokenizer(
            base_tokenizer, old_to_new_mapping, new_to_old_mapping, new_vocab, special_token_mapping
        )
        
        print(f"Created tokenizer for vocab level {vocab_level}: {len(filtered_tokenizer):,} tokens")
        return filtered_tokenizer
    
    def _add_special_tokens(self, tokenizer: AutoTokenizer) -> AutoTokenizer:
        """Add special tokens to tokenizer"""
        special_tokens = {
            "pad_token": "<pad>",
            "mask_token": "<mask>", 
            "bos_token": "<bos>",
            "eos_token": "<eos>"
        }
        
        tokens_to_add = {}
        for key, token in special_tokens.items():
            if getattr(tokenizer, key) is None:
                tokens_to_add[key] = token
        
        if tokens_to_add:
            tokenizer.add_special_tokens(tokens_to_add)
            print(f"✅ Added special tokens: {list(tokens_to_add.keys())}")
        
        return tokenizer
    
    def _compress_tokenizer(self, tokenizer: AutoTokenizer, texts: List[str]) -> AutoTokenizer:
        """Compress tokenizer vocabulary based on corpus frequency"""
        from collections import Counter
        import json
        
        print(f"🔧 Compressing vocabulary from {len(tokenizer):,} to {self.target_vocab_size:,} tokens...")
        
        # Analyze token frequencies in the corpus
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        if not all_tokens:
            print("⚠️  No tokens found in corpus, returning original tokenizer")
            return tokenizer
        
        token_freq = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Always include special tokens first
        special_token_ids = []
        special_token_mapping = {}
        
        # Map special tokens to positions 0, 1, 2, etc.
        next_id = 0
        for attr_name in ['pad_token_id', 'mask_token_id', 'bos_token_id', 'eos_token_id', 'unk_token_id']:
            original_id = getattr(tokenizer, attr_name, None)
            if original_id is not None:
                special_token_ids.append(original_id)
                special_token_mapping[original_id] = next_id
                next_id += 1
        
        print(f"🔧 Special tokens mapped: {len(special_token_mapping)} tokens")
        
        # Sort remaining tokens by frequency
        sorted_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Select tokens to keep
        selected_tokens = set(special_token_ids)
        cumulative_freq = 0
        
        # Add most frequent tokens until we hit target size or coverage threshold
        for token_id, freq in sorted_tokens:
            if token_id in selected_tokens:
                continue
                
            if len(selected_tokens) >= self.target_vocab_size:
                break
                
            selected_tokens.add(token_id)
            cumulative_freq += freq
            
            # Stop if we've covered enough of the corpus and have viable size
            if (cumulative_freq >= self.config.corpus_coverage_threshold * total_tokens and 
                len(selected_tokens) >= self.config.min_compressed_vocab_size):
                break
        
        # Create mapping: special tokens get fixed positions, then frequent tokens
        old_to_new_mapping = {}
        new_to_old_mapping = {}
        
        # First, assign special tokens to fixed positions
        for original_id, new_id in special_token_mapping.items():
            old_to_new_mapping[original_id] = new_id
            new_to_old_mapping[new_id] = original_id
        
        # Then assign remaining tokens
        current_new_id = len(special_token_mapping)
        remaining_tokens = sorted([t for t in selected_tokens if t not in special_token_mapping])
        
        for old_id in remaining_tokens:
            old_to_new_mapping[old_id] = current_new_id
            new_to_old_mapping[current_new_id] = old_id
            current_new_id += 1
        
        # Create new vocabulary
        new_vocab = {}
        for new_id, old_id in new_to_old_mapping.items():
            token_text = tokenizer.decode([old_id])
            new_vocab[token_text] = new_id
        
        # Create compressed tokenizer class (same implementation as before)
        class CompressedTokenizer:
            def __init__(self, base_tokenizer, old_to_new, new_to_old, new_vocab, special_mapping):
                self.base_tokenizer = base_tokenizer
                self.old_to_new = old_to_new
                self.new_to_old = new_to_old
                self.new_vocab = new_vocab
                self.vocab_size = len(new_vocab)
                
                # Set special token IDs using guaranteed mappings
                original_pad_id = base_tokenizer.pad_token_id
                original_mask_id = base_tokenizer.mask_token_id
                original_bos_id = base_tokenizer.bos_token_id
                original_eos_id = base_tokenizer.eos_token_id
                original_unk_id = base_tokenizer.unk_token_id
                
                self.pad_token_id = special_mapping.get(original_pad_id, 0)
                self.mask_token_id = special_mapping.get(original_mask_id, 1)
                self.bos_token_id = special_mapping.get(original_bos_id) if original_bos_id else None
                self.eos_token_id = special_mapping.get(original_eos_id) if original_eos_id else None
                self.unk_token_id = special_mapping.get(original_unk_id, 0) if original_unk_id else 0
                
                # Validate all special token IDs are within vocabulary
                for attr_name in ['pad_token_id', 'mask_token_id', 'bos_token_id', 'eos_token_id', 'unk_token_id']:
                    token_id = getattr(self, attr_name)
                    if token_id is not None and token_id >= self.vocab_size:
                        raise ValueError(f"Special token {attr_name} is outside vocabulary range")
                
                # Set special token strings
                self.pad_token = base_tokenizer.pad_token
                self.mask_token = base_tokenizer.mask_token
                self.bos_token = base_tokenizer.bos_token
                self.eos_token = base_tokenizer.eos_token
                self.unk_token = base_tokenizer.unk_token
            
            def encode(self, text, add_special_tokens=True, **kwargs):
                base_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
                compressed_ids = []
                
                for token_id in base_ids:
                    if token_id in self.old_to_new:
                        compressed_ids.append(self.old_to_new[token_id])
                    else:
                        compressed_ids.append(self.unk_token_id)
                
                return compressed_ids
            
            def decode(self, token_ids, skip_special_tokens=True, **kwargs):
                base_ids = []
                for token_id in token_ids:
                    if token_id in self.new_to_old:
                        base_ids.append(self.new_to_old[token_id])
                    else:
                        base_ids.append(self.base_tokenizer.unk_token_id or 0)
                
                return self.base_tokenizer.decode(base_ids, skip_special_tokens=skip_special_tokens, **kwargs)
            
            def save_pretrained(self, save_directory):
                from pathlib import Path
                save_path = Path(save_directory)
                save_path.mkdir(parents=True, exist_ok=True)
                
                # Save base tokenizer
                self.base_tokenizer.save_pretrained(save_path)
                
                # Save compression mapping
                compression_info = {
                    'vocab_size': self.vocab_size,
                    'target_vocab_size': self.vocab_size,
                    'old_to_new_mapping': {str(k): v for k, v in self.old_to_new.items()},
                    'new_to_old_mapping': {str(k): v for k, v in self.new_to_old.items()},
                    'new_vocab': self.new_vocab,
                    'special_tokens': {
                        'pad_token_id': self.pad_token_id,
                        'mask_token_id': self.mask_token_id,
                        'bos_token_id': self.bos_token_id,
                        'eos_token_id': self.eos_token_id,
                        'unk_token_id': self.unk_token_id
                    }
                }
                
                with open(save_path / 'compression_info.json', 'w') as f:
                    json.dump(compression_info, f, indent=2)
            
            def __len__(self):
                return self.vocab_size
        
        # Create and return compressed tokenizer
        compressed_tokenizer = CompressedTokenizer(
            tokenizer, old_to_new_mapping, new_to_old_mapping, new_vocab, special_token_mapping
        )
        
        coverage = cumulative_freq / total_tokens
        compression_ratio = len(selected_tokens) / len(tokenizer)
        
        print(f"✅ Vocabulary compressed:")
        print(f"   Original size: {len(tokenizer):,} tokens")
        print(f"   Compressed size: {len(selected_tokens):,} tokens")
        print(f"   Compression ratio: {compression_ratio:.1%}")
        print(f"   Corpus coverage: {coverage:.1%}")
        
        return compressed_tokenizer
    
    def calculate_lexical_difficulty(self, segments: List[str]) -> List[float]:
        """Calculate lexical rarity scores using config parameters"""
        if self.tokenizer is None:
            self.tokenizer = self.create_adaptive_tokenizer(segments)
        
        scores = []
        for text in segments:
            # Calculate average word rarity using config normalization
            words = re.findall(r'\b\w+\b', text.lower())
            if not words:
                scores.append(0.0)
                continue
            
            # Simple heuristic: longer words = rarer/harder
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Normalize using config value
            lexical_score = min(1.0, avg_word_length / self.config.lexical_max_word_length)
            scores.append(lexical_score)
        
        return scores
    
    def calculate_syntactic_complexity(self, segments: List[str]) -> List[float]:
        """Calculate syntactic complexity using config parameters"""
        scores = []
        for text in segments:
            try:
                # Multiple complexity measures using config values
                flesch = textstat.flesch_reading_ease(text)
                flesch_score = max(0, (self.config.flesch_score_base - flesch) / self.config.flesch_score_base)
                
                # Sentence length penalty using config
                words = text.split()
                length_score = min(1.0, len(words) / self.config.sentence_length_max_words)
                
                # Subordinate clause detection using config
                subordinate_count = sum(1 for marker in self.config.subordinate_markers 
                                      if marker in text.lower())
                subordinate_score = min(1.0, subordinate_count / self.config.subordinate_clause_max_count)
                
                # Combine scores
                complexity = (flesch_score + length_score + subordinate_score) / 3.0
                
            except Exception:
                complexity = 0.5  # Default moderate complexity
            
            scores.append(complexity)
        
        return scores
    
    def calculate_thematic_centrality(self, segments: List[str]) -> Tuple[List[float], List[int]]:
        """Calculate thematic centrality using config parameters"""
        print("Calculating sentence embeddings...")
        embeddings = self.embedding_model.encode(segments, show_progress_bar=True)
        self.embeddings = embeddings
        
        print(f"Clustering into {self.n_clusters} thematic groups...")
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(embeddings_scaled)
        
        # Calculate centrality scores using config
        centrality_scores = []
        for i, embedding in enumerate(embeddings):
            cluster_id = cluster_labels[i]
            cluster_center = self.cluster_model.cluster_centers_[cluster_id]
            
            # Distance to cluster center (inverted for centrality) using config
            distance = np.linalg.norm(embeddings_scaled[i] - cluster_center)
            centrality = max(0, 1 - distance / self.config.centrality_distance_normalizer)
            centrality_scores.append(centrality)
        
        return centrality_scores, cluster_labels
    
    def calculate_argumentative_difficulty(self, segments: List[str]) -> Tuple[List[str], List[float]]:
        """Calculate argumentative structure complexity"""
        if not self.argument_miner:
            return ['unknown'] * len(segments), [0.5] * len(segments)
        
        roles = []
        confidences = []
        
        for segment in segments:
            role, confidence = self.argument_miner.classify_sentence(segment)
            roles.append(role)
            confidences.append(confidence)
        
        return roles, confidences
    
    def assign_vocabulary_levels(self, segments: List[str]) -> List[int]:
        """Assign vocabulary curriculum levels using config thresholds"""
        if not self.vocab_curriculum:
            return [1] * len(segments)
        
        levels = []
        for segment in segments:
            # Simple heuristic: rare words → higher level
            words = re.findall(r'\b\w+\b', segment.lower())
            rare_word_count = 0
            
            for word in words:
                if word not in self.vocab_curriculum.core_vocab:
                    rare_word_count += 1
            
            # Assign level based on rare word density using config thresholds
            rare_ratio = rare_word_count / max(len(words), 1)
            level = 1
            
            for i, threshold in enumerate(self.config.vocab_rare_ratio_thresholds):
                if rare_ratio >= threshold:
                    level = i + 2  # Start from level 2
                else:
                    break
            
            # Cap at max level using config
            if rare_ratio >= self.config.vocab_rare_ratio_thresholds[-1]:
                level = min(self.config.vocab_level_multiplier, int(rare_ratio * self.config.vocab_level_multiplier))
            
            levels.append(min(level, 5))  # Max level 5
        
        return levels
    
    def combine_difficulty_scores(self, segment: TextSegment) -> float:
        """Combine all difficulty dimensions into final score"""
        static_score = (
            self.lexical_weight * segment.lexical_rarity +
            self.syntactic_weight * segment.syntactic_complexity +
            self.centrality_weight * (1 - segment.thematic_centrality) +  # Invert centrality
            self.argument_weight * (1 - segment.argumentative_confidence)  # Lower confidence = harder
        )
        
        dynamic_score = (
            segment.model_difficulty * 0.6 +
            segment.gradient_magnitude * 0.4
        )
        
        final_score = (1 - self.dynamic_weight) * static_score + self.dynamic_weight * dynamic_score
        return final_score
    
    def assign_curriculum_stages(self, segments: List[TextSegment]) -> None:
        """Assign segments to curriculum stages using config percentiles"""
        # Sort by combined difficulty
        difficulties = [s.combined_difficulty for s in segments]
        
        # Calculate percentile thresholds using config
        easy_threshold = np.percentile(difficulties, self.config.easy_difficulty_percentile)
        hard_threshold = np.percentile(difficulties, self.config.hard_difficulty_percentile)
        
        for segment in segments:
            if segment.combined_difficulty <= easy_threshold:
                segment.stage_assignment = 1  # Foundation
            elif segment.combined_difficulty <= hard_threshold:
                segment.stage_assignment = 2  # Structural
            else:
                segment.stage_assignment = 3  # Refinement
    
    def create_argument_pairs(self, segments: List[TextSegment]) -> List[Tuple[TextSegment, TextSegment]]:
        """Create evidence-claim pairs for Stage II training"""
        pairs = []
        
        # Group by argumentative roles
        evidence_segments = [s for s in segments if s.argumentative_role == 'evidence']
        claim_segments = [s for s in segments if s.argumentative_role == 'claim']
        warrant_segments = [s for s in segments if s.argumentative_role == 'warrant']
        
        # Create evidence→claim pairs
        for evidence in evidence_segments:
            # Find closest claim in embedding space
            best_claim = None
            best_similarity = -1
            
            for claim in claim_segments:
                if self.embeddings is not None:
                    similarity = np.dot(self.embeddings[evidence.index], 
                                      self.embeddings[claim.index])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_claim = claim
            
            if best_claim:
                pairs.append((evidence, best_claim))
        
        # Create warrant→claim pairs
        for warrant in warrant_segments:
            best_claim = None
            best_similarity = -1
            
            for claim in claim_segments:
                if self.embeddings is not None:
                    similarity = np.dot(self.embeddings[warrant.index], 
                                      self.embeddings[claim.index])
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_claim = claim
            
            if best_claim:
                pairs.append((warrant, best_claim))
        
        return pairs
    
    def generate_pseudo_data(self, model, stage_2_segments: List[TextSegment], 
                           num_samples: int = None) -> List[str]:
        """Generate pseudo-data for Stage III self-training using actual model"""
        if num_samples is None:
            num_samples = self.config.max_pseudo_samples_default
        
        if model is None:
            print("Warning: No model provided for pseudo-data generation, using simple paraphrases")
            return self._generate_simple_paraphrases(stage_2_segments, num_samples)
        
        pseudo_texts = []
        model.eval()
        
        with torch.no_grad():
            for segment in stage_2_segments[:num_samples]:
                original_text = segment.text
                
                # Generate multiple variations using the model
                variations = self._generate_model_variations(model, original_text)
                
                # Filter variations for quality
                high_quality_variations = self._filter_pseudo_data_quality(
                    original_text, variations
                )
                
                pseudo_texts.extend(high_quality_variations)
                
                # Limit total pseudo samples
                if len(pseudo_texts) >= num_samples:
                    break
        
        model.train()  # Reset to training mode
        return pseudo_texts[:num_samples]
    
    def _generate_model_variations(self, model, original_text: str, 
                                 num_variations: int = 3) -> List[str]:
        """Generate variations using the trained model"""
        variations = []
        
        try:
            # Tokenize the original text
            tokens = self.tokenizer.encode(original_text, add_special_tokens=True)
            
            for _ in range(num_variations):
                # Create masked versions with different masking patterns
                masked_tokens = self._create_pseudo_data_masks(tokens)
                
                # Convert to tensor
                input_ids = torch.tensor([masked_tokens])
                attention_mask = torch.ones_like(input_ids)
                
                # Generate using the model
                outputs = model(input_ids, attention_mask, masking_rate=0.0)  # No additional masking
                logits = outputs['logits']
                
                # Sample from the distribution
                probs = torch.softmax(logits, dim=-1)
                sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(input_ids.shape)
                
                # Only update masked positions
                mask_positions = (input_ids == model.mask_token_id)
                generated_ids = input_ids.clone()
                generated_ids[mask_positions] = sampled_tokens[mask_positions]
                
                # Decode the result
                generated_text = self.tokenizer.decode(
                    generated_ids[0].tolist(), 
                    skip_special_tokens=True
                )
                
                if generated_text.strip() and generated_text != original_text:
                    variations.append(generated_text.strip())
        
        except Exception as e:
            print(f"Warning: Model generation failed: {e}, falling back to paraphrases")
            variations = self._generate_simple_paraphrases([type('', (), {'text': original_text})()], 3)
        
        return variations
    
    def _create_pseudo_data_masks(self, tokens: List[int]) -> List[int]:
        """Create strategic masks for pseudo-data generation"""
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            return tokens
        
        masked_tokens = tokens.copy()
        
        # Mask 20-40% of tokens strategically
        mask_rate = np.random.uniform(0.2, 0.4)
        n_mask = int(len(tokens) * mask_rate)
        
        # Don't mask special tokens
        maskable_positions = []
        for i, token_id in enumerate(tokens):
            if (token_id not in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, 
                               self.tokenizer.eos_token_id] and i > 0 and i < len(tokens) - 1):
                maskable_positions.append(i)
        
        if maskable_positions and n_mask > 0:
            # Randomly select positions to mask
            mask_positions = np.random.choice(
                maskable_positions, 
                size=min(n_mask, len(maskable_positions)), 
                replace=False
            )
            
            for pos in mask_positions:
                masked_tokens[pos] = self.tokenizer.mask_token_id
        
        return masked_tokens
    
    def _filter_pseudo_data_quality(self, original_text: str, 
                                   variations: List[str]) -> List[str]:
        """Filter pseudo-data for quality using multiple criteria"""
        if not variations:
            return []
        
        high_quality = []
        
        for variation in variations:
            if self._is_high_quality_pseudo_data(original_text, variation):
                high_quality.append(variation)
        
        return high_quality
    
    def _is_high_quality_pseudo_data(self, original: str, variation: str) -> bool:
        """Check if generated variation meets quality criteria"""
        # Basic sanity checks
        if not variation or variation == original:
            return False
        
        # Length similarity check
        orig_len = len(original.split())
        var_len = len(variation.split())
        length_ratio = min(orig_len, var_len) / max(orig_len, var_len)
        
        if length_ratio < 0.7:  # Too different in length
            return False
        
        # Semantic similarity check using embeddings
        if hasattr(self, 'embedding_model') and self.embedding_model:
            try:
                orig_embedding = self.embedding_model.encode([original])
                var_embedding = self.embedding_model.encode([variation])
                
                # Calculate cosine similarity
                similarity = np.dot(orig_embedding[0], var_embedding[0]) / (
                    np.linalg.norm(orig_embedding[0]) * np.linalg.norm(var_embedding[0])
                )
                
                # Require reasonable semantic similarity (configurable threshold)
                min_similarity = getattr(self.config, 'pseudo_data_min_similarity', 0.7)
                if similarity < min_similarity:
                    return False
                    
            except Exception as e:
                print(f"Warning: Semantic similarity check failed: {e}")
                # Fall back to lexical overlap check
                orig_words = set(original.lower().split())
                var_words = set(variation.lower().split())
                overlap = len(orig_words & var_words) / len(orig_words | var_words)
                
                if overlap < 0.5:  # Require 50% word overlap
                    return False
        
        # Check for reasonable vocabulary (no excessive repetition)
        words = variation.split()
        unique_words = len(set(words))
        if len(words) > 5 and unique_words / len(words) < 0.6:  # Too repetitive
            return False
        
        # Check for basic grammatical structure (has both subjects and predicates)
        if len(words) > 3:
            # Simple heuristic: should have some common function words
            function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'have', 'has', 'had'}
            if not any(word.lower() in function_words for word in words):
                return False
        
        return True
    
    def _generate_simple_paraphrases(self, segments: List[TextSegment], 
                                   num_samples: int) -> List[str]:
        """Fallback: Generate simple paraphrases when model is unavailable"""
        pseudo_texts = []
        
        # More sophisticated paraphrasing patterns
        paraphrase_patterns = [
            # Verb replacements
            (' is ', ' appears to be '), (' is ', ' seems to be '), (' is ', ' represents '),
            (' shows ', ' demonstrates '), (' shows ', ' indicates '), (' shows ', ' reveals '),
            (' proves ', ' establishes '), (' proves ', ' confirms '), (' proves ', ' validates '),
            (' because ', ' since '), (' because ', ' due to the fact that '), (' because ', ' as '),
            (' however ', ' nevertheless '), (' however ', ' nonetheless '), (' however ', ' yet '),
            (' therefore ', ' thus '), (' therefore ', ' consequently '), (' therefore ', ' hence '),
            
            # Adjective replacements  
            (' important ', ' significant '), (' important ', ' crucial '), (' important ', ' vital '),
            (' large ', ' substantial '), (' large ', ' considerable '), (' large ', ' extensive '),
            (' small ', ' minor '), (' small ', ' limited '), (' small ', ' modest '),
            
            # Noun replacements
            (' evidence ', ' data '), (' evidence ', ' findings '), (' evidence ', ' observations '),
            (' result ', ' outcome '), (' result ', ' consequence '), (' result ', ' effect '),
            (' method ', ' approach '), (' method ', ' technique '), (' method ', ' procedure '),
        ]
        
        for segment in segments[:num_samples]:
            original = segment.text
            
            # Apply multiple paraphrase patterns
            for pattern_set in [paraphrase_patterns[i:i+3] for i in range(0, len(paraphrase_patterns), 3)]:
                for old_phrase, new_phrase in pattern_set:
                    if old_phrase in original:
                        variation = original.replace(old_phrase, new_phrase)
                        
                        # Quality check for simple paraphrases
                        if (variation != original and 
                            len(variation) > len(original) * 0.8 and
                            len(variation) < len(original) * 1.3):
                            pseudo_texts.append(variation)
                            break
                
                if len(pseudo_texts) >= num_samples:
                    break
            
            if len(pseudo_texts) >= num_samples:
                break
        
        return pseudo_texts[:num_samples]
    
    def update_dynamic_scores(self, segment_ids: List[int], losses: List[float], 
                            gradient_norms: List[float] = None):
        """Update dynamic difficulty scores based on training feedback"""
        for i, segment_id in enumerate(segment_ids):
            loss = losses[i]
            grad_norm = gradient_norms[i] if gradient_norms else None
            
            # Update dynamic scorer
            self.dynamic_scorer.update_segment_difficulty(segment_id, loss, grad_norm)
            
            # Update segment scores
            if segment_id < len(self.segments):
                segment = self.segments[segment_id]
                segment.model_difficulty = self.dynamic_scorer.get_model_difficulty(segment_id)
                if grad_norm:
                    segment.gradient_magnitude = min(1.0, grad_norm)
                
                # Recalculate combined difficulty
                segment.combined_difficulty = self.combine_difficulty_scores(segment)
    
    def process_text_file(self, file_path: str) -> List[TextSegment]:
        """Main processing pipeline with configuration-driven processing"""
        print(f"Processing: {file_path}")
        
        # Load and segment text
        text = self.load_text(file_path)
        sentences = self.segment_text(text)
        print(f"Found {len(sentences)} sentences")
        
        # Create vocabulary curriculum
        if self.enable_vocab_curriculum:
            print("Creating vocabulary curriculum...")
            self.vocab_curriculum = self.create_vocabulary_curriculum(sentences)
        
        # Calculate static difficulty scores
        print("Calculating lexical difficulty...")
        lexical_scores = self.calculate_lexical_difficulty(sentences)
        
        print("Calculating syntactic complexity...")
        syntactic_scores = self.calculate_syntactic_complexity(sentences)
        
        print("Calculating thematic centrality...")
        thematic_scores, cluster_labels = self.calculate_thematic_centrality(sentences)
        
        print("Analyzing argumentative structure...")
        arg_roles, arg_confidences = self.calculate_argumentative_difficulty(sentences)
        
        print("Assigning vocabulary levels...")
        vocab_levels = self.assign_vocabulary_levels(sentences)
        
        # Create segments with all scores
        self.segments = []
        for i, sentence in enumerate(sentences):
            segment = TextSegment(
                text=sentence,
                index=i,
                lexical_rarity=lexical_scores[i],
                syntactic_complexity=syntactic_scores[i],
                thematic_centrality=thematic_scores[i],
                argumentative_role=arg_roles[i],
                argumentative_confidence=arg_confidences[i],
                length=len(sentence.split()),
                cluster_id=cluster_labels[i],
                vocabulary_level=vocab_levels[i]
            )
            
            # Calculate initial combined difficulty
            segment.combined_difficulty = self.combine_difficulty_scores(segment)
            self.segments.append(segment)
        
        # Assign curriculum stages
        print("Assigning curriculum stages...")
        self.assign_curriculum_stages(self.segments)
        
        print(f"Stage assignments: Foundation={sum(1 for s in self.segments if s.stage_assignment == 1)}, "
              f"Structural={sum(1 for s in self.segments if s.stage_assignment == 2)}, "
              f"Refinement={sum(1 for s in self.segments if s.stage_assignment == 3)}")
        
        return self.segments
    
    def get_stage_data(self, stage: int, vocab_level: int = None) -> List[TextSegment]:
        """Get data for specific curriculum stage and vocabulary level"""
        stage_segments = [s for s in self.segments if s.stage_assignment == stage]
        
        if vocab_level is not None:
            stage_segments = [s for s in stage_segments if s.vocabulary_level <= vocab_level]
        
        return stage_segments
    
    def get_argument_pairs(self, stage: int = 2) -> List[Tuple[TextSegment, TextSegment]]:
        """Get argumentative pairs for Stage II training"""
        stage_segments = self.get_stage_data(stage)
        return self.create_argument_pairs(stage_segments)
    
    def save_data(self, output_dir: str):
        """Save processed data with all curriculum components"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline config
        with open(output_path / "pipeline_config.pkl", "wb") as f:
            pickle.dump(self.config, f)
        
        # Save segments
        with open(output_path / "segments.pkl", "wb") as f:
            pickle.dump(self.segments, f)
        
        # Save curriculum splits by stage
        stage_splits = {}
        for stage in [1, 2, 3]:
            stage_splits[f'stage_{stage}'] = self.get_stage_data(stage)
        
        with open(output_path / "curriculum_splits.pkl", "wb") as f:
            pickle.dump(stage_splits, f)
        
        # Save argument pairs
        if self.enable_argument_mining:
            argument_pairs = self.get_argument_pairs()
            with open(output_path / "argument_pairs.pkl", "wb") as f:
                pickle.dump(argument_pairs, f)
        
        # Save vocabulary curriculum
        if self.vocab_curriculum:
            with open(output_path / "vocab_curriculum.pkl", "wb") as f:
                pickle.dump(self.vocab_curriculum, f)
        
        # Save tokenizer with compression when vocab curriculum disabled
        if self.vocab_curriculum:
            for level in range(1, self.vocab_curriculum.max_level + 1):
                tokenizer = self.create_adaptive_tokenizer(
                    [s.text for s in self.segments], vocab_level=level
                )
                tokenizer_path = output_path / f"tokenizer_level_{level}"
                tokenizer.save_pretrained(tokenizer_path)
        else:
            # Apply compression when vocab curriculum is disabled
            if self.tokenizer:
                print("🔧 Compressing vocabulary for saving...")
                texts = [s.text for s in self.segments]
                compressed_tokenizer = self._compress_tokenizer(self.tokenizer, texts)
                
                tokenizer_path = output_path / "tokenizer"
                compressed_tokenizer.save_pretrained(tokenizer_path)
                
                print(f"💾 Compressed tokenizer saved: {len(compressed_tokenizer):,} tokens")
            else:
                print("⚠️ No tokenizer to save")
        
        # Save embeddings and models
        if self.embeddings is not None:
            np.save(output_path / "embeddings.npy", self.embeddings)
        
        if self.cluster_model:
            with open(output_path / "cluster_model.pkl", "wb") as f:
                pickle.dump(self.cluster_model, f)
        
        # Save dynamic scorer
        with open(output_path / "dynamic_scorer.pkl", "wb") as f:
            pickle.dump(self.dynamic_scorer, f)
        
        print(f"Enhanced curriculum data saved to: {output_path}")
    
    def load_data(self, data_dir: str):
        """Load previously processed curriculum data"""
        data_path = Path(data_dir)
        
        # Load pipeline config
        config_path = data_path / "pipeline_config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                self.config = pickle.load(f)
        
        # Load segments
        with open(data_path / "segments.pkl", "rb") as f:
            self.segments = pickle.load(f)
        
        # Load vocabulary curriculum
        vocab_curriculum_path = data_path / "vocab_curriculum.pkl"
        if vocab_curriculum_path.exists():
            with open(vocab_curriculum_path, "rb") as f:
                self.vocab_curriculum = pickle.load(f)
        
        # Load dynamic scorer
        scorer_path = data_path / "dynamic_scorer.pkl"
        if scorer_path.exists():
            with open(scorer_path, "rb") as f:
                self.dynamic_scorer = pickle.load(f)
        
        # Load embeddings and models
        if (data_path / "embeddings.npy").exists():
            self.embeddings = np.load(data_path / "embeddings.npy")
        
        if (data_path / "cluster_model.pkl").exists():
            with open(data_path / "cluster_model.pkl", "rb") as f:
                self.cluster_model = pickle.load(f)
        
        print(f"Enhanced curriculum data loaded from: {data_path}")


def main():
    """Example usage of configuration-driven pipeline"""
    # Initialize with custom config
    pipeline_config = PipelineConfig.high_quality()
    
    pipeline = TextDataPipeline(
        pipeline_config=pipeline_config,
        target_vocab_size=25000,
        enable_argument_mining=True,
        enable_vocab_curriculum=True
    )
    
    # Process text file
    segments = pipeline.process_text_file("data/raw/sample_text.txt")
    
    # Save results
    pipeline.save_data("data/processed")
    
    # Print statistics
    difficulties = [s.combined_difficulty for s in segments]
    print(f"\nConfiguration-Driven Pipeline Statistics:")
    print(f"Config: {type(pipeline.config).__name__}")
    print(f"Total segments: {len(segments)}")
    print(f"Difficulty range: {min(difficulties):.3f} - {max(difficulties):.3f}")
    print(f"Argumentative roles: {Counter(s.argumentative_role for s in segments)}")
    print(f"Vocabulary levels: {Counter(s.vocabulary_level for s in segments)}")
    
    # Show stage distribution
    for stage in [1, 2, 3]:
        stage_data = pipeline.get_stage_data(stage)
        print(f"Stage {stage}: {len(stage_data)} segments")
    
    # Show argument pairs for Stage II
    pairs = pipeline.get_argument_pairs()
    print(f"Argument pairs for Stage II: {len(pairs)}")


if __name__ == "__main__":
    main()