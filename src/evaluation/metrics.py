"""
Evaluation Metrics
Text quality, style, and coherence metrics for diffusion models
"""

import math
import re
import statistics
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class StyleMetrics:
    """Stylometric analysis for comparing text styles"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        
    def compare_styles(self, generated_texts: List[str], 
                      reference_texts: List[str]) -> Dict[str, float]:
        """Compare stylistic features between generated and reference texts"""
        if not generated_texts or not reference_texts:
            return {'overall_similarity': 0.0}
            
        gen_features = self._extract_style_features(generated_texts)
        ref_features = self._extract_style_features(reference_texts)
        
        similarities = {}
        for feature in gen_features:
            if feature in ref_features:
                gen_val = gen_features[feature]
                ref_val = ref_features[feature]
                
                # Calculate normalized similarity (1 - normalized absolute difference)
                max_val = max(abs(gen_val), abs(ref_val), 1e-6)
                diff = abs(gen_val - ref_val)
                similarity = max(0.0, 1.0 - (diff / max_val))
                similarities[f"{feature}_similarity"] = similarity
        
        # Overall similarity (average of all feature similarities)
        if similarities:
            similarities['overall_similarity'] = np.mean(list(similarities.values()))
        else:
            similarities['overall_similarity'] = 0.0
        
        return similarities
    
    def _extract_style_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract comprehensive stylistic features"""
        features = {}
        
        if not texts:
            return features
        
        # Combine all texts for analysis
        combined_text = ' '.join(texts)
        
        # Length-based features
        features.update(self._get_length_features(texts))
        
        # Lexical features
        features.update(self._get_lexical_features(texts))
        
        # Syntactic features
        features.update(self._get_syntactic_features(texts))
        
        # Readability features
        features.update(self._get_readability_features(texts))
        
        return features
    
    def _get_length_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract length-based stylistic features"""
        features = {}
        
        word_lengths = []
        sentence_lengths = []
        paragraph_lengths = []
        
        for text in texts:
            # Split into paragraphs (double newline) and sentences
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                sentences = [s.strip() for s in re.split(r'[.!?]+', para) if s.strip()]
                paragraph_lengths.append(len(sentences))
                
                for sent in sentences:
                    words = sent.split()
                    if words:
                        sentence_lengths.append(len(words))
                        word_lengths.extend([len(word.strip('.,!?;:"()[]')) for word in words])
        
        if word_lengths:
            features['avg_word_length'] = np.mean(word_lengths)
            features['word_length_std'] = np.std(word_lengths)
            features['word_length_range'] = max(word_lengths) - min(word_lengths)
        
        if sentence_lengths:
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['sentence_length_std'] = np.std(sentence_lengths)
            features['sentence_length_median'] = np.median(sentence_lengths)
        
        if paragraph_lengths:
            features['avg_paragraph_length'] = np.mean(paragraph_lengths)
        
        return features
    
    def _get_lexical_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract lexical diversity and complexity features"""
        features = {}
        
        all_words = []
        all_sentences = []
        
        for text in texts:
            # Clean and tokenize
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            all_sentences.extend(sentences)
            
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        if all_words:
            # Type-Token Ratio (TTR)
            unique_words = len(set(all_words))
            total_words = len(all_words)
            features['type_token_ratio'] = unique_words / total_words
            
            # Moving Average TTR (MATTR) - more stable for varying text lengths
            if total_words >= 100:
                window_size = min(100, total_words)
                ttrs = []
                for i in range(total_words - window_size + 1):
                    window_words = all_words[i:i + window_size]
                    window_ttr = len(set(window_words)) / len(window_words)
                    ttrs.append(window_ttr)
                features['mattr'] = np.mean(ttrs)
            
            # Vocabulary richness measures
            word_freq = Counter(all_words)
            features['hapax_legomena_ratio'] = sum(1 for count in word_freq.values() if count == 1) / total_words
            features['most_frequent_word_ratio'] = max(word_freq.values()) / total_words
            
            # Average word frequency (inverse of rarity)
            features['avg_word_frequency'] = np.mean(list(word_freq.values()))
        
        return features
    
    def _get_syntactic_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract syntactic complexity features"""
        features = {}
        
        # Function word analysis (common English function words)
        function_words = {
            'articles': ['the', 'a', 'an'],
            'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'to', 'of', 'from'],
            'conjunctions': ['and', 'or', 'but', 'nor', 'for', 'yet', 'so'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
        }
        
        all_words = []
        complex_sentence_count = 0
        total_sentences = 0
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
            
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            total_sentences += len(sentences)
            
            # Count complex sentences (with subordinate clauses)
            for sent in sentences:
                # Simple heuristic: contains subordinating conjunctions
                subordinating_words = ['because', 'since', 'although', 'while', 'if', 'when', 'where', 'that', 'which', 'who']
                if any(word in sent.lower() for word in subordinating_words):
                    complex_sentence_count += 1
        
        if all_words:
            total_words = len(all_words)
            
            # Function word ratios
            for category, words in function_words.items():
                count = sum(all_words.count(word) for word in words)
                features[f'{category}_ratio'] = count / total_words
            
            # Overall function word ratio
            all_function_words = [word for word_list in function_words.values() for word in word_list]
            function_word_count = sum(all_words.count(word) for word in all_function_words)
            features['function_word_ratio'] = function_word_count / total_words
        
        if total_sentences > 0:
            features['complex_sentence_ratio'] = complex_sentence_count / total_sentences
        
        return features
    
    def _get_readability_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract readability-based features"""
        features = {}
        
        total_sentences = 0
        total_words = 0
        total_syllables = 0
        
        for text in texts:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            total_sentences += len(sentences)
            
            words = re.findall(r'\b\w+\b', text.lower())
            total_words += len(words)
            
            # Estimate syllables (simple heuristic)
            for word in words:
                syllable_count = self._estimate_syllables(word)
                total_syllables += syllable_count
        
        if total_sentences > 0 and total_words > 0:
            # Average words per sentence
            features['avg_words_per_sentence'] = total_words / total_sentences
            
            # Average syllables per word
            features['avg_syllables_per_word'] = total_syllables / total_words
            
            # Flesch Reading Ease approximation
            features['flesch_score'] = (206.835 
                                      - 1.015 * (total_words / total_sentences)
                                      - 84.6 * (total_syllables / total_words))
        
        return features
    
    def _estimate_syllables(self, word: str) -> int:
        """Simple syllable estimation"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)


class StructuralMetrics:
    """Metrics for analyzing structural coherence and logical flow"""
    
    def analyze_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Analyze structural coherence of texts"""
        if not texts:
            return {'overall_coherence': 0.0}
        
        metrics = {}
        
        # Sentence-level coherence
        metrics.update(self._analyze_sentence_coherence(texts))
        
        # Discourse markers and transitions
        metrics.update(self._analyze_discourse_markers(texts))
        
        # Repetition patterns
        metrics.update(self._analyze_repetition_patterns(texts))
        
        # Topic consistency
        metrics.update(self._analyze_topic_consistency(texts))
        
        # Overall coherence score
        coherence_components = []
        for key, value in metrics.items():
            if 'coherence' in key or 'consistency' in key:
                coherence_components.append(value)
            elif 'repetition' in key:
                coherence_components.append(1.0 - value)  # Lower repetition = higher coherence
        
        if coherence_components:
            metrics['overall_coherence'] = np.mean(coherence_components)
        else:
            metrics['overall_coherence'] = 0.0
        
        return metrics
    
    def _analyze_sentence_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Analyze coherence between consecutive sentences"""
        coherence_scores = []
        
        for text in texts:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            if len(sentences) < 2:
                continue
            
            # Simple lexical overlap between consecutive sentences
            for i in range(len(sentences) - 1):
                current_words = set(re.findall(r'\b\w+\b', sentences[i].lower()))
                next_words = set(re.findall(r'\b\w+\b', sentences[i + 1].lower()))
                
                if current_words and next_words:
                    overlap = len(current_words & next_words)
                    union = len(current_words | next_words)
                    coherence = overlap / union if union > 0 else 0
                    coherence_scores.append(coherence)
        
        return {
            'sentence_coherence': np.mean(coherence_scores) if coherence_scores else 0.0,
            'coherence_variance': np.var(coherence_scores) if coherence_scores else 0.0
        }
    
    def _analyze_discourse_markers(self, texts: List[str]) -> Dict[str, float]:
        """Analyze use of discourse markers and transitions"""
        # Common discourse markers
        discourse_markers = {
            'additive': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'adversative': ['however', 'nevertheless', 'nonetheless', 'conversely', 'but'],
            'causal': ['therefore', 'thus', 'consequently', 'hence', 'accordingly'],
            'temporal': ['then', 'next', 'subsequently', 'meanwhile', 'afterwards'],
            'explanatory': ['for example', 'for instance', 'namely', 'specifically'],
        }
        
        all_sentences = []
        marker_counts = defaultdict(int)
        
        for text in texts:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            all_sentences.extend(sentences)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for category, markers in discourse_markers.items():
                    for marker in markers:
                        if marker in sentence_lower:
                            marker_counts[category] += 1
        
        total_sentences = len(all_sentences)
        metrics = {}
        
        if total_sentences > 0:
            total_markers = sum(marker_counts.values())
            metrics['discourse_marker_density'] = total_markers / total_sentences
            
            # Individual marker type ratios
            for category, count in marker_counts.items():
                metrics[f'{category}_marker_ratio'] = count / total_sentences
        
        return metrics
    
    def _analyze_repetition_patterns(self, texts: List[str]) -> Dict[str, float]:
        """Analyze problematic repetition patterns"""
        metrics = {}
        
        # Sentence-level repetition
        all_sentences = []
        sentence_starts = []
        
        for text in texts:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            all_sentences.extend(sentences)
            
            # Analyze sentence beginnings (first 3 words)
            for sentence in sentences:
                words = sentence.split()
                if len(words) >= 3:
                    start = ' '.join(words[:3]).lower()
                    sentence_starts.append(start)
        
        if sentence_starts:
            unique_starts = len(set(sentence_starts))
            total_starts = len(sentence_starts)
            metrics['sentence_start_repetition'] = 1.0 - (unique_starts / total_starts)
        
        # Phrase-level repetition
        all_phrases = []
        for text in texts:
            # Extract 3-gram phrases
            words = re.findall(r'\b\w+\b', text.lower())
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                all_phrases.append(phrase)
        
        if all_phrases:
            phrase_counts = Counter(all_phrases)
            repeated_phrases = sum(1 for count in phrase_counts.values() if count > 1)
            metrics['phrase_repetition'] = repeated_phrases / len(set(all_phrases))
        
        return metrics
    
    def _analyze_topic_consistency(self, texts: List[str]) -> Dict[str, float]:
        """Analyze topic consistency using simple keyword overlap"""
        if len(texts) < 2:
            return {'topic_consistency': 1.0}
        
        # Extract content words (filter out common function words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'shall', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        text_keywords = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            text_keywords.append(set(keywords))
        
        # Calculate pairwise topic overlap
        consistency_scores = []
        for i in range(len(text_keywords)):
            for j in range(i + 1, len(text_keywords)):
                keywords1 = text_keywords[i]
                keywords2 = text_keywords[j]
                
                if keywords1 and keywords2:
                    overlap = len(keywords1 & keywords2)
                    union = len(keywords1 | keywords2)
                    consistency = overlap / union if union > 0 else 0
                    consistency_scores.append(consistency)
        
        return {
            'topic_consistency': np.mean(consistency_scores) if consistency_scores else 0.0
        }


class SemanticSimilarity:
    """Semantic similarity metrics using embeddings"""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    def calculate_bert_score(self, generated_texts: List[str], 
                           reference_texts: List[str]) -> Dict[str, float]:
        """Calculate BERTScore-like semantic similarity"""
        if not self.model or not self.tokenizer:
            return self._tfidf_similarity(generated_texts, reference_texts)
        
        # This would require a proper BERT model for embeddings
        # For now, fall back to TF-IDF similarity
        return self._tfidf_similarity(generated_texts, reference_texts)
    
    def _tfidf_similarity(self, generated_texts: List[str], 
                         reference_texts: List[str]) -> Dict[str, float]:
        """Calculate TF-IDF based semantic similarity"""
        if not generated_texts or not reference_texts:
            return {'semantic_similarity': 0.0}
        
        try:
            # Combine texts for vectorization
            all_texts = generated_texts + reference_texts
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Split back into generated and reference
            gen_vectors = tfidf_matrix[:len(generated_texts)]
            ref_vectors = tfidf_matrix[len(generated_texts):]
            
            # Calculate average similarity
            similarities = []
            for i in range(gen_vectors.shape[0]):
                gen_vec = gen_vectors[i:i+1]
                sims = cosine_similarity(gen_vec, ref_vectors).flatten()
                similarities.extend(sims)
            
            return {
                'semantic_similarity': np.mean(similarities),
                'semantic_similarity_std': np.std(similarities)
            }
        
        except Exception:
            return {'semantic_similarity': 0.0}


class PerplexityMetrics:
    """Perplexity and language modeling metrics"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def calculate_perplexity(self, texts: List[str], 
                           batch_size: int = 8,
                           masking_rate: float = 0.15) -> float:
        """Calculate perplexity on text corpus"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                loss, tokens = self._process_batch(batch_texts, masking_rate)
                total_loss += loss
                total_tokens += tokens
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)
    
    def _process_batch(self, texts: List[str], masking_rate: float) -> Tuple[float, int]:
        """Process a batch of texts for perplexity calculation"""
        # Tokenize texts
        encoded = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) > 512:  # Truncate long sequences
                tokens = tokens[:512]
            encoded.append(tokens)
        
        if not encoded:
            return 0.0, 0
        
        # Pad to same length
        max_len = max(len(tokens) for tokens in encoded)
        input_ids = torch.zeros((len(encoded), max_len), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        
        for i, tokens in enumerate(encoded):
            input_ids[i, :len(tokens)] = torch.tensor(tokens)
            attention_mask[i, :len(tokens)] = 1
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            masking_rate=masking_rate
        )
        
        loss = self.model.compute_loss(outputs)
        valid_tokens = attention_mask.sum().item()
        
        return loss.item() * valid_tokens, valid_tokens