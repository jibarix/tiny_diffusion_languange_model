"""
Format-aware Dataset Classes for Curriculum Training
"""

import torch
import random
from torch.utils.data import Dataset
from typing import List


class SentenceDataset(Dataset):
    """Individual sentences (Stage I)"""
    
    def __init__(self, segments, tokenizer, max_length: int = 512):
        self.segments = segments
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        text = segment.text
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor and pad
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'difficulty': segment.combined_difficulty
        }


class PairDataset(Dataset):
    """Evidence-claim pairs (Stage II)"""
    
    def __init__(self, segments, tokenizer, max_length: int = 512):
        self.segments = segments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = tokenizer.sep_token or " [SEP] "
        
    def __len__(self):
        return len(self.segments) // 2  # Pairs
    
    def __getitem__(self, idx):
        # Create evidence-claim pairs from consecutive segments
        evidence = self.segments[idx * 2].text
        claim = self.segments[idx * 2 + 1].text if (idx * 2 + 1) < len(self.segments) else evidence
        
        # Format as evidence [SEP] claim
        text = f"{evidence}{self.sep_token}{claim}"
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        avg_difficulty = (self.segments[idx * 2].combined_difficulty + 
                         self.segments[min(idx * 2 + 1, len(self.segments) - 1)].combined_difficulty) / 2
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'difficulty': avg_difficulty
        }


class ParagraphDataset(Dataset):
    """Multi-sentence paragraphs (Stage III)"""
    
    def __init__(self, segments, tokenizer, max_length: int = 512, sentences_per_paragraph: int = 4):
        self.segments = segments
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sentences_per_paragraph = sentences_per_paragraph
        
    def __len__(self):
        return len(self.segments) // self.sentences_per_paragraph
    
    def __getitem__(self, idx):
        # Combine multiple sentences into paragraph
        start_idx = idx * self.sentences_per_paragraph
        end_idx = min(start_idx + self.sentences_per_paragraph, len(self.segments))
        
        paragraph_sentences = [self.segments[i].text for i in range(start_idx, end_idx)]
        text = " ".join(paragraph_sentences)
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        # Average difficulty across sentences
        avg_difficulty = sum(self.segments[i].combined_difficulty 
                           for i in range(start_idx, end_idx)) / (end_idx - start_idx)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'difficulty': avg_difficulty
        }