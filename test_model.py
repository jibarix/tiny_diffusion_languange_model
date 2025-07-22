#!/usr/bin/env python3
"""Test the model architecture"""

import torch
from config import ProjectConfig
from src.model.diffusion import MaskedDiffusionLM

def test_model():
    """Test model creation and forward pass"""
    
    # Load config
    config = ProjectConfig.default()
    model_config = config.model
    
    # Create model
    model = MaskedDiffusionLM(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
        pad_token_id=0,
        mask_token_id=1
    )
    
    print(f"🏗️  Model created: {model.get_num_params():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(2, model_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Training forward pass (with masking)
    model.train()
    outputs = model(input_ids, attention_mask, masking_rate=0.3)
    loss = model.compute_loss(outputs)
    
    print(f"✅ Training pass: Loss = {loss.item():.4f}")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Masked tokens: {outputs['corruption_mask'].sum().item()}")
    
    # Inference test
    model.eval()
    with torch.no_grad():
        # Start with some masks
        test_input = input_ids.clone()
        test_input[:, 50:60] = 1  # Set positions to mask token
        
        denoised = model.generate_step(test_input, temperature=0.8)
        print(f"✅ Inference: Filled {(test_input == 1).sum().item()} masked tokens")
    
    # Memory usage
    if torch.cuda.is_available():
        model = model.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        outputs = model(input_ids, attention_mask, masking_rate=0.3)
        loss = model.compute_loss(outputs)
        
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"🔥 GPU Memory: {memory_mb:.1f} MB")
        
        if memory_mb < 4000:
            print("✅ Memory usage looks good for 8GB VRAM")
        else:
            print("⚠️  High memory usage - may need smaller batch size")

if __name__ == "__main__":
    test_model()