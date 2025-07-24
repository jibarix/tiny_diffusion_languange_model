#!/usr/bin/env python3
"""
Visual Diffusion Text Generator
Shows real-time masked diffusion process with animated token revealing
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import click

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import MaskedDiffusionLM, load_model_checkpoint
from src.data import CompressedTokenizer

console = Console()

class VisualDiffusionGenerator:
    """Generator that shows the actual diffusion process step by step"""
    
    def __init__(self, model: MaskedDiffusionLM, tokenizer: CompressedTokenizer, device: str = 'auto'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Special tokens
        self.mask_token_id = self.tokenizer.token_mapping.get('[MASK]', 1)
        self.pad_token_id = self.tokenizer.token_mapping.get('[PAD]', 0)
    
    def generate_with_visualization(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_diffusion_steps: int = 20,
        temperature: float = 0.6,
        top_k: int = 20,
        animation_speed: float = 0.3
    ) -> str:
        """Generate text with step-by-step visualization"""
        
        console.clear()
        console.print(Panel.fit(f"🌟 [bold cyan]Masked Diffusion Generation[/bold cyan] 🌟\nPrompt: [yellow]{prompt}[/yellow]"))
        
        # Initialize sequence with properly decoded prompt tokens (same as web interface)
        prompt_token_ids = self.tokenizer.encode(prompt)
        
        # Decode each token properly using the tokenizer's vocab
        prompt_tokens = []
        for token_id in prompt_token_ids:
            # Get token from vocab
            vocab_items = list(self.tokenizer.compressed_vocab.items())
            token_text = next((token for token, tid in vocab_items if tid == token_id), str(token_id))
            
            # Clean up the token
            clean_token = token_text.replace('Ġ', ' ').strip()
            prompt_tokens.append(clean_token if clean_token else token_text)
        
        # Initialize revealed tokens and states
        revealed_tokens = prompt_tokens + [-1] * max_new_tokens
        token_states = ['prompt'] * len(prompt_tokens) + ['masked'] * max_new_tokens
        
        # Display initial state
        self._display_current_state(revealed_tokens, token_states, 0, num_diffusion_steps, 1.0)
        time.sleep(animation_speed)
        
        # Diffusion loop
        for step in range(num_diffusion_steps):
            # Calculate masking schedule
            progress = (step + 1) / num_diffusion_steps
            masking_rate = 1.0 - progress
            
            # Find masked positions
            masked_positions = [i for i, state in enumerate(token_states) if state == 'masked']
            
            if masked_positions:
                # Determine how many tokens to reveal this step
                tokens_to_reveal = max(1, int(len(masked_positions) * 0.3))
                positions_to_reveal = random.sample(masked_positions, 
                                                   min(tokens_to_reveal, len(masked_positions)))
                
                # Use actual model inference instead of random selection
                with torch.no_grad():
                    # Prepare input tensor 
                    input_tensor = torch.tensor([prompt_token_ids + [self.mask_token_id] * max_new_tokens], device=self.device)
                    
                    # Update input tensor with revealed tokens
                    for i, token in enumerate(revealed_tokens):
                        if token != -1 and i < len(prompt_token_ids):
                            continue  # Keep prompt tokens
                        elif token != -1:
                            # Find token ID for revealed token
                            vocab_items = list(self.tokenizer.compressed_vocab.items())
                            token_id = next((tid for tok, tid in vocab_items if tok == token), self.mask_token_id)
                            input_tensor[0, i] = token_id
                    
                    # Get model predictions
                    outputs = self.model(input_ids=input_tensor)
                    logits = outputs['logits'][0]  # [seq_len, vocab_size]
                    
                    # Sample tokens for masked positions
                    for pos in positions_to_reveal:
                        pos_logits = logits[pos]
                        
                        # Apply temperature
                        pos_logits = pos_logits / temperature
                        
                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(pos_logits, min(top_k, pos_logits.size(-1)))
                            indices_to_remove = pos_logits < top_k_logits[..., -1, None]
                            pos_logits[indices_to_remove] = float('-inf')
                        
                        # Sample from distribution
                        probabilities = torch.softmax(pos_logits, dim=-1)
                        predicted_token_id = torch.multinomial(probabilities, 1).item()
                        
                        # Get token text
                        vocab_items = list(self.tokenizer.compressed_vocab.items())
                        predicted_token = next((token for token, tid in vocab_items if tid == predicted_token_id), str(predicted_token_id))
                        
                        # Clean up the token for display
                        display_token = predicted_token.replace('Ġ', ' ').strip()
                        if not display_token:
                            display_token = predicted_token
                        
                        revealed_tokens[pos] = display_token
                        token_states[pos] = 'revealing'
                
                # Show revealing animation
                self._display_current_state(revealed_tokens, token_states, step + 1, num_diffusion_steps, masking_rate)
                time.sleep(animation_speed)
                
                # Mark as revealed after animation
                for pos in positions_to_reveal:
                    token_states[pos] = 'revealed'
                
                # Show final revealed state
                self._display_current_state(revealed_tokens, token_states, step + 1, num_diffusion_steps, masking_rate)
                time.sleep(animation_speed / 2)
        
        # Generate final text (same as web interface)
        final_tokens = [str(t) for t in revealed_tokens if t != -1]
        final_text = ' '.join(final_tokens).replace('Ġ', ' ').strip()
        
        # Show completion
        self._display_current_state(revealed_tokens, token_states, num_diffusion_steps, num_diffusion_steps, 0.0, final=True)
        
        return final_text
    
    def _display_current_state(self, tokens: List[str], states: List[str], 
                              current_step: int, total_steps: int, masking_rate: float,
                              final: bool = False):
        """Display the current state of token generation"""
        
        # Create visual representation
        text_parts = []
        for i, (token, state) in enumerate(zip(tokens, states)):
            if state == 'prompt':
                # Prompt tokens in blue
                text_parts.append(f"[bold blue]{token}[/bold blue]")
            elif state == 'masked':
                # Masked tokens as red blocks
                text_parts.append("[bold red]▓[/bold red]")
            elif state == 'revealing':
                # Currently revealing in cyan
                text_parts.append(f"[bold cyan]{token}[/bold cyan]")
            elif state == 'revealed':
                # Revealed tokens in white
                text_parts.append(f"[white]{token}[/white]")
            elif token == -1:
                # Skip uninitialized tokens
                continue
            else:
                # Fallback
                text_parts.append(f"[white]{token}[/white]")
        
        # Join with spaces
        display_text = " ".join(text_parts)
        
        # Calculate stats
        masked_count = sum(1 for s in states if s == 'masked')
        revealed_count = sum(1 for s in states if s in ['revealed', 'revealing'])
        total_tokens = len([t for t in tokens if t != -1])
        
        # Create info panel
        info_text = f"""
[bold]Step:[/bold] {current_step}/{total_steps}
[bold]Masking Rate:[/bold] {masking_rate:.1%}
[bold]Tokens:[/bold] {revealed_count}/{total_tokens} revealed
[bold]Status:[/bold] {'Complete!' if final else 'Generating...'}
        """.strip()
        
        # Clear and display
        console.clear()
        console.print(Panel.fit(f"🌟 [bold cyan]Masked Diffusion Generation[/bold cyan] 🌟"))
        console.print()
        console.print(Panel(display_text, title="Generated Text", border_style="cyan"))
        console.print()
        console.print(Panel(info_text, title="Statistics", border_style="blue"))
    
    def interactive_generation(self):
        """Interactive mode with real-time diffusion visualization"""
        console.clear()
        console.print(Panel.fit("""
[bold cyan]🎭 Interactive Visual Diffusion Generator 🎭[/bold cyan]

Watch your text materialize through the diffusion process!
Type prompts and see tokens reveal step by step.

Commands:
  • [yellow]quit[/yellow] - Exit
  • [yellow]settings[/yellow] - Adjust parameters
        """))
        
        # Default settings
        settings = {
            'max_tokens': 40,
            'steps': 15,
            'temperature': 0.6,
            'top_k': 20,
            'speed': 0.2
        }
        
        while True:
            console.print()
            prompt = console.input("[bold green]Enter prompt:[/bold green] ")
            
            if prompt.lower() in ['quit', 'exit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif prompt.lower() == 'settings':
                settings = self._configure_settings(settings)
                continue
            elif not prompt.strip():
                continue
            
            try:
                # Generate with visualization
                result = self.generate_with_visualization(
                    prompt=prompt,
                    max_new_tokens=settings['max_tokens'],
                    num_diffusion_steps=settings['steps'],
                    temperature=settings['temperature'],
                    top_k=settings['top_k'],
                    animation_speed=settings['speed']
                )
                
                console.print()
                console.print(Panel(f"[bold green]Final Result:[/bold green]\n{result}", 
                                  title="✨ Generated Text", border_style="green"))
                
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")
    
    def _configure_settings(self, current_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Interactive settings configuration"""
        console.print("\n[bold cyan]⚙️ Configuration[/bold cyan]")
        console.print("Press Enter to keep current value\n")
        
        new_settings = current_settings.copy()
        
        try:
            # Max tokens
            response = console.input(f"Max tokens ({current_settings['max_tokens']}): ")
            if response.strip():
                new_settings['max_tokens'] = max(10, min(200, int(response)))
            
            # Diffusion steps
            response = console.input(f"Diffusion steps ({current_settings['steps']}): ")
            if response.strip():
                new_settings['steps'] = max(5, min(100, int(response)))
            
            # Temperature
            response = console.input(f"Temperature ({current_settings['temperature']}): ")
            if response.strip():
                new_settings['temperature'] = max(0.1, min(2.0, float(response)))
            
            # Top-k
            response = console.input(f"Top-k ({current_settings['top_k']}): ")
            if response.strip():
                new_settings['top_k'] = max(1, min(100, int(response)))
            
            # Animation speed
            response = console.input(f"Animation speed in seconds ({current_settings['speed']}): ")
            if response.strip():
                new_settings['speed'] = max(0.05, min(2.0, float(response)))
            
            console.print("[green]✅ Settings updated![/green]")
            
        except ValueError:
            console.print("[red]❌ Invalid input, keeping current settings[/red]")
        
        return new_settings


def load_model_for_visualization(checkpoint_path: str):
    """Load model and tokenizer for visualization with correct architecture detection"""
    console.print(f"[cyan]Loading model from:[/cyan] {checkpoint_path}")
    
    # Load tokenizer first
    tokenizer_path = "data/processed/compressed_tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = CompressedTokenizer.load(tokenizer_path)
    console.print(f"[green]✅ Loaded compressed tokenizer:[/green] {len(tokenizer.compressed_vocab)} tokens")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        config = checkpoint['config']
        if 'model' in config:
            model_config = config['model']
            # Ensure vocab_size matches tokenizer
            model_config['vocab_size'] = len(tokenizer.compressed_vocab)
            model_config['mask_token_id'] = tokenizer.token_mapping.get('[MASK]', 1)
            model_config['pad_token_id'] = tokenizer.token_mapping.get('[PAD]', 0)
            console.print(f"[green]✅ Using model config from checkpoint[/green]")
            console.print(f"  d_model: {model_config.get('d_model')}")
            console.print(f"  n_layers: {model_config.get('n_layers')}")
            console.print(f"  n_heads: {model_config.get('n_heads')}")
        else:
            raise ValueError("No model config found in checkpoint")
    else:
        console.print("[yellow]⚠️  No valid config in checkpoint, inferring from state_dict...[/yellow]")
        
        # Try to infer model config from state dict shapes
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Infer dimensions from embedding layer
        embed_shape = state_dict['embed_tokens.weight'].shape
        vocab_size, d_model = embed_shape
        
        # Count layers by looking for layer keys
        layer_keys = [k for k in state_dict.keys() if k.startswith('layers.')]
        max_layer = max([int(k.split('.')[1]) for k in layer_keys]) + 1
        
        # Infer number of heads from attention projection shapes
        q_proj_shape = state_dict['layers.0.self_attn.q_proj.weight'].shape
        # Common configurations based on d_model
        if d_model == 768:
            n_heads = 12
            ffn_hidden_size = 2048
        elif d_model == 512:
            n_heads = 8
            ffn_hidden_size = 1344
        elif d_model == 128:
            n_heads = 4
            ffn_hidden_size = 320
        else:
            n_heads = max(1, d_model // 64)  # General rule
            ffn_hidden_size = d_model * 4
        
        model_config = {
            'd_model': d_model,
            'n_layers': max_layer,
            'n_heads': n_heads,
            'head_dim': d_model // n_heads,
            'ffn_hidden_size': ffn_hidden_size,
            'vocab_size': vocab_size,
            'max_position_embeddings': 2048,
            'attention_dropout': 0.0,
            'hidden_dropout': 0.1,
            'use_causal_mask': False,
            'mask_token_id': tokenizer.token_mapping.get('[MASK]', 1),
            'pad_token_id': tokenizer.token_mapping.get('[PAD]', 0),
            'norm_eps': 1e-6,
            'initializer_range': 0.02,
            'gradient_checkpointing': False,
            'use_cache': False,
        }
        
        console.print(f"[green]✅ Inferred model config:[/green]")
        console.print(f"  d_model: {model_config['d_model']}")
        console.print(f"  n_layers: {model_config['n_layers']}")
        console.print(f"  n_heads: {model_config['n_heads']}")
        console.print(f"  vocab_size: {model_config['vocab_size']}")
    
    # Create model with correct config
    model = MaskedDiffusionLM(model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    console.print(f"[green]✅ Model loaded successfully[/green]")
    console.print(f"  Parameters: {model.get_num_params():,}")
    
    return model, tokenizer


@click.command()
@click.option('--checkpoint', required=True, help='Path to model checkpoint')
@click.option('--prompt', help='Single prompt for generation')
@click.option('--interactive', is_flag=True, help='Interactive mode')
@click.option('--max-tokens', default=50, help='Maximum new tokens')
@click.option('--steps', default=20, help='Diffusion steps')
@click.option('--temperature', default=0.6, help='Sampling temperature')
@click.option('--speed', default=0.3, help='Animation speed (seconds)')
def main(checkpoint, prompt, interactive, max_tokens, steps, temperature, speed):
    """Visual Diffusion Text Generator - Watch text emerge from noise!"""
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_for_visualization(checkpoint)
        
        # Create generator
        generator = VisualDiffusionGenerator(model, tokenizer)
        
        if interactive:
            generator.interactive_generation()
        else:
            if not prompt:
                prompt = console.input("[bold green]Enter prompt:[/bold green] ")
            
            result = generator.generate_with_visualization(
                prompt=prompt,
                max_new_tokens=max_tokens,
                num_diffusion_steps=steps,
                temperature=temperature,
                animation_speed=speed
            )
            
            console.print()
            console.print(Panel(f"[bold green]Final Result:[/bold green]\n{result}", 
                              title="✨ Generated Text", border_style="green"))
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Generation interrupted![/yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()