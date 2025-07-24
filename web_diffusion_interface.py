#!/usr/bin/env python3
"""
Complete Web Interface for Visual Diffusion Generation
Flask app that connects to your trained model with real-time diffusion visualization
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Generator

from flask import Flask, render_template, render_template_string, request, jsonify, Response
from flask_socketio import SocketIO, emit
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import load_model_checkpoint
from src.data import CompressedTokenizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diffusion_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global model storage
model = None
tokenizer = None
device = None

class WebDiffusionGenerator:
    """Real-time diffusion generator for web interface using actual model predictions"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mask_token_id = tokenizer.token_mapping.get('[MASK]', 1)
        
    def generate_streaming(self, prompt: str, config: Dict[str, Any]):
        """Stream generation events to web client using actual model inference"""
        import random
        
        # Parse config
        max_new_tokens = config.get('max_tokens', 50)
        num_diffusion_steps = config.get('steps', 20)
        temperature = config.get('temperature', 0.6)
        top_k = config.get('top_k', 20)
        
        # Initialize sequence with properly decoded prompt tokens
        prompt_token_ids = self.tokenizer.encode(prompt)
        
        # Create input sequence: prompt + masked tokens
        input_ids = prompt_token_ids + [self.mask_token_id] * max_new_tokens
        
        # Only track the NEW tokens for display (not the prompt)
        display_tokens = [-1] * max_new_tokens  # Only new tokens
        token_states = ['masked'] * max_new_tokens  # Only new tokens
        
        # Send initial state
        yield {
            'type': 'init',
            'prompt': prompt,
            'total_tokens': max_new_tokens,  # Only count new tokens
            'steps': num_diffusion_steps
        }
        
        # Diffusion loop
        for step in range(num_diffusion_steps):
            # Calculate masking schedule
            progress = (step + 1) / num_diffusion_steps
            masking_rate = 1.0 - progress
            
            # Send step update
            yield {
                'type': 'step',
                'step': step + 1,
                'progress': progress,
                'masking_rate': masking_rate
            }
            
            # Find masked positions (only in the new token range)
            masked_positions = [i for i, state in enumerate(token_states) if state == 'masked']
            
            if masked_positions:
                # Use actual model inference
                with torch.no_grad():
                    # Prepare input tensor
                    input_tensor = torch.tensor([input_ids], device=self.device)
                    
                    # Get model predictions
                    outputs = self.model(input_ids=input_tensor)
                    logits = outputs['logits'][0]  # [seq_len, vocab_size]
                
                # Determine how many tokens to reveal this step
                tokens_to_reveal = max(1, int(len(masked_positions) * 0.3))
                positions_to_reveal = random.sample(masked_positions, 
                                                   min(tokens_to_reveal, len(masked_positions)))
                
                for pos in positions_to_reveal:
                    # Get logits for this position (offset by prompt length)
                    actual_pos = pos + len(prompt_token_ids)
                    pos_logits = logits[actual_pos]
                    
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
                    
                    # Update sequence
                    input_ids[actual_pos] = predicted_token_id
                    display_tokens[pos] = display_token
                    token_states[pos] = 'revealing'
                    
                    # Send token reveal event
                    yield {
                        'type': 'reveal',
                        'position': pos,
                        'token': display_token,
                        'state': 'revealing'
                    }
                
                # Mark as revealed after animation
                for pos in positions_to_reveal:
                    token_states[pos] = 'revealed'
                    yield {
                        'type': 'stabilize',
                        'position': pos,
                        'state': 'revealed'
                    }
        
        # Generate final text (only from new tokens)
        final_tokens = [str(t) for t in display_tokens if t != -1]
        final_text = ' '.join(final_tokens).replace('Ġ', ' ').strip()
        
        yield {
            'type': 'complete',
            'final_text': final_text,
            'total_tokens': len(final_tokens)
        }


def load_models():
    """Load model and tokenizer"""
    global model, tokenizer, device
    
    checkpoint_path = "outputs/checkpoints/best_stage3.pt"
    tokenizer_path = "data/processed/compressed_tokenizer.json"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return False
    
    try:
        # Load tokenizer first
        tokenizer = CompressedTokenizer.load(tokenizer_path)
        print(f"✅ Tokenizer loaded: {len(tokenizer.compressed_vocab)} tokens")
        
        # Load checkpoint
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model configuration from checkpoint
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config = checkpoint['config']
            if 'model' in config:
                model_config = config['model']
                # Ensure vocab_size matches tokenizer
                model_config['vocab_size'] = len(tokenizer.compressed_vocab)
                model_config['mask_token_id'] = tokenizer.token_mapping.get('[MASK]', 1)
                model_config['pad_token_id'] = tokenizer.token_mapping.get('[PAD]', 0)
                print(f"✅ Using model config from checkpoint")
                print(f"  d_model: {model_config.get('d_model')}")
                print(f"  n_layers: {model_config.get('n_layers')}")
                print(f"  n_heads: {model_config.get('n_heads')}")
            else:
                raise ValueError("No model config found in checkpoint")
        else:
            print("⚠️  No valid config in checkpoint, attempting to infer from state_dict...")
            
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
            n_heads = 12 if d_model == 768 else 4  # Common configurations
            
            model_config = {
                'd_model': d_model,
                'n_layers': max_layer,
                'n_heads': n_heads,
                'head_dim': d_model // n_heads,
                'ffn_hidden_size': 2048 if d_model >= 512 else d_model * 4,
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
            
            print(f"✅ Inferred model config:")
            print(f"  d_model: {model_config['d_model']}")
            print(f"  n_layers: {model_config['n_layers']}")
            print(f"  n_heads: {model_config['n_heads']}")
        
        # Create model with correct config
        from src.model import MaskedDiffusionLM
        model = MaskedDiffusionLM(model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume checkpoint is just the state dict
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        print(f"  Parameters: {model.get_num_params():,}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Main interface"""
    if model is None:
        return render_template_string("""
        <h1>Model Loading Error</h1>
        <p>The diffusion model could not be loaded. Please check server logs.</p>
        """)
    
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    """HTTP endpoint for generation"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400
    
    config = {
        'max_tokens': data.get('max_tokens', 50),
        'steps': data.get('steps', 20),
        'temperature': data.get('temperature', 0.6),
        'top_k': data.get('top_k', 20)
    }
    
    try:
        generator = WebDiffusionGenerator(model, tokenizer, device)
        
        # Collect all events
        events = []
        for event in generator.generate_streaming(prompt, config):
            events.append(event)
        
        return jsonify({'events': events})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@socketio.on('generate_stream')
def handle_streaming_generation(data):
    """WebSocket endpoint for real-time streaming"""
    if model is None:
        emit('error', {'message': 'Model not loaded'})
        return
    
    prompt = data.get('prompt', '').strip()
    if not prompt:
        emit('error', {'message': 'Prompt required'})
        return
    
    config = {
        'max_tokens': data.get('max_tokens', 50),
        'steps': data.get('steps', 20),
        'temperature': data.get('temperature', 0.6),
        'top_k': data.get('top_k', 20)
    }
    
    try:
        generator = WebDiffusionGenerator(model, tokenizer, device)
        
        for event in generator.generate_streaming(prompt, config):
            emit('generation_event', event)
            socketio.sleep(0.1)  # Control animation speed
            
    except Exception as e:
        emit('error', {'message': str(e)})


@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'loaded': False})
    
    return jsonify({
        'loaded': True,
        'vocab_size': len(tokenizer.compressed_vocab),
        'device': str(device),
        'model_params': model.get_num_params() if hasattr(model, 'get_num_params') else 'Unknown'
    })


# Complete HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tiny Diffusion Model - Live Generation</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Monaco', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            color: #e8e8e8;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .title {
            font-size: 2.5em;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .controls {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        .input-panel, .settings-panel {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .prompt-display {
            background: rgba(69, 183, 209, 0.1);
            border: 1px solid rgba(69, 183, 209, 0.3);
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 14px;
            color: #45b7d1;
        }
        .prompt-input {
            width: 100%;
            padding: 15px;
            background: rgba(0,0,0,0.3);
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #e8e8e8;
            font-size: 16px;
            margin-bottom: 15px;
        }
        .prompt-input:focus {
            outline: none;
            border-color: #4ecdc4;
            box-shadow: 0 0 15px rgba(78,205,196,0.3);
        }
        .generate-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .generate-btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(78, 205, 196, 0.4);
        }
        .generate-btn:disabled { 
            opacity: 0.6; 
            cursor: not-allowed; 
            transform: none;
        }
        .setting-group {
            margin-bottom: 15px;
        }
        .setting-label {
            display: block;
            margin-bottom: 5px;
            color: #bbb;
            font-size: 0.9em;
        }
        .setting-input {
            width: 100%;
            padding: 8px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 5px;
            color: #e8e8e8;
        }
        .output-section {
            background: rgba(0,0,0,0.4);
            padding: 30px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            min-height: 500px;
        }
        .progress-info {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            padding: 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 3px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            width: 0%;
            transition: width 0.3s ease;
        }
        .step-indicator {
            display: flex;
            justify-content: center;
            gap: 8px;
            margin-bottom: 20px;
        }
        .step {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        .step.active {
            background: #4ecdc4;
            transform: scale(1.3);
            box-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
        }
        .step.completed {
            background: #45b7d1;
        }
        .generation-display {
            font-size: 18px;
            line-height: 1.8;
            min-height: 300px;
            padding: 20px;
            border: 2px dashed rgba(255,255,255,0.1);
            border-radius: 10px;
            position: relative;
        }
        .token {
            display: inline;
            position: relative;
            transition: all 0.5s ease;
            margin-right: 4px;
        }
        .token.masked {
            background: rgba(255, 107, 107, 0.3);
            color: #ff6b6b;
            border-radius: 3px;
            padding: 2px 4px;
            animation: pulse 1.5s infinite;
        }
        .token.revealing {
            background: rgba(78, 205, 196, 0.3);
            color: #4ecdc4;
            border-radius: 3px;
            padding: 2px 4px;
            animation: reveal 0.8s ease-out;
        }
        .token.revealed {
            background: transparent;
            color: #e8e8e8;
            animation: stabilize 0.5s ease-out;
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        @keyframes reveal {
            0% {
                background: rgba(255, 107, 107, 0.3);
                color: #ff6b6b;
                transform: scale(1.2);
            }
            50% {
                background: rgba(78, 205, 196, 0.5);
                color: #4ecdc4;
                transform: scale(1.1);
            }
            100% {
                background: rgba(78, 205, 196, 0.3);
                color: #4ecdc4;
                transform: scale(1);
            }
        }
        @keyframes stabilize {
            0% {
                background: rgba(78, 205, 196, 0.3);
                color: #4ecdc4;
            }
            100% {
                background: transparent;
                color: #e8e8e8;
            }
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4ecdc4;
        }
        .stat-label {
            font-size: 0.9em;
            color: #999;
        }
        .placeholder {
            color: #666;
            font-style: italic;
            text-align: center;
            margin-top: 100px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">🤖 Tiny Diffusion Model</div>
            <div style="color: #999;">Real-time Masked Diffusion Text Generation</div>
        </div>

        <div class="controls">
            <div class="input-panel">
                <div class="prompt-display" id="promptDisplay" style="display: none;">
                    Prompt: <span id="currentPrompt"></span>
                </div>
                <input type="text" class="prompt-input" id="promptInput" 
                       placeholder="Enter your prompt..." value="The creature">
                <button class="generate-btn" id="generateBtn">🚀 Generate with Diffusion</button>
            </div>

            <div class="settings-panel">
                <div class="setting-group">
                    <label class="setting-label">Max Tokens</label>
                    <input type="number" class="setting-input" id="maxTokens" value="40" min="10" max="200">
                </div>
                <div class="setting-group">
                    <label class="setting-label">Diffusion Steps</label>
                    <input type="number" class="setting-input" id="steps" value="15" min="5" max="50">
                </div>
                <div class="setting-group">
                    <label class="setting-label">Temperature</label>
                    <input type="number" class="setting-input" id="temperature" value="0.6" min="0.1" max="2.0" step="0.1">
                </div>
                <div class="setting-group">
                    <label class="setting-label">Top-K</label>
                    <input type="number" class="setting-input" id="topK" value="20" min="1" max="100">
                </div>
            </div>
        </div>

        <div class="output-section">
            <div class="progress-info" id="progressInfo" style="display: none;">
                <span>Step: <span id="currentStep">0</span></span>
                <span>Progress: <span id="progressPercent">0%</span></span>
                <span>Masking Rate: <span id="maskingRate">100%</span></span>
            </div>
            
            <div class="progress-bar" id="progressBar" style="display: none;">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            
            <div class="step-indicator" id="stepIndicator"></div>
            
            <div class="generation-display" id="generationDisplay">
                <div class="placeholder">
                    🎯 Enter a prompt and click "Generate with Diffusion" to watch text emerge from noise!
                </div>
            </div>
            
            <div class="stats" id="stats" style="display: none;">
                <div class="stat">
                    <div class="stat-value" id="tokensRevealed">0</div>
                    <div class="stat-label">Tokens Revealed</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="timeElapsed">0s</div>
                    <div class="stat-label">Time Elapsed</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="finalLength">0</div>
                    <div class="stat-label">Final Length</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        class DiffusionInterface {
            constructor() {
                this.socket = io();
                this.currentTokens = [];
                this.tokenStates = [];
                this.isGenerating = false;
                this.startTime = null;
                this.maskToken = '[MASK]';
                
                this.initializeElements();
                this.bindEvents();
            }
            
            initializeElements() {
                this.promptInput = document.getElementById('promptInput');
                this.promptDisplay = document.getElementById('promptDisplay');
                this.currentPrompt = document.getElementById('currentPrompt');
                this.generateBtn = document.getElementById('generateBtn');
                this.progressInfo = document.getElementById('progressInfo');
                this.progressBar = document.getElementById('progressBar');
                this.progressFill = document.getElementById('progressFill');
                this.stepIndicator = document.getElementById('stepIndicator');
                this.generationDisplay = document.getElementById('generationDisplay');
                this.stats = document.getElementById('stats');
                
                // Setting inputs
                this.maxTokens = document.getElementById('maxTokens');
                this.steps = document.getElementById('steps');
                this.temperature = document.getElementById('temperature');
                this.topK = document.getElementById('topK');
            }
            
            bindEvents() {
                this.generateBtn.addEventListener('click', () => this.startGeneration());
                this.promptInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !this.isGenerating) {
                        this.startGeneration();
                    }
                });
                
                // WebSocket events
                this.socket.on('generation_event', (event) => this.handleGenerationEvent(event));
                this.socket.on('error', (error) => this.handleError(error));
            }
            
            startGeneration() {
                const prompt = this.promptInput.value.trim();
                if (!prompt) {
                    alert('Please enter a prompt');
                    return;
                }
                
                if (this.isGenerating) return;
                
                this.isGenerating = true;
                this.startTime = Date.now();
                this.generateBtn.disabled = true;
                this.generateBtn.textContent = '⏳ Generating...';
                
                // Show prompt display
                this.currentPrompt.textContent = prompt;
                this.promptDisplay.style.display = 'block';
                
                // Show progress elements
                this.progressInfo.style.display = 'flex';
                this.progressBar.style.display = 'block';
                this.stats.style.display = 'none';
                
                // Send generation request
                this.socket.emit('generate_stream', {
                    prompt: prompt,
                    max_tokens: parseInt(this.maxTokens.value),
                    steps: parseInt(this.steps.value),
                    temperature: parseFloat(this.temperature.value),
                    top_k: parseInt(this.topK.value)
                });
            }
            
            handleGenerationEvent(event) {
                switch (event.type) {
                    case 'init':
                        this.handleInit(event);
                        break;
                    case 'step':
                        this.handleStep(event);
                        break;
                    case 'reveal':
                        this.handleReveal(event);
                        break;
                    case 'stabilize':
                        this.handleStabilize(event);
                        break;
                    case 'complete':
                        this.handleComplete(event);
                        break;
                }
            }
            
            handleInit(event) {
                // Initialize tokens array (only for new tokens, not prompt)
                this.currentTokens = new Array(event.total_tokens).fill(this.maskToken);
                this.tokenStates = new Array(event.total_tokens).fill('masked');
                
                // Create step indicators
                this.createStepIndicators(event.steps);
                this.renderTokens();
            }
            
            handleStep(event) {
                // Update progress
                this.updateProgress(event.progress);
                
                // Update step indicator
                this.updateStepIndicator(event.step);
                
                // Update stats
                this.updateStats(event.step, event.masking_rate);
                
                document.getElementById('currentStep').textContent = event.step;
                document.getElementById('progressPercent').textContent = 
                    Math.round(event.progress * 100) + '%';
                document.getElementById('maskingRate').textContent = 
                    Math.round(event.masking_rate * 100) + '%';
            }
            
            handleReveal(event) {
                // Update token and state
                this.currentTokens[event.position] = event.token;
                this.tokenStates[event.position] = event.state;
                
                // Update display
                this.updateTokenDisplay(event.position, event.token, event.state);
            }
            
            handleStabilize(event) {
                // Update token state
                this.tokenStates[event.position] = event.state;
                
                // Update display
                this.updateTokenState(event.position, event.state);
            }
            
            handleComplete(event) {
                this.isGenerating = false;
                this.generateBtn.disabled = false;
                this.generateBtn.textContent = '🚀 Generate with Diffusion';
                
                // Hide progress, show stats
                this.progressInfo.style.display = 'none';
                this.progressBar.style.display = 'none';
                this.stats.style.display = 'grid';
                
                // Update final stats
                document.getElementById('finalLength').textContent = event.total_tokens;
                document.getElementById('tokensRevealed').textContent = 
                    this.currentTokens.filter(t => t !== this.maskToken).length;
                
                // Mark all steps as completed
                const steps = this.stepIndicator.querySelectorAll('.step');
                steps.forEach(step => {
                    step.classList.add('completed');
                    step.classList.remove('active');
                });
            }
            
            handleError(error) {
                this.isGenerating = false;
                this.generateBtn.disabled = false;
                this.generateBtn.textContent = '🚀 Generate with Diffusion';
                
                alert('Error: ' + error.message);
                console.error('Generation error:', error);
            }
            
            createStepIndicators(totalSteps) {
                this.stepIndicator.innerHTML = '';
                for (let i = 0; i < totalSteps; i++) {
                    const step = document.createElement('div');
                    step.className = 'step';
                    this.stepIndicator.appendChild(step);
                }
            }
            
            updateStepIndicator(currentStep) {
                const steps = this.stepIndicator.querySelectorAll('.step');
                steps.forEach((step, index) => {
                    step.classList.remove('active', 'completed');
                    if (index < currentStep - 1) {
                        step.classList.add('completed');
                    } else if (index === currentStep - 1) {
                        step.classList.add('active');
                    }
                });
            }
            
            updateProgress(ratio) {
                this.progressFill.style.width = `${ratio * 100}%`;
            }
            
            updateStats(step, maskingRate) {
                if (this.startTime) {
                    const elapsed = (Date.now() - this.startTime) / 1000;
                    document.getElementById('timeElapsed').textContent = 
                        elapsed.toFixed(1) + 's';
                }
            }
            
            renderTokens() {
                this.generationDisplay.innerHTML = this.currentTokens
                    .map((token, index) => {
                        const state = this.tokenStates[index];
                        return `<span class="token ${state}" id="token-${index}">${
                            token === this.maskToken ? '▢' : token
                        }</span>`;
                    })
                    .join(' ');
            }
            
            updateTokenDisplay(position, token, state) {
                const tokenElement = document.getElementById(`token-${position}`);
                if (tokenElement) {
                    tokenElement.textContent = token;
                    tokenElement.className = `token ${state}`;
                }
            }
            
            updateTokenState(position, state) {
                const tokenElement = document.getElementById(`token-${position}`);
                if (tokenElement) {
                    tokenElement.className = `token ${state}`;
                }
            }
        }
        
        // Initialize interface when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new DiffusionInterface();
        });
    </script>
</body>
</html>
'''


def run_app():
    """Run the Flask application"""
    if not load_models():
        print("❌ Failed to load models. Please check your model and tokenizer paths.")
        return
    
    print("\n🚀 Starting Diffusion Web Interface...")
    print("📱 Open http://localhost:5000 in your browser")
    print("🔧 WebSocket enabled for real-time generation")
    print("\n⚡ Features:")
    print("  • Real-time token revealing animation")
    print("  • Adjustable diffusion parameters")
    print("  • Step-by-step progress tracking")
    print("  • Visual masking/revealing process")
    print("  • Interactive parameter controls")
    print("  • Prompt-only display (no duplication in output)")
    print("\n🎯 Press Ctrl+C to stop the server")
    
    try:
        socketio.run(app, 
                    host='0.0.0.0', 
                    port=5000, 
                    debug=False,
                    allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")


if __name__ == '__main__':
    run_app()