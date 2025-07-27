#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization (HPO) Script using Optuna

This script leverages Optuna to implement a modern, efficient search for the
best curriculum settings. It replaces a brute-force grid search with:
- Bayesian Sampling (TPE): Intelligently focuses on promising parameter regions.
- Multi-Fidelity Pruning (Hyperband): Starts many trials cheaply and only
  allocates more resources (epochs) to the best-performing ones, saving
  significant compute time.
- Composite Objective: Optimizes for a balanced score of loss, semantic
  quality (BERTScore), and diversity (Distinct-2, Distinct-3), aligning the search
  with final evaluation goals.

Usage:
    # Make sure you have prepared the data first:
    python scripts/train.py --book data/raw/your_book.txt --debug

    # Then run the hyperparameter search:
    python scripts/hyperparameter_search.py --n-trials 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
# --- NEW: Import for visualization ---
import optuna.visualization as vis
# --- END NEW ---
import torch
import random
import numpy as np

from config import ProjectConfig
from src.data import DataPipeline
from src.trainer import create_trainer_from_config, CurriculumTrainer
from src.evaluation import EvaluationSuite

# Configure logging for the search script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - HPO - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
# Silence Optuna's informational logs unless there's a warning
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_lightweight_eval(trainer: CurriculumTrainer, data_pipeline: DataPipeline) -> Dict[str, float]:
    """
    Runs a fast, minimal evaluation to get metrics for the composite score.
    This function is designed to be fast and is called once per trial. Heavy metrics
    like perplexity are avoided.
    """
    # Generate a few samples
    prompts = ["The monster", "Victor felt", "In the laboratory", "A sense of dread"]
    generated_texts = []
    if trainer.evaluation_suite:
        generated_texts = [
            trainer.evaluation_suite.generator.generate(p, max_new_tokens=30).generated_text
            for p in prompts
        ]
    
    if not generated_texts:
        return {"bert_f1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0}

    # Sample reference sentences for stable BERTScore
    if data_pipeline.segments:
        num_ref_sentences = min(200, len(data_pipeline.segments))
        reference_pool = [
            seg.text for seg in random.sample(data_pipeline.segments, num_ref_sentences)
        ]
    else:
        reference_pool = [data_pipeline.raw_text]
    
    # Match number of references to predictions for BERTScore
    references_for_bertscore = random.choices(reference_pool, k=len(generated_texts))

    # Calculate a minimal set of metrics
    bert_results = trainer.evaluation_suite.benchmarker.bertscore_benchmark(
        generated_texts, 
        references_for_bertscore
    )
    
    all_generated_text = " ".join(generated_texts)
    style_results = trainer.evaluation_suite.analyzer.analyze_text(all_generated_text)
    
    return {
        "bert_f1": bert_results.get('bert_score_f1', 0.0),
        "distinct_2": style_results.distinct_2,
        "distinct_3": style_results.distinct_3,
    }


def objective(trial: optuna.trial.Trial, base_config: ProjectConfig, data_pipeline: DataPipeline) -> float:
    """
    The main objective function for an Optuna trial.
    This function defines the search space, runs a training trial, and returns
    a score to be minimized.
    """
    try:
        # 1. Define the Search Space
        lr = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
        e1 = trial.suggest_int("epochs_s1", 1, 3)
        e2 = trial.suggest_int("epochs_s2", 2, 5)
        e3 = trial.suggest_int("epochs_s3", 3, 8)
        
        s1_mask_min = trial.suggest_float("s1_mask_min", 0.10, 0.40)
        s1_mask_max = trial.suggest_float("s1_mask_max", s1_mask_min + 0.05, 0.50)
        
        s3_mask_min = trial.suggest_float("s3_mask_min", 0.05, 0.15)
        s3_mask_max = trial.suggest_float("s3_mask_max", s3_mask_min + 0.02, s3_mask_min + 0.05)

        # 2. Create the configuration for this trial
        overrides = {
            'training.learning_rate': lr,
            'curriculum.stages[0].epochs': e1,
            'curriculum.stages[1].epochs': e2,
            'curriculum.stages[2].epochs': e3,
            'curriculum.stages[0].masking_rate_range': (s1_mask_min, s1_mask_max),
            'curriculum.stages[2].masking_rate_range': (s3_mask_min, s3_mask_max),
        }
        trial_config = base_config.override(**overrides)
        config_dict = trial_config.to_dict()

        # 3. Create and run the trainer, passing the Optuna trial for pruning
        trainer = create_trainer_from_config(
            config_dict, 
            data_pipeline, 
            device='auto', 
            debug_mode=True, 
            optuna_trial=trial, # This enables pruning hooks in the trainer
            run_advanced_eval=True # Ensure EvaluationSuite is created
        )
        
        # 4. Multi-Fidelity Training with Pruning
        # The trainer will report loss after each epoch and check for pruning signals.
        stage_results = trainer.train_full_curriculum()

        if not stage_results:
            return float('inf')

        # 5. Calculate the Composite Score (only for unpruned trials)
        final_stage_result = stage_results[-1]
        val_loss = final_stage_result.best_loss

        eval_metrics = run_lightweight_eval(trainer, data_pipeline)
        bert_f1 = eval_metrics["bert_f1"]
        distinct_2 = eval_metrics["distinct_2"]
        distinct_3 = eval_metrics["distinct_3"]
        
        composite_score = (
            0.50 * val_loss +               # 50% weight on validation loss
            0.25 * (1 - bert_f1) +          # 25% weight on semantic fidelity
            0.15 * (1 - distinct_2) +       # 15% weight on bigram diversity
            0.10 * (1 - distinct_3)         # 10% weight on trigram diversity
        )
        
        logger.info(f"Trial {trial.number} finished. Loss: {val_loss:.4f}, BERT F1: {bert_f1:.4f}, D-2: {distinct_2:.4f}, D-3: {distinct_3:.4f}, Score: {composite_score:.4f}")
        
        return composite_score

    except optuna.TrialPruned:
        # Let Optuna handle the pruned trial
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with an error: {e}", exc_info=True)
        return float('inf')


def main():
    """
    Main function to orchestrate the hyperparameter search.
    """
    parser = argparse.ArgumentParser(description="Advanced Hyperparameter Optimization Script")
    parser.add_argument('--book', type=str, default='data/raw/frankenstein.txt', help='Path to the book text file.')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of optimization trials to run.')
    parser.add_argument('--study-name', type=str, default='tiny-diffusion-hpo', help='Name for the Optuna study.')
    args = parser.parse_args()

    # Seed everything for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Starting advanced hyperparameter search with Optuna...")

    base_config = ProjectConfig.debug()

    logger.info("Loading pre-processed data...")
    try:
        data_pipeline = DataPipeline.load_processed_data(base_config.to_dict(), debug_mode=True)
        logger.info("Successfully loaded existing debug data.")
    except FileNotFoundError:
        logger.warning("No pre-processed debug data found. Preparing it now...")
        data_pipeline = DataPipeline(base_config.to_dict(), debug_mode=True)
        data_pipeline.process_book(args.book, save_dir="data/processed")
        logger.info("Data preparation complete.")

    # Setup and run the Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=HyperbandPruner(min_resource=1, max_resource=8, reduction_factor=3),
    )

    objective_func = lambda trial: objective(trial, base_config, data_pipeline)

    study.optimize(objective_func, n_trials=args.n_trials, timeout=7200)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    logger.info("--- Hyperparameter Search Complete ---")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Composite Score): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- NEW: Generate and show intermediate values plot ---
    if vis.is_available():
        print("\nGenerating intermediate values plot to show pruning...")
        fig = vis.plot_intermediate_values(study)
        fig.show()
    else:
        print("\nInstall plotly to visualize pruning: pip install plotly")
    # --- END NEW ---


if __name__ == '__main__':
    main()
