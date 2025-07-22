#!/usr/bin/env python3
"""
Script to create the complete project directory structure
"""

import os
from pathlib import Path

def create_project_structure():
    """Create all directories for the tiny-diffusion project"""
    
    # Define the project structure
    structure = {
        "config": [
            "__init__.py",
            "model_config.py", 
            "training_config.py",
            "curriculum_config.py"
        ],
        "data": {
            "raw": [],
            "processed": [],
            "tokenizer": []
        },
        "src": {
            "model": [
                "__init__.py",
                "attention.py",
                "transformer.py", 
                "diffusion.py"
            ],
            "data": [
                "__init__.py",
                "pipeline.py"
            ],
            "training": [
                "__init__.py",
                "trainer.py",
                "scheduler.py",
                "metrics.py"
            ],
            "evaluation": [
                "__init__.py",
                "generate.py",
                "metrics.py",
                "analysis.py"
            ]
        },
        "scripts": [
            "prepare_data.py",
            "train.py", 
            "generate.py",
            "evaluate.py"
        ],
        "notebooks": [
            "data_exploration.ipynb",
            "curriculum_analysis.ipynb", 
            "results_analysis.ipynb"
        ],
        "outputs": {
            "checkpoints": [],
            "logs": [],
            "samples": []
        }
    }
    
    def create_structure(base_path, structure_dict):
        """Recursively create directory structure"""
        for key, value in structure_dict.items():
            current_path = base_path / key
            current_path.mkdir(exist_ok=True)
            print(f"Created: {current_path}")
            
            if isinstance(value, dict):
                # Subdirectory
                create_structure(current_path, value)
            elif isinstance(value, list):
                # Files to create
                for filename in value:
                    file_path = current_path / filename
                    if not file_path.exists():
                        file_path.touch()
                        print(f"Created: {file_path}")
    
    # Create project root
    project_root = Path("tiny-diffusion")
    project_root.mkdir(exist_ok=True)
    print(f"Created: {project_root}")
    
    # Create structure
    create_structure(project_root, structure)
    
    # Create root files
    root_files = [
        "README.md",
        "requirements.txt", 
        ".gitignore"
    ]
    
    for filename in root_files:
        file_path = project_root / filename
        if not file_path.exists():
            file_path.touch()
            print(f"Created: {file_path}")
    
    print("\n✅ Project structure created successfully!")
    print(f"📁 Root directory: {project_root.absolute()}")

if __name__ == "__main__":
    create_project_structure()