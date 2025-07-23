#!/usr/bin/env python3
"""
Project Cleanup Script
Removes outputs, processed data, and cache files for fresh training runs
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Set
import time


class ProjectCleaner:
    """Clean up project directories and files safely"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.to_remove: List[Path] = []
        self.protected_files = {
            # Source code
            "src", "config", "scripts",
            # Documentation and config
            "README.md", "requirements.txt", "config.yaml", 
            # Raw data (keep original books/texts)
            "data/raw",
            # Git and IDE
            ".git", ".gitignore", ".vscode", "__pycache__"
        }
    
    def scan_outputs(self) -> List[Path]:
        """Find output directories and files"""
        output_patterns = [
            "outputs",
            "runs",
            "logs", 
            "checkpoints",
            "models",
            "results"
        ]
        
        found = []
        for pattern in output_patterns:
            path = self.project_root / pattern
            if path.exists():
                found.append(path)
        
        return found
    
    def scan_processed_data(self) -> List[Path]:
        """Find processed data files and directories"""
        data_dir = self.project_root / "data"
        if not data_dir.exists():
            return []
        
        processed_items = []
        
        # Common processed data patterns
        patterns = [
            "data/processed",
            "data/cache", 
            "data/embeddings",
            "data/tokenizer",
            "data/*.pkl",
            "data/*.npy",
            "data/*.pt",
            "data/*.pth"
        ]
        
        for pattern in patterns:
            if "*" in pattern:
                # Glob pattern
                base_path = self.project_root / pattern.split("*")[0]
                if base_path.exists():
                    extension = pattern.split("*")[1]
                    for item in base_path.glob(f"*{extension}"):
                        processed_items.append(item)
            else:
                # Direct path
                path = self.project_root / pattern
                if path.exists():
                    processed_items.append(path)
        
        return processed_items
    
    def scan_cache_files(self) -> List[Path]:
        """Find cache and temporary files"""
        cache_patterns = [
            "cache",
            "__pycache__/",
            ".pytest_cache",
            "*.pyc",
            "*.pyo", 
            "*~",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        found = []
        for pattern in cache_patterns:
            if "*" in pattern:
                # Find files matching pattern recursively
                for item in self.project_root.rglob(pattern):
                    if not self._is_protected(item):
                        found.append(item)
            else:
                path = self.project_root / pattern
                if path.exists() and not self._is_protected(path):
                    found.append(path)
        
        return found
    
    def scan_tensorboard_logs(self) -> List[Path]:
        """Find tensorboard log directories"""
        tb_patterns = [
            "tensorboard",
            "tb_logs",
            "runs/*/tensorboard", 
            "outputs/*/tensorboard"
        ]
        
        found = []
        for pattern in tb_patterns:
            if "*" in pattern:
                parts = pattern.split("*")
                base = self.project_root / parts[0]
                if base.exists():
                    for item in base.glob("*" + parts[1]):
                        found.append(item)
            else:
                path = self.project_root / pattern
                if path.exists():
                    found.append(path)
        
        return found
    
    def _is_protected(self, path: Path) -> bool:
        """Check if path should be protected from deletion"""
        rel_path = path.relative_to(self.project_root)
        
        # Check if any part of path is protected
        for part in rel_path.parts:
            if part in self.protected_files:
                return True
        
        # Check if path starts with protected directory
        for protected in self.protected_files:
            if str(rel_path).startswith(protected):
                return True
        
        return False
    
    def get_cleanup_plan(self, targets: List[str]) -> dict:
        """Generate cleanup plan based on targets"""
        plan = {
            'outputs': [],
            'processed_data': [],
            'cache': [],
            'tensorboard': [],
            'total_size': 0
        }
        
        if 'outputs' in targets or 'all' in targets:
            plan['outputs'] = self.scan_outputs()
        
        if 'data' in targets or 'all' in targets:
            plan['processed_data'] = self.scan_processed_data()
        
        if 'cache' in targets or 'all' in targets:
            plan['cache'] = self.scan_cache_files()
        
        if 'tensorboard' in targets or 'logs' in targets or 'all' in targets:
            plan['tensorboard'] = self.scan_tensorboard_logs()
        
        # Calculate total size
        all_items = (plan['outputs'] + plan['processed_data'] + 
                    plan['cache'] + plan['tensorboard'])
        
        for item in all_items:
            if item.exists():
                if item.is_file():
                    plan['total_size'] += item.stat().st_size
                elif item.is_dir():
                    plan['total_size'] += self._get_dir_size(item)
        
        return plan
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory"""
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except (PermissionError, OSError):
            pass  # Skip inaccessible files
        return total
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def preview_cleanup(self, plan: dict):
        """Show what will be deleted"""
        print("🧹 CLEANUP PLAN")
        print("=" * 50)
        
        total_items = 0
        
        for category, items in plan.items():
            if category == 'total_size':
                continue
                
            if items:
                print(f"\n📁 {category.upper().replace('_', ' ')}:")
                for item in items:
                    rel_path = item.relative_to(self.project_root)
                    if item.is_dir():
                        print(f"   📂 {rel_path}/")
                    else:
                        size = self._format_size(item.stat().st_size)
                        print(f"   📄 {rel_path} ({size})")
                    total_items += 1
        
        print(f"\n📊 SUMMARY:")
        print(f"   Items to delete: {total_items}")
        print(f"   Total size: {self._format_size(plan['total_size'])}")
        
        if total_items == 0:
            print("\n✨ Nothing to clean - project is already clean!")
            return False
        
        return True
    
    def execute_cleanup(self, plan: dict, confirm: bool = True) -> bool:
        """Execute the cleanup plan"""
        all_items = []
        for category, items in plan.items():
            if category != 'total_size':
                all_items.extend(items)
        
        if not all_items:
            print("✨ Nothing to clean!")
            return True
        
        if confirm:
            response = input(f"\n⚠️  DELETE {len(all_items)} items? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Cleanup cancelled")
                return False
        
        print("\n🗑️  Deleting files...")
        deleted_count = 0
        errors = []
        
        for item in all_items:
            try:
                if item.exists():
                    if item.is_dir():
                        shutil.rmtree(item)
                        print(f"   🗂️  Deleted directory: {item.relative_to(self.project_root)}")
                    else:
                        item.unlink()
                        print(f"   📄 Deleted file: {item.relative_to(self.project_root)}")
                    deleted_count += 1
            except Exception as e:
                error_msg = f"Failed to delete {item}: {e}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
        
        print(f"\n✅ Cleanup complete!")
        print(f"   Deleted: {deleted_count} items")
        print(f"   Freed: {self._format_size(plan['total_size'])}")
        
        if errors:
            print(f"   Errors: {len(errors)}")
            print("\n⚠️  Some items could not be deleted:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"      {error}")
            if len(errors) > 5:
                print(f"      ... and {len(errors) - 5} more")
        
        return len(errors) == 0
    
    def quick_clean(self):
        """Quick cleanup for most common case"""
        print("🚀 QUICK CLEAN (outputs + cache)")
        plan = self.get_cleanup_plan(['outputs', 'cache'])
        
        if self.preview_cleanup(plan):
            return self.execute_cleanup(plan, confirm=True)
        return True
    
    def deep_clean(self):
        """Deep cleanup - everything except source code"""
        print("🔥 DEEP CLEAN (outputs + data + cache + logs)")
        plan = self.get_cleanup_plan(['all'])
        
        if self.preview_cleanup(plan):
            print("\n⚠️  WARNING: This will delete ALL processed data!")
            print("   You'll need to re-run data preparation and training from scratch.")
            return self.execute_cleanup(plan, confirm=True)
        return True


def main():
    parser = argparse.ArgumentParser(description="Clean up project for fresh training")
    
    parser.add_argument('--targets', nargs='+', 
                       choices=['outputs', 'data', 'cache', 'tensorboard', 'logs', 'all'],
                       default=['outputs', 'cache'],
                       help='What to clean (default: outputs cache)')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without deleting')
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick clean (outputs + cache)')
    
    parser.add_argument('--deep', action='store_true',
                       help='Deep clean (everything except source)')
    
    parser.add_argument('--yes', action='store_true',
                       help='Skip confirmation prompts')
    
    parser.add_argument('--project-root', default='.',
                       help='Project root directory (default: current)')
    
    args = parser.parse_args()
    
    # Create cleaner
    cleaner = ProjectCleaner(args.project_root)
    
    print(f"🧹 Project Cleaner")
    print(f"📁 Project root: {cleaner.project_root}")
    print(f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if args.quick:
            success = cleaner.quick_clean()
        elif args.deep:
            success = cleaner.deep_clean()
        else:
            # Custom targets
            plan = cleaner.get_cleanup_plan(args.targets)
            
            if args.dry_run:
                cleaner.preview_cleanup(plan)
                print("\n🔍 DRY RUN - Nothing was deleted")
                success = True
            else:
                if cleaner.preview_cleanup(plan):
                    success = cleaner.execute_cleanup(plan, confirm=not args.yes)
                else:
                    success = True
        
        if success:
            print("\n🎉 Cleanup successful! Ready for fresh training.")
        else:
            print("\n❌ Cleanup completed with errors.")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Cleanup interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Cleanup failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
