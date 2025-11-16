"""
Cleanup script to remove old experimental results and generated files
Keeps essential data and model files for next phase (text + image training)
"""

import os
import shutil
from pathlib import Path

# Define files/folders to DELETE (old experimental outputs)
DELETE_DIRS = [
    "./runs",  # Old TensorBoard runs - can be regenerated
    "./integrated_gradients/results",  # Old gradient analysis results
    "./model_output",  # Old model outputs - will regenerate new ones
]

DELETE_FILES = [
    "./Untitled.ipynb",  # Old/test notebook
]

# Files/folders to KEEP (needed for training)
KEEP_DIRS = [
    "./data",  # All data (CSV, JSON) - ESSENTIAL
    "./MMBT",  # Model code - ESSENTIAL
    "./preprocess",  # Preprocessing code
    "./integrated_gradients",  # Keep code, delete results
]

KEEP_FILES = [
    "./simple_train.py",  # Text-only training
    "./run_bert_text_only.ipynb",  # Text-only training notebook
    "./run_mmbt.ipynb",  # MMBT training notebook
    "./run_mmbt_masked_text_eval.ipynb",
    "./textBert_utils.py",  # BERT utilities
    "./image_submodel.ipynb",  # Image model training
    "./bertviz_attention.ipynb",  # Visualization
    "./README.md",
    "./LICENSE",
    "./requirements.txt",
    "./requirements_bare.txt",
    "./requirements_no_builds.txt",
    "./LAP_environment.yaml",
    "./LAP_environment_no_versions.yaml",
]

def cleanup():
    """Remove old experimental files"""
    print("=" * 60)
    print("CLEANUP: Removing old experimental files")
    print("=" * 60)
    
    deleted_size = 0
    
    # Delete directories
    for dir_path in DELETE_DIRS:
        full_path = Path(dir_path)
        if full_path.exists():
            size = sum(f.stat().st_size for f in full_path.rglob('*') if f.is_file())
            try:
                shutil.rmtree(full_path)
                print(f"✓ Deleted: {dir_path} ({size / (1024**2):.2f} MB)")
                deleted_size += size
            except Exception as e:
                print(f"✗ Error deleting {dir_path}: {e}")
        else:
            print(f"⊘ Not found: {dir_path}")
    
    # Delete files
    for file_path in DELETE_FILES:
        full_path = Path(file_path)
        if full_path.exists():
            try:
                size = full_path.stat().st_size
                full_path.unlink()
                print(f"✓ Deleted: {file_path} ({size / 1024:.2f} KB)")
                deleted_size += size
            except Exception as e:
                print(f"✗ Error deleting {file_path}: {e}")
    
    # Delete integrated_gradients/results subdirectory specifically
    ig_results = Path("./integrated_gradients/results")
    if ig_results.exists():
        try:
            shutil.rmtree(ig_results)
            print(f"✓ Deleted: ./integrated_gradients/results")
        except Exception as e:
            print(f"✗ Error deleting results: {e}")
    
    print("\n" + "=" * 60)
    print(f"CLEANUP COMPLETE: Freed {deleted_size / (1024**2):.2f} MB")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("PHASE 1 (TEXT-ONLY) CLEANUP SUMMARY")
    print("=" * 60)
    print("✓ Deleted:")
    print("  - Old TensorBoard runs (./runs/)")
    print("  - Old gradient results (./integrated_gradients/results/)")
    print("  - Old model outputs (./model_output/)")
    print("\n✓ Kept for Phase 2 (TEXT + IMAGE):")
    print("  - All data files (CSV, JSON) in ./data/")
    print("  - MMBT model code in ./MMBT/")
    print("  - Training scripts")
    print("  - ChexNet weights (./data/models/saved_chexnet.pt)")
    print("\nReady for Phase 2 training! 🚀")
    print("=" * 60)

if __name__ == "__main__":
    cleanup()
