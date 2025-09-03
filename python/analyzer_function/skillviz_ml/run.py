"""
(ML Version) Feature extraction pipeline.
"""

import argparse
from pathlib import Path
import time

from . import io
from . import scoring
from . import llm

def main():
    """Main function to run the feature extraction pipeline."""
    parser = argparse.ArgumentParser(description="Extracts feature vectors from browsing logs for ML modeling.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the config file (rules.yml)")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input data (logs.csv)")
    parser.add_argument("--out", type=Path, required=True, help="Path to the output directory")
    parser.add_argument("--llm-eval", action="store_true", help="Enable LLM-based page evaluation.")

    args = parser.parse_args()
    start_time = time.time()
    print(f"Starting Feature Extraction pipeline...")

    # 1. Load data
    print(f"Loading config from {args.config} and data from {args.input}")
    try:
        rules = io.load_config(args.config)
        df_logs = io.load_logs(args.input)
    except Exception as e:
        print(f"Failed to load initial files. Aborting: {e}")
        return

    # 2. LLM Evaluation (if enabled)
    if args.llm_eval:
        df_logs = llm.evaluate_pages_with_llm(df_logs, rules)
    
    # 3. Core Feature Extraction
    print("Extracting features for each (user, skill) pair...")
    df_features = scoring.extract_features(df_logs, rules, llm_enabled=args.llm_eval)

    if df_features.empty:
        print("Could not extract any features. Aborting.")
        return

    # 4. Save final feature matrix
    output_path = args.out / "ml_features.parquet"
    print(f"Saving feature matrix to {output_path}")
    io.save_parquet(df_features, output_path)

    end_time = time.time()
    print(f"\nFeature extraction finished in {end_time - start_time:.2f} seconds.")
    print(f"Feature matrix is ready for model training.")
    print(f"Next step: python -m skillviz_ml.train --features {output_path}")

if __name__ == "__main__":
    main()