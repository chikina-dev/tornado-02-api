"""
(ML Version) Generates personalized, LLM-written feedback for a specific user.
"""

import argparse
from pathlib import Path
import pandas as pd

from . import analyzer

def main():
    """Loads the feature matrix, filters for a user, and generates feedback."""
    parser = argparse.ArgumentParser(description="Generate LLM-written feedback for a user.")
    parser.add_argument("--features", type=Path, required=True, help="Path to the feature matrix (ml_features.parquet)")
    parser.add_argument("--user_id", type=str, required=True, help="The user ID to generate feedback for.")
    args = parser.parse_args()

    print(f"Loading feature matrix from {args.features}...")
    try:
        df = pd.read_parquet(args.features)
    except Exception as e:
        print(f"Failed to load feature file: {e}")
        return

    # Filter data for the specified user
    user_data = df[df['user_id'] == args.user_id]

    if user_data.empty:
        print(f"No data found for user_id: {args.user_id}")
        return

    print(f"\nGenerating personalized feedback for user: {args.user_id}...")
    print("--------------------------------------------------")
    
    # Generate feedback using the analyzer
    feedback_text = analyzer.generate_llm_feedback(user_data)
    
    print(feedback_text)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
