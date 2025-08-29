
"""
Individual user skill analysis script.

Calculates and displays:
1. The total absolute skill scores for a given user over all time.
2. The weekly growth of skill scores for that user.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime, timedelta

try:
    from .skillviz_ml import io, llm, scoring
except ImportError:
    print("Error: Could not import from 'skillviz_ml'. Make sure the package is structured correctly.")
    sys.exit(1)

def analyze_user_progress(user_id: str, df_logs: pd.DataFrame, rules: dict):
    """
    Performs and prints the absolute and weekly growth analysis for a user.
    """
    df_user = df_logs[df_logs['user_id'] == user_id].copy()
    if df_user.empty:
        print(f"No data found for user_id: {user_id}")
        return

    print(f"--- Analyzing progress for user: {user_id} ---")

    # --- Part 1: Calculate Absolute Knowledge Amount (Total) ---
    print("\n[1] Total Accumulated Skill Scores (Absolute Knowledge Amount)")
    
    # Run evaluation and scoring on the user's entire history
    df_evaluated_total = llm.evaluate_pages_with_llm(df_user, rules)
    df_features_total = scoring.extract_features(df_evaluated_total, rules, llm_enabled=True)

    if df_features_total.empty:
        print("Could not extract any skill features for this user.")
    else:
        total_scores = df_features_total[['skill', 'heuristic_score_sum']].sort_values(
            'heuristic_score_sum', ascending=False
        ).reset_index(drop=True)
        print("Top skills based on entire browsing history:")
        print(total_scores.to_markdown(index=False))

    # --- Part 2: Calculate Weekly Growth Rate ---
    print("\n[2] Weekly Skill Score Growth")
    
    df_user['timestamp'] = pd.to_datetime(df_user['timestamp'])
    df_user['week'] = df_user['timestamp'].dt.to_period('W')
    
    all_weekly_scores = []
    
    unique_weeks = sorted(df_user['week'].unique())
    
    print(f"Analyzing {len(unique_weeks)} weeks of activity...")

    for week in unique_weeks:
        print(f"  - Processing {week.start_time.date()} to {week.end_time.date()}...")
        df_week = df_user[df_user['week'] == week]
        
        # We re-use the page evaluations from the total run to avoid re-calling the LLM
        evaluated_urls = df_evaluated_total['url'].unique()
        df_week_evaluated = df_evaluated_total[df_evaluated_total['url'].isin(df_week['url'].unique())]

        if df_week_evaluated.empty:
            continue

        df_features_week = scoring.extract_features(df_week_evaluated, rules, llm_enabled=True)
        
        if not df_features_week.empty:
            df_features_week['week'] = week
            all_weekly_scores.append(df_features_week[['skill', 'week', 'heuristic_score_sum']])

    if not all_weekly_scores:
        print("No weekly skill progression data could be calculated.")
        return

    df_growth = pd.concat(all_weekly_scores)
    
    # Pivot to show skills as rows and weeks as columns
    df_pivot = df_growth.pivot_table(
        index='skill',
        columns='week',
        values='heuristic_score_sum',
        fill_value=0
    )
    
    # Calculate week-over-week growth
    # The .diff(axis=1) calculates the difference between a column and the one before it
    df_growth_diff = df_pivot.diff(axis=1).fillna(0)

    print("\n--- Weekly Scores ---")
    print("Each cell shows the total score accumulated in that week.")
    print(df_pivot.to_markdown())
    
    print("\n--- Weekly Growth (Change from Previous Week) ---")
    print("Each cell shows the change in score compared to the previous week.")
    print(df_growth_diff.to_markdown())


def main():
    parser = argparse.ArgumentParser(description="Analyze an individual user's skill knowledge and weekly growth.")
    parser.add_argument("--config", type=Path, default=Path("analyzer_function/config/rules.yml"), help="Path to the config file (rules.yml)")
    parser.add_argument("--input", type=Path, default=Path("analyzer_function/data/logs.csv"), help="Path to the input data (logs.csv)")
    parser.add_argument("--user", type=str, required=True, help="The user_id to analyze.")
    
    args = parser.parse_args()

    print("Loading data...")
    try:
        rules = io.load_config(args.config)
        df_logs = io.load_logs(args.input)
    except Exception as e:
        print(f"Failed to load initial files. Aborting: {e}")
        return

    analyze_user_progress(args.user, df_logs, rules)


if __name__ == "__main__":
    main()
