
"""
Individual user skill analysis script.

Phase 1: Evaluates page difficulty and content from user logs.
Phase 2: Analyzes the evaluated data to score skills and track growth.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Optional

try:
    from .skillviz_ml import io, llm, scoring
except ImportError:
    print("Error: Could not import from 'skillviz_ml'. Make sure the package is structured correctly.")
    sys.exit(1)

def evaluate_user_pages(user_id: str, df_logs: pd.DataFrame, rules: Dict) -> Optional[pd.DataFrame]:
    """
    Phase 1: Filters logs for a user and evaluates their browsed pages using an LLM.
    
    This corresponds to: URL取得 -> 難易度評価
    
    Args:
        user_id: The user to analyze.
        df_logs: DataFrame containing all browsing logs.
        rules: The configuration dictionary.

    Returns:
        A DataFrame with page evaluation data (including difficulty, skills, etc.),
        or None if the user has no data. This is the "リスト出力".
    """
    df_user = df_logs[df_logs['user_id'] == user_id].copy()
    if df_user.empty:
        print(f"No data found for user_id: {user_id}")
        return None

    print(f"--- Phase 1: Evaluating pages for user: {user_id} ---")
    df_evaluated = llm.evaluate_pages_with_llm(df_user, rules)
    print(f"Evaluation complete. {len(df_evaluated)} pages were processed.")
    
    return df_evaluated

def analyze_user_skills(user_id: str, df_evaluated: pd.DataFrame, rules: Dict) -> str:
    """
    Phase 2: Analyzes the evaluated page data to calculate skill scores and weekly growth.

    This corresponds to: URLとその点数を受け取り -> スキル評価

    Args:
        user_id: The user being analyzed.
        df_evaluated: The DataFrame returned from the `evaluate_user_pages` function.
        rules: The configuration dictionary.

    Returns:
        A formatted string containing the full analysis report.
    """
    analysis_report = [f"--- Phase 2: Analyzing Skill Profile for User: {user_id} ---"]

    # --- Part 1: Calculate Absolute Knowledge Amount (Total) ---
    analysis_report.append("\n[1] Total Accumulated Skill Scores (Absolute Knowledge Amount)")
    
    df_features_total = scoring.extract_features(df_evaluated, rules, llm_enabled=True)

    if df_features_total.empty:
        analysis_report.append("Could not extract any skill features for this user.")
    else:
        total_scores = df_features_total[['skill', 'heuristic_score_sum']].sort_values(
            'heuristic_score_sum', ascending=False
        ).reset_index(drop=True)
        analysis_report.append("Top skills based on entire browsing history:")
        analysis_report.append(total_scores.to_markdown(index=False))

    # --- Part 2: Calculate Weekly Growth Rate ---
    analysis_report.append("\n[2] Weekly Skill Score Growth")
    
    df_evaluated['timestamp'] = pd.to_datetime(df_evaluated['timestamp'])
    df_evaluated['week'] = df_evaluated['timestamp'].dt.to_period('W')
    
    all_weekly_scores = []
    unique_weeks = sorted(df_evaluated['week'].unique())
    
    analysis_report.append(f"Analyzing {len(unique_weeks)} weeks of activity...")

    for week in unique_weeks:
        df_week_evaluated = df_evaluated[df_evaluated['week'] == week]
        if df_week_evaluated.empty:
            continue

        df_features_week = scoring.extract_features(df_week_evaluated, rules, llm_enabled=True)
        
        if not df_features_week.empty:
            df_features_week['week'] = week
            all_weekly_scores.append(df_features_week[['skill', 'week', 'heuristic_score_sum']])

    if not all_weekly_scores:
        analysis_report.append("No weekly skill progression data could be calculated.")
    else:
        df_growth = pd.concat(all_weekly_scores)
        df_pivot = df_growth.pivot_table(
            index='skill', columns='week', values='heuristic_score_sum', fill_value=0
        )
        df_growth_diff = df_pivot.diff(axis=1).fillna(0)

        analysis_report.append("\n--- Weekly Scores ---")
        analysis_report.append("Each cell shows the total score accumulated in that week.")
        analysis_report.append(df_pivot.to_markdown())
        
        analysis_report.append("\n--- Weekly Growth (Change from Previous Week) ---")
        analysis_report.append("Each cell shows the change in score compared to the previous week.")
        analysis_report.append(df_growth_diff.to_markdown())

    return "\n".join(analysis_report)

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

    # Phase 1: Evaluate pages and get the list of scores
    df_evaluated_pages = evaluate_user_pages(args.user, df_logs, rules)

    # Phase 2: Analyze skills from the evaluated pages
    if df_evaluated_pages is not None and not df_evaluated_pages.empty:
        analysis_result = analyze_user_skills(args.user, df_evaluated_pages, rules)
        print(analysis_result)
    else:
        print(f"Could not generate an analysis for user {args.user} because no pages were evaluated.")


if __name__ == "__main__":
    main()
