"""
(ML Version) API handler to encapsulate the entire analysis pipeline.
Takes a user's log data and returns a personalized Markdown report.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Import all necessary modules from the package
from . import io
from . import llm
from . import scoring
from . import analyzer

# --- Configuration ---
# In a real API, this might be loaded once when the server starts.
CONFIG_PATH = Path(__file__).parent.parent / "config/rules.yml"
RULES = io.load_config(CONFIG_PATH)

def generate_user_report_for_api(user_logs_df: pd.DataFrame) -> str:
    """
    A single entry point for an API to generate a personalized skill report.

    Args:
        user_logs_df: A pandas DataFrame containing the log data for a SINGLE user.
                      Expected columns are the same as in logs.csv.

    Returns:
        A string containing the full analysis report in Markdown format.
    """
    if not isinstance(user_logs_df, pd.DataFrame) or user_logs_df.empty:
        return "# 分析エラー\n入力データが空か、無効な形式です。"

    user_id = user_logs_df['user_id'].iloc[0]
    print(f"[API Handler] Starting analysis for user: {user_id}")

    # --- Step 1: LLM-based Page Evaluation ---
    # This step uses caching to avoid re-evaluating the same URLs.
    # The OPENAI_API_KEY environment variable must be set in the API server's environment.
    df_evaluated = llm.evaluate_pages_with_llm(user_logs_df, RULES)

    # --- Step 2: Feature Extraction ---
    df_features = scoring.extract_features(df_evaluated, RULES, llm_enabled=True)

    if df_features.empty:
        return f"# 分析レポート ({user_id}様)\n\n分析の結果、有益なスキル情報を抽出できませんでした。"

    # --- Step 3: LLM-based Feedback Generation ---
    feedback_text = analyzer.generate_llm_feedback(df_features)

    # --- Step 4: Format Output as Markdown ---
    report_parts = []
    report_parts.append(f"# スキル分析レポート ({user_id}様)")
    report_parts.append("## あなたの興味・関心についての分析")
    report_parts.append(feedback_text)
    report_parts.append("\n---")
    report_parts.append("\n## 分析の基になったトップスキル")
    
    top_skills_md = df_features.sort_values("heuristic_score_sum", ascending=False).head(5).to_markdown(index=False)
    report_parts.append(top_skills_md)

    print(f"[API Handler] Finished analysis for user: {user_id}")
    return "\n".join(report_parts)


# --- Example Usage (how you might call this from a web framework) ---

def example_run():
    """An example of how to use the handler function."""
    # In your real API (e.g., FastAPI, Flask), you would get this data from your database
    # based on the logged-in user's ID.
    print("--- Running API Handler Example ---")
    
    # 1. Load the full log data
    full_logs_df = pd.read_csv(Path(__file__).parent.parent / "data/logs.csv")

    # 2. Get data for a specific user
    sample_user_id = 'user_g'
    user_g_data = full_logs_df[full_logs_df['user_id'] == sample_user_id].copy()

    # 3. Call the handler function
    markdown_report = generate_user_report_for_api(user_g_data)

    # 4. Print the result
    print("\n--- Generated Markdown Report ---")
    print(markdown_report)

    # You would then return this markdown_report string as the API response.

if __name__ == '__main__':
    # To run this example: python -m skillviz_ml.api_handler
    # Make sure your OPENAI_API_KEY is set.
    example_run()
