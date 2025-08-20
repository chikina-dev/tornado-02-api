import argparse
import json
import os
from analyzer import analyze_user_profile
from llm_core import call_llm_summarize # analyzer.pyで使われているので、ここでもインポートが必要

def main():
    parser = argparse.ArgumentParser(description="Analyze user profile from a JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file containing categories and terms data.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="Your API key for the LLM service. Can also be set via OPENAI_API_KEY environment variable.")
    parser.add_argument("--model", default="gpt-4o-mini", help="The LLM model to use (default: gpt-4o-mini).")

    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] APIキーが必須です。--api_key引数またはOPENAI_API_KEY環境変数を設定してください。", file=sys.stderr)
        return

    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found at '{args.json_file}'")
        return

    try:
        with open(args.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.json_file}'. Please ensure it's a valid JSON file.")
        return
    except Exception as e:
        print(f"Error reading file '{args.json_file}': {e}")
        return

    categories_data = data.get("categories_data", {})
    terms_data = data.get("terms_data", {})

    if not categories_data and not terms_data:
        print("Warning: No 'categories_data' or 'terms_data' found in the JSON file. Analysis might be limited.")

    print("Analyzing user profile...")
    analysis_result = analyze_user_profile(
        categories_data=categories_data,
        terms_data=terms_data,
        api_key=args.api_key,
        model=args.model
    )

    print("\n--- Analysis Result ---")
    print(analysis_result)
    print("-----------------------")

if __name__ == "__main__":
    main()
