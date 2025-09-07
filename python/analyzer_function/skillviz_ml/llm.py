"""
(ML Version) Functions for evaluating page difficulty, category, and specialization using an LLM.
(Caching has been removed as per user request.)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import concurrent.futures

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    print("Warning: `openai` is not installed. LLM evaluation will not work.")
    OpenAI = None

MAX_WORKERS = 5

def _call_llm_api(url: str, content: str) -> Optional[Dict[str, Any]]:
    """Makes the actual API call to the LLM to get a structured evaluation with dynamic categories."""
    if not OpenAI or not content:
        return None
    try:
        client = OpenAI()
        truncated_content = content[:12000]

        prompt = f"""
        Analyze the web page content and return a JSON object with three keys:
        1. \"difficulty_score\": A float from 1.0 (beginner) to 5.0 (expert).
        2. \"specialization_level\": A string, one of \"summary\", \"general\", or \"specialized\".
        3. \"skill_categories\": A list of 1-3 short, descriptive, and general skill tags (e.g., [\"Python\", \"Asyncio\"], [\"Music Theory\", \"Jazz Harmony\"]).

        URL: {url}
        CONTENT: {truncated_content}
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are an expert analyst of web pages. Your response must be a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        llm_response = json.loads(response.choices[0].message.content)
        
        if all(k in llm_response for k in ["difficulty_score", "specialization_level", "skill_categories"]):
            # Normalize skill tags to be consistent (e.g., lowercase, no spaces)
            llm_response['skill_categories'] = [tag.lower().replace(' ', '_') for tag in llm_response['skill_categories']]
            return llm_response
        else:
            return None

    except Exception as e:
        print(f"- Error during OpenAI API call for {url}: {e}")
        return None

def evaluate_text_content_with_llm(content: str, api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    テキストコンテンツをLLMで評価し、構造化されたJSONを返す関数。
    """
    if not OpenAI or not content:
        return None
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        truncated_content = content[:12000]

        prompt = f"""
        Analyze the web page content and return a JSON object with three keys:
        1. "difficulty_score": A float from 1.0 (beginner) to 5.0 (expert).
        2. "specialization_level": A string, one of "summary", "general", or "specialized".
        3. "skill_categories": A list of 1-3 short, descriptive, and general skill tags (e.g., ["Python", "Asyncio"], ["Music Theory", "Jazz Harmony"]).

        CONTENT: {truncated_content}
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are an expert analyst of web pages. Your response must be a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )

        llm_response = json.loads(response.choices[0].message.content)
        
        if all(k in llm_response for k in ["difficulty_score", "specialization_level", "skill_categories"]):
            llm_response['skill_categories'] = [tag.lower().replace(' ', '_') for tag in llm_response['skill_categories']]
            return llm_response
        else:
            return None

    except Exception as e:
        print(f"- Error during OpenAI API call: {e}")
        return None

def _fetch_and_evaluate_url(url: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Worker function to fetch content and evaluate a single URL."""
    try:
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator='\n', strip=True)
    except Exception as e:
        print(f"- Failed to fetch content for {url}: {e}")
        content = ""
    
    evaluation = _call_llm_api(url, content)
    return url, evaluation

def evaluate_pages_with_llm(df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
    print("Starting LLM-based page evaluation for difficulty, specialization, and dynamic skills...")
    
    urls_to_evaluate = [url for url in df['url'].dropna().unique() if isinstance(url, str)]
    evaluations = {}

    if not urls_to_evaluate:
        print("No valid URLs to evaluate.")
    else:
        print(f"Found {len(urls_to_evaluate)} URLs to evaluate in parallel (max_workers={MAX_WORKERS})...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(_fetch_and_evaluate_url, url): url for url in urls_to_evaluate}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    _, evaluation = future.result()
                    if evaluation:
                        print(f"  - Evaluated: {url}")
                        evaluations[url] = evaluation
                except Exception as exc:
                    print(f"  - {url} generated an exception: {exc}")

    print("All tasks complete.")

    df['llm_evaluation'] = df['url'].map(evaluations)
    print("LLM evaluation complete.")
    return df
