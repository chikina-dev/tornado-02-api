
"""
Functions for inferring page attributes based on rules.
"""

import pandas as pd
from typing import Dict, Any, Optional
import re

def infer_domain_tier(row: pd.Series, rules: Dict[str, Any]) -> str:
    """Infers the domain tier for a given log entry."""
    if pd.notna(row.get('domain_tier')):
        return str(row['domain_tier'])

    domain = row.get('domain', '')
    if not domain:
        return 'T3'

    # Check overrides first
    for override in rules.get('tier_overrides', []):
        if override['domain'] == domain:
            return override['tier']

    # Heuristics for common high-quality domains
    if any(domain.endswith(suffix) for suffix in ['.gov', '.edu', '.org', '.ac.jp']):
        return 'T1'
    if any(sub in domain for sub in ['readthedocs', 'github.io']):
        return 'T2'

    return 'T3' # Default

def compute_depth_bonus(row: pd.Series, rules: Dict[str, Any]) -> float:
    """Computes the depth bonus based on URL path and title."""
    path = str(row.get('path', '')).lower()
    title = str(row.get('title', '')).lower()
    text_to_check = path + " " + title

    strong_keywords = rules.get('depth_keywords', {}).get('strong', [])
    if any(keyword in text_to_check for keyword in strong_keywords):
        return rules.get('scoring', {}).get('depth_bonus_strong', 0.0)

    weak_keywords = rules.get('depth_keywords', {}).get('weak', [])
    if any(keyword in text_to_check for keyword in weak_keywords):
        return rules.get('scoring', {}).get('depth_bonus_weak', 0.0)

    return 0.0

def estimate_expected_read_time(row: pd.Series, rules: Dict[str, Any]) -> float:
    """Estimates the expected read time. Simplified for now."""
    # A more sophisticated version could use word count from page content.
    # Here, we use a simple heuristic based on title length.
    title_len = len(str(row.get('title', '')))
    default_time = rules.get('scoring', {}).get('read_time_default_sec', 40)
    
    # Simple heuristic: 5 seconds per character in title, but capped.
    estimated_time = max(default_time, title_len * 0.5) 
    return estimated_time

def normalize_difficulty(page_difficulty_llm: Optional[float], domain_tier_w: float) -> float:
    """Normalizes page difficulty to a 0-1 scale."""
    if pd.isna(page_difficulty_llm) or page_difficulty_llm is None:
        return 0.5 * domain_tier_w  # Fallback to domain-based difficulty
    else:
        # Normalize 1-5 scale to 0-1
        return (float(page_difficulty_llm) - 1) / 4
