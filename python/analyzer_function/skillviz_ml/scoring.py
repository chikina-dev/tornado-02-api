
"""
(ML Version) Extracts feature vectors for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from . import rules as rules_engine
from . import terms as terms_engine

def extract_features(df: pd.DataFrame, rules: Dict[str, Any], llm_enabled: bool) -> pd.DataFrame:
    """Extracts a feature vector for each (user, skill) pair for ML modeling."""
    
    df_eval = df.copy()

    # --- Page-level feature generation ---
    if llm_enabled:
        df_eval = df_eval[df_eval['llm_evaluation'].notna()].copy()
        if df_eval.empty: return pd.DataFrame()
        eval_data = pd.json_normalize(df_eval['llm_evaluation'])
        df_eval = pd.concat([df_eval.reset_index(drop=True), eval_data.reset_index(drop=True)], axis=1)
        specialization_map = {"specialized": 0.4, "general": 0.2, "summary": 0.0}
        df_eval['depth_bonus'] = df_eval['specialization_level'].map(specialization_map).fillna(0.0)
        df_eval['difficulty'] = df_eval.apply(lambda row: rules_engine.normalize_difficulty(row.get('difficulty_score'), 1.0), axis=1)
        df_eval['skill_matches'] = df_eval['skill_categories'].apply(lambda skills: {skill: 1.0 for skill in skills})
    else:
        df_eval['domain_tier_w'] = df_eval.apply(rules_engine.infer_domain_tier, axis=1, rules=rules).map(rules.get('tiers', {})).fillna(0.4)
        df_eval['depth_bonus'] = df_eval.apply(rules_engine.compute_depth_bonus, axis=1, rules=rules)
        df_eval['difficulty'] = df_eval.apply(lambda row: rules_engine.normalize_difficulty(row.get('page_difficulty_llm'), row['domain_tier_w']), axis=1)
        df_eval['extracted_terms'] = df_eval['extracted_terms'].fillna('').astype(str)
        df_eval['skill_matches'] = df_eval['extracted_terms'].apply(lambda x: terms_engine.map_terms_to_skills(x.split(';'), rules))

    df_eval['dwell_sec'] = pd.to_numeric(df_eval['dwell_sec'], errors='coerce').fillna(0)
    df_eval['visit_count'] = pd.to_numeric(df_eval['visit_count'], errors='coerce').fillna(1)
    df_eval['expected_read_time'] = df_eval.apply(rules_engine.estimate_expected_read_time, axis=1, rules=rules)
    df_eval['read_ratio'] = np.clip(df_eval['dwell_sec'] / df_eval['expected_read_time'], 0, rules.get('scoring', {}).get('read_ratio_cap', 2.0))
    df_eval['engagement'] = (rules.get('scoring', {}).get('engagement_w_read_ratio', 0.7) * df_eval['read_ratio']) + \
                           (rules.get('scoring', {}).get('engagement_w_visits', 0.3) * np.log1p(df_eval['visit_count']))

    page_skill_rows = []
    for _, row in df_eval.iterrows():
        if not row['skill_matches']: continue
        for skill, q_match in row['skill_matches'].items():
            row_data = row.to_dict()
            row_data['skill'] = skill
            row_data['q_match'] = q_match
            page_skill_rows.append(row_data)
    
    if not page_skill_rows: return pd.DataFrame()
    df_page_skill = pd.DataFrame(page_skill_rows)

    user_skill_groups = df_page_skill.groupby(['user_id', 'skill'])

    feature_vectors = user_skill_groups.agg(
        total_dwell_sec=('dwell_sec', 'sum'),
        n_pages=('url', 'count'),
        n_unique_domains=('domain', 'nunique'),
        mean_difficulty=('difficulty', 'mean'),
        mean_engagement=('engagement', 'mean'),
        mean_depth_bonus=('depth_bonus', 'mean'),
        total_visits=('visit_count', 'sum')
    ).reset_index()

    if llm_enabled:
        df_page_skill['heuristic_score'] = (1 + df_page_skill['depth_bonus']) * df_page_skill['engagement'] * (0.5 + 0.5 * df_page_skill['difficulty']) * df_page_skill['q_match']
    else:
        df_page_skill['heuristic_score'] = df_page_skill['domain_tier_w'] * (1 + df_page_skill['depth_bonus']) * df_page_skill['engagement'] * (0.5 + 0.5 * df_page_skill['difficulty']) * df_page_skill['q_match']
    
    feature_vectors['heuristic_score_sum'] = user_skill_groups['heuristic_score'].sum().values

    return feature_vectors.fillna(0)
