
"""
Functions for mapping extracted terms to skills.
"""

from typing import Dict, List, Any

def map_terms_to_skills(terms: List[str], rules: Dict[str, Any]) -> Dict[str, float]:
    """Maps a list of extracted terms to skills, returning the max match quality."""
    if not terms:
        return {}

    skill_map = rules.get('skills', {})
    match_weights = rules.get('match', {})
    exact_weight = match_weights.get('exact_weight', 1.0)
    fuzzy_weight = match_weights.get('fuzzy_weight', 0.7)

    skill_matches: Dict[str, float] = {}

    for term in terms:
        term_lower = term.lower().strip()
        if not term_lower:
            continue

        for skill, keywords in skill_map.items():
            current_max_quality = skill_matches.get(skill, 0.0)

            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Exact match
                if term_lower == keyword_lower:
                    skill_matches[skill] = max(current_max_quality, exact_weight)
                # Fuzzy match (substring)
                elif keyword_lower in term_lower or term_lower in keyword_lower:
                    skill_matches[skill] = max(current_max_quality, fuzzy_weight)
    
    return skill_matches
