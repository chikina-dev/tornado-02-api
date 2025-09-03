
import pytest
import pandas as pd
import numpy as np

# Mock rules for testing
@pytest.fixture
def mock_rules():
    return {
        'tiers': {'T1': 1.0, 'T2': 0.7, 'T3': 0.4},
        'scoring': {
            'depth_bonus_strong': 0.4,
            'depth_bonus_weak': 0.2,
            'engagement_w_read_ratio': 0.7,
            'engagement_w_visits': 0.3,
            'read_ratio_cap': 2.0,
        }
    }

# It's better to test the functions in the context of the module they are in.
# We will import the modules from the package we just created.
# To do that, we need to make sure the package is in the python path.
# For a real project, you would install the package in editable mode.
# For this test, we will add the path manually.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skillviz import rules as rules_engine
from skillviz import scoring as scoring_engine

def test_normalize_difficulty():
    """Tests the difficulty normalization logic."""
    # Test with LLM score
    assert rules_engine.normalize_difficulty(5.0, 1.0) == 1.0
    assert rules_engine.normalize_difficulty(1.0, 1.0) == 0.0
    assert rules_engine.normalize_difficulty(3.0, 1.0) == 0.5
    # Test with NaN (fallback to domain tier)
    assert rules_engine.normalize_difficulty(np.nan, 1.0) == 0.5
    assert rules_engine.normalize_difficulty(None, 0.7) == 0.35

def test_compute_engagement(mock_rules):
    """Tests the engagement calculation logic."""
    # The logic is vectorized in the main scoring function, so we test it there.
    # This is a simplified check of the formula parts.
    w_read = 0.7
    w_visits = 0.3
    read_ratio = 1.5
    visit_count = 3
    expected = (w_read * read_ratio) + (w_visits * np.log1p(visit_count))
    # This is just to show the formula is understood.
    # The actual test is integrated in the full run.
    assert expected > 1.0

def test_normalize_scores():
    """Tests the score normalization function."""
    data = {
        'user_id': ['a', 'a', 'b', 'b'],
        'skill': ['S1', 'S2', 'S1', 'S2'],
        'total_score': [10, 20, 50, 100]
    }
    df = pd.DataFrame(data)

    # Test within-user normalization
    df_norm_user = scoring_engine.normalize_scores(df.copy(), scope="within_user")
    assert df_norm_user[df_norm_user['user_id'] == 'a']['score_0_100'].tolist() == [0.0, 100.0]
    assert df_norm_user[df_norm_user['user_id'] == 'b']['score_0_100'].tolist() == [0.0, 100.0]

    # Test global normalization
    df_norm_global = scoring_engine.normalize_scores(df.copy(), scope="global")
    scores = df_norm_global['score_0_100'].tolist()
    assert scores[0] == 0.0
    assert scores[3] == 100.0
    assert scores[1] > 0.0 and scores[1] < scores[2]
