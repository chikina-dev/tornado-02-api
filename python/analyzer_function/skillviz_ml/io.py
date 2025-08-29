"""
Data I/O functions for loading logs, configs, and saving outputs.
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

def load_logs(path: Path) -> pd.DataFrame:
    """Loads user activity logs from a CSV file."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {path}")
        raise
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def load_config(path: Path) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {path}")
        raise
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        raise

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Saves a DataFrame to a Parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Successfully saved parquet file to {path}") # MODIFIED

def save_html(content: str, path: Path) -> None:
    """Saves HTML content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully saved dashboard to {path}")

def save_markdown(content: str, path: Path) -> None:
    """Saves Markdown content to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully saved report to {path}")

def save_json(data: Dict[str, Any], path: Path) -> None:
    """Saves a dictionary to a JSON file."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Successfully saved feature spec to {path}")