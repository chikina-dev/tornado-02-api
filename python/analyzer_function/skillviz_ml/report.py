
"""
Functions for generating the run summary report.
"""

from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from python.summary_function.openai_llm import call_llm_verbalize_numbers

def write_report(
    run_metadata: Dict[str, Any],
    df_user_skill: Optional[pd.DataFrame] = None,
    output_path: str = "",
    feedback_text: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """Generates a summary of the execution in Markdown format."""
    
    report = []
    report.append("# SkillViz 実行レポート")
    report.append(f"- **実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **入力ファイル**: `{run_metadata.get('input_file', 'N/A')}`")
    report.append(f"- **出力ディレクトリ**: `{run_metadata.get('output_dir', 'N/A')}`")
    report.append(f"- **処理総行数**: {run_metadata.get('total_rows', 'N/A')}")
    report.append(f"- **ユニークユーザー数**: {run_metadata.get('unique_users', 'N/A')}`")
    report.append(f"- **正規化スコープ**: `{run_metadata.get('norm_scope', 'N/A')}`")
    report.append(f"- **LLM評価**: `{'有効' if run_metadata.get('llm_eval_enabled') else '無効'}`")

    if df_user_skill is not None and not df_user_skill.empty:
        report.append("\n## ユーザースキルスコア Top 5 (全体)")
        top_5 = df_user_skill.nlargest(5, 'total_score')
        report.append(top_5[[ 'user_id', 'skill', 'total_score', 'confidence']].to_markdown(index=False))
        
        # LLMによる言語化
        if api_key:
            verbalized_top_5 = call_llm_verbalize_numbers(top_5, api_key=api_key)
            report.append("\n### LLMによるTop 5の分析")
            report.append(verbalized_top_5)

        report.append("\n## 主要指標（全体平均）")
        # **MODIFIED**: Ensure authority_ratio is not here and engagement_quality is.
        kpi_columns = [
            'engagement_quality', 'depth_rate', 'revisit_index', 
            'difficulty_exposure', 'diversity', 'confidence'
        ]
        # Filter for columns that actually exist in the dataframe to prevent errors
        kpi_columns_exist = [col for col in kpi_columns if col in df_user_skill.columns]
        kpi_df = df_user_skill[kpi_columns_exist].mean().to_frame(name='平均値')
        kpi_df.index.name = "指標"
        kpi_df = kpi_df.rename(index={
            'engagement_quality': 'エンゲージメント品質',
            'depth_rate': '専門性スコア',
            'revisit_index': '再訪インデックス',
            'difficulty_exposure': '平均難易度',
            'diversity': '多様性スコア',
            'confidence': '信頼度'
        })
        report.append(kpi_df.to_markdown())

        # LLMによる言語化
        if api_key:
            verbalized_kpi = call_llm_verbalize_numbers(kpi_df, api_key=api_key)
            report.append("\n### LLMによる主要指標の分析")
            report.append(verbalized_kpi)
    else:
        report.append("\n## 結果\nスキルスコアは生成されませんでした。")

    if feedback_text:
        report.append("\n---")
        report.append(f"\n## あなたの興味・関心についての分析 (ユーザー: {run_metadata.get('user_id')})")
        report.append(feedback_text)

    report.append(f"\n## 出力ファイル")
    report.append(f"- **スコア (Parquet)**: `{output_path}.parquet`")
    report.append(f"- **ダッシュボード (HTML)**: `{output_path}.html`")

    return "\n".join(report)
