"""
Visualization functions to generate the skill dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import base64
from io import BytesIO

# --- 日本語フォント設定 ---
try:
    # 使用環境に応じて一般的な日本語フォントを試す
    plt.rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meiryo', 'TakaoPGothic', 'IPAexGothic', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止
except Exception as e:
    print(f"Warning: Could not set Japanese font. Graphs may not display Japanese characters correctly. Error: {e}")
# ------------------------

def _fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def _create_radar_chart(user_data: pd.DataFrame) -> str:
    """Creates a radar chart for a single user's skills."""
    labels = user_data['skill'].values
    stats = user_data['score_0_100'].values
    confidence = user_data['confidence'].values

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='skyblue', alpha=0.4)
    ax.plot(angles, stats, color='blue', linewidth=2)
    
    ax.scatter(angles[:-1], stats[:-1], c='blue', s=50, alpha=confidence)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_title("スキルレーダー (スコア 0-100, 透明度は信頼度)", size=15, color='gray', y=1.1)
    
    plt.close(fig)
    return _fig_to_base64(fig)

def _create_heatmap(page_scores: pd.DataFrame, user_id: str) -> str:
    """Creates a heatmap of skill contribution by domain."""
    user_page_scores = page_scores[page_scores['user_id'] == user_id]
    if user_page_scores.empty:
        return ""

    heatmap_data = user_page_scores.groupby(['skill', 'domain'])['page_score'].sum().unstack(fill_value=0)
    
    top_domains = heatmap_data.sum().nlargest(20).index
    heatmap_data = heatmap_data[top_domains]

    if heatmap_data.empty:
        return ""

    fig, ax = plt.subplots(figsize=(12, max(5, len(heatmap_data) * 0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, ax=ax)
    ax.set_title(f"{user_id}様 のスキル別・ドメイン別貢献度", size=15)
    ax.set_xlabel("ドメイン", size=12)
    ax.set_ylabel("スキル", size=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.close(fig)
    return _fig_to_base64(fig)

def _create_summary_bars(df_user_skill: pd.DataFrame) -> str:
    """Creates bar charts for all users' average scores."""
    summary_data = df_user_skill.groupby('skill')['score_0_100'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_data.plot(kind='bar', ax=ax, color='teal', alpha=0.7)
    ax.set_title("全ユーザーの平均スキルスコア", size=15)
    ax.set_ylabel("平均スコア (0-100)", size=12)
    ax.set_xlabel("スキル", size=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.close(fig)
    return _fig_to_base64(fig)

def build_dashboard(df_user_skill: pd.DataFrame, page_scores: pd.DataFrame, user_id: Optional[str] = None) -> str:
    """Builds a single HTML file with all visualizations."""
    
    if user_id:
        user_data = df_user_skill[df_user_skill['user_id'] == user_id]
        if user_data.empty:
            return f"<h1>ユーザーID: {user_id} のデータが見つかりません</h1>"
        title = f"{user_id}様 のスキルダッシュボード"
        radar_b64 = _create_radar_chart(user_data)
        heatmap_b64 = _create_heatmap(page_scores, user_id)
        charts_html = f'''
            <div class="chart-container">
                <h2>スキルレーダー</h2>
                <img src="data:image/png;base64,{radar_b64}">
            </div>
            <div class="chart-container">
                <h2>ドメイン別貢献度ヒートマップ</h2>
                <img src="data:image/png;base64,{heatmap_b64}">
            </div>
        '''
    else:
        title = "全ユーザーのスキル概要ダッシュボード"
        summary_bars_b64 = _create_summary_bars(df_user_skill)
        charts_html = f'''
            <div class="chart-container">
                <h2>平均スキルスコア</h2>
                <img src="data:image/png;base64,{summary_bars_b64}">
            </div>
        '''

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 2em; background-color: #f9f9f9; color: #333; }}
            h1, h2 {{ color: #1a1a1a; }}
            .container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 2em; }}
            .chart-container {{ background-color: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="container">
            {charts_html}
        </div>
    </body>
    </html>
    """
    return html_template