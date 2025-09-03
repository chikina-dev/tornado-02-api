import os
import pandas as pd
from typing import Dict, Any

try:
    from openai import OpenAI
except ImportError:
    print("Warning: `openai` is not installed. LLM feedback generation will not work.")
    OpenAI = None

def generate_llm_feedback(user_data: pd.DataFrame) -> str:
    """Generates personalized feedback for a user by calling an LLM."""
    if not OpenAI:
        return "OpenAIライブラリがインストールされていないため、フィードバックを生成できません。"
    if user_data.empty:
        return "分析対象のデータがありません。"

    # --- Create a text summary of the user's data to feed to the LLM ---
    summary_parts = ["ユーザーの学習活動サマリー:"]
    
    top_skills = user_data.sort_values("heuristic_score_sum", ascending=False).head(5)

    summary_parts.append("\n[トップスキル Top 5]")
    for _, row in top_skills.iterrows():
        summary_parts.append(
            f"- スキル: {row['skill']}, "
            f"総エンゲージメントスコア: {row['heuristic_score_sum']:.2f}, "
            f"平均難易度: {row['mean_difficulty']:.2f}, "
            f"閲覧ページ数: {int(row['n_pages'])}"
        )

    summary_parts.append("\n[全体的な学習傾向]")
    try:
        mean_diff = user_data['mean_difficulty'].mean()
        mean_eng = user_data['mean_engagement'].mean()
        total_p = user_data['n_pages'].sum()
        summary_parts.append(f"- 平均難易度: {mean_diff:.2f}")
        summary_parts.append(f"- 平均エンゲージメント: {mean_eng:.2f}")
        summary_parts.append(f"- 総閲覧ページ数: {int(total_p)}")
    except KeyError as e:
        summary_parts.append(f"- (全体傾向の算出中にエラー: {e})")

    data_summary = "\n".join(summary_parts)

    # --- Call LLM to generate feedback ---
    try:
        client = OpenAI()
        prompt = f"""
        あなたは、利用者の学習活動を分析し、キャリアや学習についてアドバイスする専門家です。
        以下のデータサマリーに基づいて、このユーザーの興味、強み、そして今後の学習への提案を、
        ポジティブで分かりやすい日本語の文章で、2〜3パラグラフで作成してください。

        --- データサマリー ---
        {data_summary}
        ---------------------
        """
        
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "あなたは、学習者を勇気づける優秀なキャリアアドバイザーです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"LLMによるフィードバック生成中にエラーが発生しました: {e}"