import os
import time
import random
from typing import Optional

from openai import OpenAI

# .env 読み込み（.env を使わない場合は不要）
try:
    from dotenv import load_dotenv
    load_dotenv()  # 存在すれば自動読み込み
except Exception:
    pass

def with_retries(fn, *, retries=3, base=0.8):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep = base * (2 ** i) + random.uniform(0, 0.2)
            time.sleep(sleep)
    if last:
        raise last

def call_llm_summarize(
    text: str,
    model: str = "gpt-4o-mini",
    system_hint: Optional[str] = None,
    user_task_prompt: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    OpenAI API を使って要約する簡易関数。
    長文は外側でチャンク分割してから渡してください。
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY がありません。--api-key か環境変数を設定してください。(.env も可)")

    if system_hint is None:
        system_hint = (
            "あなたはプロの要約作成者です。提供されたテキストから、簡潔で有益な要約を作成してください。"
            "重要なポイントや主要なアイデアに焦点を当て、分かりやすく提示してください。"
        )

    if user_task_prompt is None:
        user_task_prompt = "以下のテキストをレビュー用に要約してください。主要なポイントと重要な結論に焦点を当ててください。"

    # user_task_promptが空でない場合のみ、テキストと結合する
    if user_task_prompt:
        user_prompt = f"{user_task_prompt}\n\n---\n{text}\n---"
    else:
        user_prompt = text

    client = OpenAI(api_key=key)

    def _run():
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_hint},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    return with_retries(_run)


def call_llm_verbalize_numbers(
    data: Any,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None
) -> str:
    """
    数値データをLLMに渡して自然言語で説明させる関数。
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY がありません。--api-key か環境変数を設定してください。(.env も可)")

    client = OpenAI(api_key=key)

    # データ形式を文字列に変換
    if isinstance(data, pd.DataFrame):
        data_str = data.to_markdown(index=True)
    elif isinstance(data, dict):
        data_str = json.dumps(data, indent=2)
    else:
        data_str = str(data)

    system_hint = (
        "あなたはデータ分析の専門家です。提供された数値データを分析し、人間が理解しやすい言葉で説明してください。"
        "特に重要なポイントや傾向を強調し、Markdown形式で出力してください。"
    )

    user_prompt = f"以下の数値データを分析し、その内容を説明してください。\n\n---\n{data_str}\n---"

    def _run():
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_hint},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    return with_retries(_run)
