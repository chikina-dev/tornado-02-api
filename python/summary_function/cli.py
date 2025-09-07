#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize URLs / files (text, image(OCR), PDF) via OpenAI API.
- URL, Text, Image(OCR), PDFに対応
- チャンク分割 → 要約（map）→ 任意で最終統合（reduce）
- リトライ/バックオフ、requestsセッション、CLIオプション
"""

import argparse
import os
import re
import sys
import time
import csv
import json
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

# Add project root to sys.path for module imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import httpx
import asyncio
from bs4 import BeautifulSoup
from pypdf import PdfReader
import pytesseract

# 相対 import で実行したとき/スクリプト直実行の両対応
try:
    from .openai_llm import call_llm_summarize
    from .ocr_utils import ocr_image
    from ..analyzer_function.skillviz_ml.llm import evaluate_text_content_with_llm
except ImportError:
    from openai_llm import call_llm_summarize
    from ocr_utils import ocr_image
    from analyzer_function.skillviz_ml.llm import evaluate_text_content_with_llm

# -------- 共通ユーティリティ --------
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
TXT_EXTS = {'.txt', '.md', '.log'}
PDF_EXTS = {'.pdf'}
CSV_EXTS = {'.csv'}


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def sanitize_filename(name: str) -> str:
    # \w を正しく使う（元コードはバックスラッシュを二重にしていた）
    safe = re.sub(r"[^\\w\-.]+", "_", name.strip())
    return safe[:200] if len(safe) > 200 else safe


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def chunk_text(text: str, max_chars: int = 3500) -> List[str]:
    text = text or ""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start + int(max_chars * 0.5):
            cut = end
        chunks.append(text[start:cut])
        start = cut
    return chunks


# -------- コンテンツ抽出 --------

def extract_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        texts = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(texts)
    except Exception as e:
        print(f"[WARN] PDF抽出失敗: {path} ({e})")
        return ""


def read_csv_text(path: Path) -> str:
    text_lines = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                text_lines.append(','.join(row))
        return "\n".join(text_lines)
    except Exception as e:
        print(f"[WARN] CSV読み込み失敗: {path} ({e})")
        return ""


async def fetch_url(url: str, ocr_lang: str) -> Tuple[str, str]:
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "SummarizerBot/1.0"}) as client:
        r = await client.get(url)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()

        if "image" in ctype:
            ext = "." + ctype.split("/")[-1].split(";")[0]
            fname = sanitize_filename(Path(urlparse(url).path).name or "download") + ext
            tmp = Path(".tmp_downloads")
            ensure_dir(tmp)
            fpath = tmp / fname
            with open(fpath, "wb") as f:
                f.write(r.content)
            return ("image", ocr_image(fpath, lang=ocr_lang))

        if "html" in ctype:
            # 依存を減らすため標準の parser を使用
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = soup.get_text("\n")
            text = re.sub(r"\n{3,}", "\n\n", text)
            return ("html", text)

        if "text/plain" in ctype:
            return ("text", r.text)

        if "application/pdf" in ctype:
            tmp = Path(".tmp_downloads")
            ensure_dir(tmp)
            fname = sanitize_filename(Path(urlparse(url).path).name or "download") + ".pdf"
            fpath = tmp / fname
            with open(fpath, "wb") as f:
                f.write(r.content)
            return ("pdf", extract_pdf_text(fpath))

        # JSONやその他は text として扱う（httpx は .text で自動デコード）
        return (ctype or "unknown", r.text)


def read_local(path: Path, ocr_lang: str) -> Tuple[str, str]:
    ext = path.suffix.lower()
    if ext in TXT_EXTS:
        return ("text", path.read_text(encoding="utf-8", errors="ignore"))
    if ext in IMG_EXTS:
        return ("image", ocr_image(path, lang=ocr_lang))
    if ext in PDF_EXTS:
        return ("pdf", extract_pdf_text(path))
    if ext in CSV_EXTS:
        return ("csv", read_csv_text(path))
    return ("unknown", "")


# -------- LLMタスク --------
def generate_title(text: str, api_key: str, model: str) -> str:
    try:
        return call_llm_summarize(
            text=text[:4000],
            model=model,
            system_hint="あなたは優れたコピーライターです。",
            user_task_prompt=(
                "以下のテキストの内容を最も的確に表現する、簡潔で魅力的なタイトルを一つだけ生成してください。"
                "タイトル以外の余計な言葉は含めないでください。"
            ),
            api_key=api_key,
        ).strip().replace('"', '')
    except Exception as e:
        print(f"[WARN] タイトル生成中にエラー: {e}")
        return "(タイトル生成失敗)"


def summarize_one_text(text: str, api_key: str, model: str, max_chars: int) -> str:
    chunks = chunk_text(text, max_chars=max_chars)
    summaries = []
    system_hint = "あなたは優秀な学習アシスタントです。提供されたテキストの要点を、後から見返して内容をすぐに思い出せるように、構造化してまとめてください。"
    user_task_prompt = (
        "以下のテキストについて、まず全体を3〜5行で要約し、次にその要約を補足する重要なキーワードやポイントを5〜7個、箇条書きで挙げてください。"
        "このノートの目的は、内容を完全に網羅することではなく、後から見た人が「ああ、こういう内容だった」と思い出すためのトリガーとなることです。\n"
        "出力はMarkdown形式でお願いします。例えば、見出し、箇条書き、太字などを使って構造化してください。"
    )
    for idx, ch in enumerate(chunks, 1):
        if idx > 1:
            time.sleep(0.2)
        summaries.append(
            call_llm_summarize(
                ch,
                model=model,
                system_hint=system_hint,
                user_task_prompt=user_task_prompt,
                api_key=api_key,
            )
        )
    return "\n".join(summaries)


def suggest_category(text: str, api_key: str, model: str) -> str:
    try:
        return call_llm_summarize(
            text=text[:4000],
            model=model,
            system_hint="あなたはコンテンツ分類の専門家です。",
            user_task_prompt=(
                "以下のテキストに最もふさわしい、**より大まかで一般的な**カテゴリ名を、日本語で、"
                "単語または短いフレーズで一つだけ提案してください。例: 'テクノロジー', 'ビジネス', '教育', '健康', 'エンターテイメント'"
            ),
            api_key=api_key,
        ).strip().replace('"', '')
    except Exception as e:
        print(f"[WARN] カテゴリ推薦中にエラー: {e}")
        return ""

def extract_categorical_keywords(text: str, api_key: str, model: str) -> Dict[str, List[str]]:
    try:
        system_hint = "あなたは、テキストを深く理解し、内容を構造化して整理することに特化した高度なAIアナリストです。"
        user_task_prompt = ('''
            以下のテキストを分析し、次の2つのステップを実行してください。
            ステップ1: このテキストに関連する主要なカテゴリを特定してください。
            ステップ2: 特定した各カテゴリについて、そのカテゴリに直接関連する専門用語をテキスト中から5個ずつ抽出してください。

            結果は、必ず以下のJSON形式で出力してください。カテゴリ名がキー、専門用語のリストが値となります。

            {
              "カテゴリ名1": ["専門用語1", "専門用語2", "専門用語3", "専門用語4", "専門用語5"],
              "カテゴリ名2": ["専門用語A", "専門用語B", "専門用語C", "専門用語D", "専門用語E"]
            }

            他の説明や前置きは一切含めず、JSONオブジェクトのみを出力してください。
            ''')

        json_output = call_llm_summarize(
            text=text[:6000],
            model=model,
            system_hint=system_hint,
            user_task_prompt=user_task_prompt,
            api_key=api_key,
        )

        match = re.search(r'\{.*\}', json_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            if isinstance(data, dict):
                return data
        
        print(f"[WARN] カテゴリ別キーワード抽出時のJSONパースに失敗。出力: {json_output}")
        return {}

    except json.JSONDecodeError as e:
        print(f"[WARN] カテゴリ別キーワード抽出時のJSONデコードエラー: {e}")
        return {}
    except Exception as e:
        print(f"[WARN] カテゴリ別キーワード抽出中にエラー: {e}")
        return {}


# -------- 並列処理オーケストレーション --------
async def _process_one_source_fully(args: tuple) -> Dict[str, Any]:
    i, src, api_key, model, max_chars, ocr_lang = args
    source_id = f"source-{i+1}"
    print(f"[STATUS]   - ソース {i+1} ({src[:50]}...) の処理を開始")
    try:
        source_type = "unknown" # Initialize source_type
        if is_url(src):
            source_type, text = await fetch_url(src, ocr_lang=ocr_lang)
            source_name = src
        else:
            p = Path(src)
            if not p.exists():
                raise FileNotFoundError(f"Not found: {src}")
            loop = asyncio.get_running_loop()
            source_type, text = await loop.run_in_executor(None, read_local, p, ocr_lang)
            source_name = p.name

        if not text or not text.strip():
            return {"id": source_id, "name": source_name, "error": "テキスト抽出不可"}

        llm_evaluation = None
        if source_type in ["image", "pdf"]:
            print(f"[STATUS]   - ソース {i+1} ({src[:50]}...) のLLM評価を開始")
            llm_evaluation = await asyncio.get_running_loop().run_in_executor(
                None, evaluate_text_content_with_llm, text, api_key
            )
            if llm_evaluation:
                print(f"[STATUS]   - ソース {i+1} ({src[:50]}...) のLLM評価を完了")
            else:
                print(f"[WARN]   - ソース {i+1} ({src[:50]}...) のLLM評価に失敗")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as inner_executor:
            loop = asyncio.get_running_loop()
            future_title = loop.run_in_executor(inner_executor, generate_title, text, api_key, model)
            future_summary = loop.run_in_executor(inner_executor, summarize_one_text, text, api_key, model, max_chars)
            future_category = loop.run_in_executor(inner_executor, suggest_category, text, api_key, model)

            title, summary, category = await asyncio.gather(
                future_title, future_summary, future_category
            )
            result = {
                "id": source_id,
                "name": source_name,
                "title": title,
                "summary": summary,
                "category": category,
            }
            if llm_evaluation:
                result["llm_evaluation"] = llm_evaluation
        print(f"[STATUS]   - ソース {i+1} ({src[:50]}...) の処理を完了")
        return result
    except Exception as e:
        print(f"[ERROR] ソース処理中に致命的なエラー: {src} ({e})")
        return {"id": source_id, "name": src, "error": str(e)}

def aggregate_category_summary(
    category: str,
    summaries: List[str],
    api_key: str,
    model: str,
) -> str:
    """
    カテゴリごとに集約された要約を生成する。
    """
    if not summaries:
        return ""

    category_text = "\n\n---\n\n".join(summaries)
    system_hint_cat = "あなたは専門誌の編集者です。与えられたカテゴリに関する複数のテキストを統合し、構造化された詳細な要約を作成します。"
    user_task_prompt_cat = (
        f"以下のテキスト群はすべて「{category}」に関するものです。""これらの情報を統合し、重要なポイントや結論がわかるように、詳細な要約を作成してください。\n"
        "出力は必ずMarkdown形式で、以下のように構造化してください:\n" \
        f"- カテゴリ名を `### {category}` のようにレベル3見出しにします。\n" \
        "- 要約の主要な段落は通常のテキストとして記述します。\n" \
        "- 重要なキーワードやリスト項目は `-` を使った箇条書きにします。\n" \
        "- 全体を囲むタグなどは不要です。"
    )
    return call_llm_summarize(
        text=category_text,
        model=model,
        system_hint=system_hint_cat,
        user_task_prompt=user_task_prompt_cat,
        api_key=api_key,
    )


async def summarize_multiple_inputs(
    srcs: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_chars: int = 3500,
    ocr_lang: str = "jpn+eng",
    aggregate: bool = False,
    **kwargs,
) -> str:
    """
    複数のソースを処理し、集約されたサマリーを返す。
    --aggregateが指定されている場合、カテゴリ別のMarkdownサマリーを生成する。
    そうでない場合は、単純な連結テキストを返す（ただし現状はCLIからしか呼ばれない想定）。

    戻り値: (aggregated_summary)
    """
    print(f"[STATUS] 全ソースの並列処理を開始します... (全{len(srcs)}件)")

    tasks = [(i, src, api_key, model, max_chars, ocr_lang) for i, src in enumerate(srcs)]
    processed_futures = [_process_one_source_fully(task_args) for task_args in tasks]
    processed_results = await asyncio.gather(*processed_futures)
    source_results = [res for res in processed_results if res and not res.get("error")]
    print(f"[STATUS] 全ソースの並列処理が完了。({len(source_results)}件のソース処理に成功)")

    if not source_results:
        return ""

    # --aggregate が指定されていない場合は、単純に連結（ただし現状このパスは使われない）
    if not aggregate:
        all_summaries_text = "\n\n---\n\n".join([res.get("summary", "") for res in source_results if res.get("summary")])
        return all_summaries_text

    # --aggregate が指定されている場合、カテゴリ別のMarkdown要約を生成
    print("[STATUS] カテゴリ別の集約要約をMarkdown形式で生成中...")
    summaries_by_category: Dict[str, List[str]] = {}
    for res in source_results:
        cat = res.get("category") or "カテゴリなし"
        if cat not in summaries_by_category:
            summaries_by_category[cat] = []
        summaries_by_category[cat].append(res.get("summary", ""))

    md_parts = []
    for category, summaries in summaries_by_category.items():
        cat_summary_md = aggregate_category_summary(
            category=category,
            summaries=summaries,
            api_key=api_key,
            model=model,
        )
        if cat_summary_md:
            md_parts.append(cat_summary_md)

    aggregated_summary = "\n\n".join(md_parts)
    print("[STATUS] カテゴリ別集約要約の生成が完了しました。")

    print("[STATUS] 全ての処理が完了しました。")
    return aggregated_summary



def generate_summary_markdown(aggregated_summary: str) -> str:
    """
    Generates a Markdown report from the aggregated summary.
    """
    if not aggregated_summary:
        return "# 要約\n\n要約は生成されませんでした。"

    return f"# 全体要約\n\n{aggregated_summary}"


# -------- CLIラッパー --------
async def main():
    parser = argparse.ArgumentParser(
        description="Webアプリ版と同等の高度な並列処理をコマンドラインから実行します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("inputs", nargs="*", help="処理対象のURLまたはローカルファイルパス（複数可）")

    # LLM
    parser.add_argument("--model", default="gpt-4o-mini", help="LLMモデル名（例：gpt-4o-mini）")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key. 環境変数 OPENAI_API_KEY も利用可")

    # I/O
    parser.add_argument("--output-file", type=Path, default=Path("cli_summary_output.md"), help="出力Markdownファイル名")
    parser.add_argument("--max-chars", type=int, default=3500, help="1チャンクの最大文字数")
    parser.add_argument("--tesseract-cmd", default=None, help="Tesseractの実行パス（Windows等で必要なら指定）")
    parser.add_argument("--ocr-lang", default="jpn+eng", help="OCR言語（例: 'jpn', 'eng'）")
    parser.add_argument("--google-credentials", default=None, help="Google Cloud Vision APIのサービスアカウントキーJSONファイルへのパス")

    # 集約要約 ON/OFF
    parser.add_argument("--aggregate", action="store_true", help="複数の入力がある場合に、個別の要約をさらに集約して要約します。")

    args = parser.parse_args()

    if not args.inputs:
        print("[ERROR] 処理対象のURLまたはローカルファイルパスを指定してください。", file=sys.stderr)
        sys.exit(1)

    if not args.api_key:
        print("[ERROR] APIキーが必須です。--api-key引数またはOPENAI_API_KEY環境変数を設定してください。", file=sys.stderr)
        sys.exit(1)

    # --aggregateが指定されていないと意味がないのでチェック
    if not args.aggregate:
        print("[WARN] --aggregate フラグが指定されていません。集約要約は生成されず、何も出力されません。", file=sys.stderr)
        sys.exit(0)

    # OCR設定
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    cred_path_str = args.google_credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path_str:
        cred_path = Path(cred_path_str)
        if cred_path.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path.resolve())
            print("[INFO] Google Cloud Vision API の認証情報を使用します。")
        else:
            print(f"[WARN] 指定されたGoogle認証情報ファイルが見つかりません: {cred_path_str}")

    try:
        aggregated_summary = await summarize_multiple_inputs(
            srcs=args.inputs,
            api_key=args.api_key,
            model=args.model,
            max_chars=args.max_chars,
            ocr_lang=args.ocr_lang,
            aggregate=args.aggregate,
        )

        # 結果をターミナルに表示
        print("\n" + "=" * 20 + " 処理結果 " + "=" * 20)
        if aggregated_summary:
            print(f"■ 全体集約要約:\n{aggregated_summary}")
        else:
            print("要約は生成されませんでした。")
        print("\n" + "=" * 52)

        # Markdownファイルに保存
        if aggregated_summary:
            print(f"[STATUS] 最終Markdownを生成中...")
            final_md = generate_summary_markdown(aggregated_summary)
            args.output_file.write_text(final_md, encoding="utf-8")
            print(f"\n✅ 詳細な要約Markdownを {args.output_file} に保存しました。")

    except Exception as e:
        print(f"[FATAL] 処理中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)



if __name__ == "__main__":
    asyncio.run(main())
