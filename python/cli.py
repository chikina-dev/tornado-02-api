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
import random
import csv
import json
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import sys
from pathlib import Path

# Add project root to sys.path for module imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import httpx
import asyncio
from bs4 import BeautifulSoup
from pypdf import PdfReader
import json # New import

from openai_llm import call_llm_summarize
from ocr_utils import ocr_image # Added this import
import markdown

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
    safe = re.sub(r"[^\w\-.]+", "_", name.strip())
    return safe[:200] if len(safe) > 200 else safe


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def chunk_text(text: str, max_chars: int = 3500):
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
    text = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                text.append(','.join(row))
        return "\n".join(text)
    except Exception as e:
        print(f"[WARN] CSV読み込み失敗: {path} ({e})")
        return ""

async def fetch_url(url: str, ocr_lang: str, google_credentials_path: Optional[str]) -> tuple[str, str]:
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "SummarizerBot/1.0"}) as client:
        # httpx does not have built-in retry like requests.adapters.Retry.
        # For simplicity, we omit complex retry logic for now.
        r = await client.get(url)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "").lower()
        if "image" in ctype:
            ext = "." + ctype.split("/")[-1].split(";")[0]
            fname = sanitize_filename(Path(urlparse(url).path).name or "download") + ext
            tmp = Path(".tmp_downloads"); ensure_dir(tmp); fpath = tmp / fname
            # File I/O is synchronous, but for small files, it's often acceptable in async context
            # For large files, aiofiles would be better.
            with open(fpath, "wb") as f: f.write(r.content)
            return ("image", ocr_image(fpath, lang=ocr_lang, google_credentials_path=google_credentials_path))
        elif "html" in ctype:
            soup = BeautifulSoup(r.text, "lxml")
            for tag in soup(["script", "style", "noscript"]): tag.extract()
            return ("html", re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")))
        elif "text/plain" in ctype:
            return ("text", r.text)
        elif "application/pdf" in ctype:
            tmp = Path(".tmp_downloads"); ensure_dir(tmp)
            fname = sanitize_filename(Path(urlparse(url).path).name or "download") + ".pdf"
            fpath = tmp / fname
            with open(fpath, "wb") as f: f.write(r.content)
            return ("pdf", extract_pdf_text(fpath))
        else:
            try:
                return (ctype or "unknown", r.content.decode(r.apparent_encoding or "utf-8", errors="ignore"))
            except Exception: return (ctype or "unknown", "")

def read_local(path: Path, ocr_lang: str, google_credentials_path: Optional[str]) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in TXT_EXTS: return ("text", path.read_text(encoding="utf-8", errors="ignore"))
    elif ext in IMG_EXTS: return ("image", ocr_image(path, lang=ocr_lang, google_credentials_path=google_credentials_path))
    elif ext in PDF_EXTS: return ("pdf", extract_pdf_text(path))
    elif ext in CSV_EXTS: return ("csv", read_csv_text(path))
    else: return ("unknown", "")

# -------- LLMタスク --------
def generate_title(text: str, api_key: str, model: str) -> str:
    try:
        return call_llm_summarize(text=text[:4000], model=model, system_hint="あなたは優れたコピーライターです。", user_task_prompt="以下のテキストの内容を最も的確に表現する、簡潔で魅力的なタイトルを一つだけ生成してください。タイトル以外の余計な言葉は含めないでください。", api_key=api_key).strip().replace('"', '')
    except Exception as e:
        print(f"[WARN] タイトル生成中にエラー: {e}")
        return "(タイトル生成失敗)"

def summarize_one_text(text: str, api_key: str, model: str, max_chars: int) -> str:
    chunks = chunk_text(text, max_chars=max_chars)
    summaries = []
    system_hint = "あなたは優秀な学習アシスタントです。提供されたテキストの要点を、後から見返して内容をすぐに思い出せるように、構造化してまとめてください。"
    user_task_prompt = '''以下のテキストについて、まず全体を3〜5行で要約し、次にその要約を補足する重要なキーワードやポイントを5〜7個、箇条書きで挙げてください。このノートの目的は、内容を完全に網羅することではなく、後から見た人が「ああ、こういう内容だった」と思い出すためのトリガーとなることです。
出力はMarkdown形式でお願いします。例えば、見出し、箇条書き、太字などを使って構造化してください。'''
    for idx, ch in enumerate(chunks, 1):
        if idx > 1: time.sleep(0.2)
        summaries.append(call_llm_summarize(ch, model=model, system_hint=system_hint, user_task_prompt=user_task_prompt, api_key=api_key))
    return "\n".join(summaries)

def extract_technical_terms(text: str, api_key: str, model: str) -> List[str]:
    try:
        response_text = call_llm_summarize(text=text[:8000], model=model, system_hint="あなたはテキストから専門用語を抽出する専門家です。", user_task_prompt='以下のテキストから、主要な専門用語を10個以内で抽出し、純粋なJSON配列（文字列のリスト）として出力してください。例: ["機械学習", "Python"]', api_key=api_key)
        match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if match: response_text = match.group(1)
        terms = json.loads(response_text)
        return [str(term) for term in terms] if isinstance(terms, list) else []
    except Exception as e:
        print(f"[WARN] 専門用語の抽出中にエラー: {e}")
        return []

def suggest_category(text: str, api_key: str, model: str) -> str:
    try:
        return call_llm_summarize(text=text[:4000], model=model, system_hint="あなたはコンテンツ分類の専門家です。",         user_task_prompt="以下のテキストに最もふさわしい、**より大まかで一般的な**カテゴリ名を、日本語で、単語または短いフレーズで一つだけ提案してください。例: 'テクノロジー', 'ビジネス', '教育', '健康', 'エンターテイメント'", api_key=api_key).strip().replace('"', '')
    except Exception as e:
        print(f"[WARN] カテゴリ推薦中にエラー: {e}")
        return ""

# -------- 並列処理オーケストレーション --------
async def _process_one_source_fully(args: tuple) -> dict:
    i, src, api_key, model, max_chars, ocr_lang, tesseract_cmd, google_credentials_path = args
    source_id = f"source-{i+1}"
    print(f"[STATUS]   - ソース {i+1} ({src[:50]}...) の処理を開始")
    try:
        if is_url(src):
            _, text = await fetch_url(src, ocr_lang=ocr_lang, google_credentials_path=google_credentials_path)
            source_name = src
        else:
            p = Path(src)
            if not p.exists(): raise FileNotFoundError(f"Not found: {src}")
            # read_local is synchronous, run it in a thread pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            _, text = await loop.run_in_executor(None, read_local, p, ocr_lang, google_credentials_path)
            source_name = p.name
        if not text or not text.strip():
            return {"id": source_id, "name": source_name, "error": "テキスト抽出不可"}

        # Use a ThreadPoolExecutor for CPU-bound tasks (LLM calls, OCR)
        # This ensures they don't block the asyncio event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as inner_executor:
            loop = asyncio.get_running_loop()
            future_title = loop.run_in_executor(inner_executor, generate_title, text, api_key, model)
            future_summary = loop.run_in_executor(inner_executor, summarize_one_text, text, api_key, model, max_chars)
            future_terms = loop.run_in_executor(inner_executor, extract_technical_terms, text, api_key, model)
            future_category = loop.run_in_executor(inner_executor, suggest_category, text, api_key, model)

            title, summary, terms, category = await asyncio.gather(
                future_title, future_summary, future_terms, future_category
            )
            result = {
                "id": source_id, "name": source_name,
                "title": title, "summary": summary,
                "terms": terms, "category": category,
            }
        print(f"[STATUS]   - ソース {i+1} ({src[:50]}...) の処理を完了")
        return result
    except Exception as e:
        print(f"[ERROR] ソース処理中に致命的なエラー: {src} ({e})")
        return {"id": source_id, "name": src, "error": str(e)}

async def summarize_multiple_inputs(
    srcs: List[str], api_key: str, model: str = "gpt-4o-mini",
    highlight: bool = False, max_chars: int = 3500,
    tesseract_cmd: Optional[str] = None, ocr_lang: str = "jpn+eng",
    google_credentials_path: Optional[str] = None, **kwargs
) -> tuple[list[str], str, List[Dict[str, Any]]]: # Changed return type
    print(f"[STATUS] 全ソースの並列処理を開始します... (全{len(srcs)}件)")
    if tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    tasks = [(i, src, api_key, model, max_chars, ocr_lang, tesseract_cmd, google_credentials_path) for i, src in enumerate(srcs)]
    # Use asyncio.gather to run _process_one_source_fully concurrently
    processed_futures = [
        _process_one_source_fully(task_args) for task_args in tasks
    ]
    processed_results = await asyncio.gather(*processed_futures)
    source_results = [res for res in processed_results if res and not res.get("error")]
    print(f"[STATUS] 全ソースの並列処理が完了。({len(source_results)}件のソース処理に成功)")

    if not source_results:
        return [], "", [] # Changed return value

    all_terms = {term for res in source_results for term in res.get("terms", [])}
    all_categories = [res.get("category") for res in source_results if res.get("category")]
    final_category = max(set(all_categories), key=all_categories.count) if all_categories else ""

    print("[STATUS] 全ての処理が完了しました。")
    return sorted(list(all_terms)), final_category, source_results # Changed return value

def generate_summary_markdown(source_results: List[Dict[str, Any]]) -> str:
    """
    Generates a Markdown report from the summarized source results.
    """
    md_parts = ["# 今日のまとめ\n\n"]

    # Create a table of contents
    for res in source_results:
        md_parts.append(f'- [{res["title"]}](#{res["id"]})\n')
    md_parts.append("\n---\n\n")

    # Add content for each source
    for res in source_results:
        md_parts.append(f'<a id="{res["id"]}"></a>\n')
        md_parts.append(f'## {res["title"]}\n\n')
        md_parts.append(f'**ソース:** [{res["name"]}]({res["name"]})\n\n')
        
        md_parts.append("### 要約\n\n")
        md_parts.append(res.get("summary", "要約はありません。") + "\n\n")

        # Add terms if they exist
        terms = res.get("terms", [])
        if terms:
            md_parts.append("### 関連キーワード\n\n")
            for term in terms:
                md_parts.append(f"- {term}\n")
            md_parts.append("\n")
        
        md_parts.append("---\n\n")

    return "".join(md_parts)


# -------- CLIラッパー --------
async def main():
    parser = argparse.ArgumentParser(
        description="Webアプリ版と同等の高度な並列処理をコマンドラインから実行します。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help="処理対象のURLまたはローカルファイルパス（複数可）")

    # LLM
    parser.add_argument("--model", default="gpt-4o-mini", help="LLMモデル名（例：gpt-4o-mini）")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key. 環境変数 OPENAI_API_KEY も利用可")
    
    # I/O
    parser.add_argument("--output-file", type=Path, default=Path("cli_summary_output.md"), help="出力Markdownファイル名")
    parser.add_argument("--json-output-file", type=Path, default=None, help="カテゴリと専門用語を蓄積するJSONファイル名")
    parser.add_argument("--max-chars", type=int, default=3500, help="1チャンクの最大文字数")
    parser.add_argument("--tesseract-cmd", default=None, help="Tesseractの実行パス（Windows等で必要なら指定）")
    parser.add_argument("--ocr-lang", default="jpn+eng", help="OCR言語（例: 'jpn', 'eng'）")
    parser.add_argument("--google-credentials", default=None, help="Google Cloud Vision APIのサービスアカウントキーJSONファイルへのパス")

    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] APIキーが必須です。--api-key引数またはOPENAI_API_KEY環境変数を設定してください。", file=sys.stderr)
        sys.exit(1)

    try:
        final_terms, final_category, source_results = await summarize_multiple_inputs(
            srcs=args.inputs,
            api_key=args.api_key,
            model=args.model,
            max_chars=args.max_chars,
            tesseract_cmd=args.tesseract_cmd,
            ocr_lang=args.ocr_lang,
            google_credentials_path=args.google_credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            highlight=False
        )

        # 結果をターミナルに表示
        print("\n" + "="*20 + " 処理結果 " + "="*20)
        print(f"■ 検出されたカテゴリ:")
        for res in source_results:
            print(f"  - {res['name'][:50]}...: {res.get('category', 'N/A')}") # ソース名とカテゴリを表示

        print(f"■ 抽出された専門用語 ({len(final_terms)}個):")
        # ターミナルに見やすく表示するため、一定数で改行
        term_lines = []
        line = "  "
        for term in final_terms:
            if len(line) + len(term) + 2 > 80:
                term_lines.append(line)
                line = "  "
            line += term + ", "
        term_lines.append(line.rstrip(", "))
        print("\n".join(term_lines))
        print("\n" + "="*52)

        # Markdownファイルに保存
        print(f"[STATUS] 最終Markdownを生成中...")
        final_md = generate_summary_markdown(source_results)
        args.output_file.write_text(final_md, encoding="utf-8")
        print(f"\n✅ 詳細な要約Markdownを {args.output_file} に保存しました。")

        # JSON出力ファイルが指定されている場合
        if args.json_output_file:
            json_data = []
            if args.json_output_file.exists():
                try:
                    with open(args.json_output_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[WARN] 既存のJSONファイル '{args.json_output_file}' の読み込みに失敗しました。新しいファイルとして扱います。", file=sys.stderr)
                except Exception as e:
                    print(f"[WARN] 既存のJSONファイル '{args.json_output_file}' の読み込み中にエラーが発生しました: {e}", file=sys.stderr)

            updated_count = 0
            for res in source_results:
                # 重複チェック (URL/パスで判断)
                is_duplicate = False
                for existing_item in json_data:
                    if existing_item.get("source_name") == res["name"]:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    json_data.append({
                        "source_name": res["name"],
                        "title": res["title"],
                        "category": res.get("category", "N/A"),
                        "terms": res.get("terms", []),
                        "summary_markdown_file": str(args.output_file) # 生成されたMarkdownファイルへのリンク
                    })
                    updated_count += 1
            
            if updated_count > 0:
                with open(args.json_output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"✅ {updated_count}件の新しいカテゴリ/専門用語データを {args.json_output_file} に保存しました。")
            else:
                print(f"ℹ️ 新しいカテゴリ/専門用語データはありませんでした。")

    except Exception as e:
        print(f"[FATAL] 処理中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
