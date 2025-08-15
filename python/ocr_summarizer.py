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
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from pypdf import PdfReader

# ========== LLM Provider (OpenAI) ==========
# pip install "openai>=1.40"
from openai import OpenAI

# .env 読み込み（.env を使わない場合は不要）
# pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()  # 存在すれば自動読み込み
except Exception:
    pass


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
    # URLや任意文字列から安全なファイル名を作成
    safe = re.sub(r"[^\\w\\-.]+", "_", name.strip())
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
        # 行区切りに寄せて切る（最後から最寄りの改行へ）
        cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start + int(max_chars * 0.5):
            cut = end
        chunks.append(text[start:cut])
        start = cut
    return chunks


def build_requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "SummarizerBot/1.0"})
    return session


HTTP = build_requests_session()


# -------- OCR / PDF / CSV --------
def ocr_image(path: Path, tesseract_cmd: Optional[str] = None, lang: str = "jpn+eng") -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        with Image.open(path) as im:
            return pytesseract.image_to_string(im, lang=lang)
    except Exception as e:
        print(f"[WARN] OCR失敗: {path} ({e})")
        return ""

def extract_pdf_text(path: Path) -> str:
    """
    埋め込みテキストがあるPDFからの抽出に対応。
    スキャンPDFはここではOCRしません（必要ならpdf2image+OCRで拡張）。
    """
    try:
        reader = PdfReader(str(path))
        texts = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                texts.append(t)
            else:
                # 空ならページ番号だけでもログ
                print(f"[INFO] PDF page {i+1}: テキストなし（スキャンの可能性）")
        return "\n\n".join(texts)
    except Exception as e:
        print(f"[WARN] PDF抽出失敗: {path} ({e})")
        return ""

def read_csv_text(path: Path) -> str:
    """
    CSVファイルを読み込み、全てのセルを結合してテキストとして返す。
    """
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


# -------- 入力取得（URL／ファイル） --------
def fetch_url(url: str, ocr_lang: str) -> tuple[str, str]:
    """
    URLを取得して (content_type, text) を返す。
    画像はバイナリを一時保存しOCR、HTMLは本文抽出、プレーンはそのまま。
    """
    r = HTTP.get(url, timeout=30)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").lower()

    if "image" in ctype:
        # 一時ファイルに保存して OCR
        ext = "." + ctype.split("/")[-1].split(";")[0]
        fname = sanitize_filename(Path(urlparse(url).path).name or "download") + ext
        tmp = Path(".tmp_downloads")
        ensure_dir(tmp)
        fpath = tmp / fname
        with open(fpath, "wb") as f:
            f.write(r.content)
        text = ocr_image(fpath, lang=ocr_lang)
        return ("image", text)

    elif "text/html" in ctype or "application/xhtml+xml" in ctype:
        soup = BeautifulSoup(r.text, "lxml")
        # script / style 除去
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        main_text = soup.get_text("\n")
        # 余分な空白処理
        cleaned = re.sub(r"\n{3,}", "\n\n", main_text).strip()
        return ("html", cleaned)

    elif "text/plain" in ctype:
        return ("text", r.text)

    elif "application/pdf" in ctype:
        # PDF テキスト抽出を試みる
        tmp = Path(".tmp_downloads")
        ensure_dir(tmp)
        fname = sanitize_filename(Path(urlparse(url).path).name or "download") + ".pdf"
        fpath = tmp / fname
        with open(fpath, "wb") as f:
            f.write(r.content)
        text = extract_pdf_text(fpath)
        return ("pdf", text)

    else:
        # 未対応CTはそのまま文字化け覚悟でdecodeを試す
        try:
            enc = r.apparent_encoding if hasattr(r, "apparent_encoding") else None
            return (ctype or "unknown", r.content.decode(enc or "utf-8", errors="ignore"))
        except Exception:
            return (ctype or "unknown", "")


def read_local(path: Path, ocr_lang: str) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext in TXT_EXTS:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return ("text", txt)
    elif ext in IMG_EXTS:
        return ("image", ocr_image(path, lang=ocr_lang))
    elif ext in PDF_EXTS:
        return ("pdf", extract_pdf_text(path))
    elif ext in CSV_EXTS:
        return ("csv", read_csv_text(path))
    else:
        return ("unknown", "")


# -------- LLM呼び出しヘルパ --------
def with_retries(fn, *, retries=3, base=0.8):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            sleep = base * (2 ** i) + random.uniform(0, 0.2)
            time.sleep(sleep)
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

    user_prompt = f"{user_task_prompt}\n\n---\n{text}\n---"

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


def extract_technical_terms(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """LLM を使ってテキストから専門用語を抽出する"""
    system_hint = (
        "あなたはテキストから専門用語を抽出する専門家です。"
        "一般的な単語は無視し、特定の分野に特有の重要なキーワードや専門用語のみを特定してください。"
    )
    user_task_prompt = (
        "以下のテキストから、主要な専門用語を10個以内で抽出してください。"
        "結果は、説明や他のテキストを含めず、純粋なJSON配列（文字列のリスト）として出力してください。"
        "例: [\"機械学習\", \"ディープラーニング\", \"Python\"]"
    )
    
    try:
        # 1回のAPIコールで済ませるため、チャンク分割はしない
        # 長すぎる場合は切り詰める
        truncated_text = text[:8000]
        
        response_text = call_llm_summarize(
            text=truncated_text,
            model=model,
            system_hint=system_hint,
            user_task_prompt=user_task_prompt,
            api_key=api_key
        )
        
        # LLMからの出力がマークダウンのコードブロックを含む場合を考慮
        match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if match:
            response_text = match.group(1)

        # JSON文字列をパースしてリストを返す
        terms = json.loads(response_text)
        if isinstance(terms, list):
            return [str(term) for term in terms]
        return []
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[WARN] 専門用語の抽出またはパースに失敗しました: {e}")
        # 失敗した場合は、単純な単語分割で代替することも可能だが、今回は空リストを返す
        return []
    except Exception as e:
        print(f"[WARN] 専門用語の抽出中に予期せぬエラーが発生しました: {e}")
        return []

def suggest_category(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> str:
    """LLM を使ってテキストからカテゴリを推薦させる"""
    system_hint = (
        "あなたはコンテンツ分類の専門家です。"
        "提示されたテキストの内容を分析し、最も的確な単一のカテゴリ名を提案してください。"
    )
    user_task_prompt = (
        "以下のテキストに最もふさわしいカテゴリ名を、日本語で、単語または短いフレーズで一つだけ提案してください。"
        "例: 'プログラミング', '機械学習', '料理レシピ', '国際ニュース'。"
        "説明や他の余計なテキストは一切含めず、カテゴリ名のみを出力してください。"
    )
    try:
        # テキストが長すぎる場合、主要部分を切り出す
        truncated_text = text[:4000]
        category = call_llm_summarize(
            text=truncated_text,
            model=model,
            system_hint=system_hint,
            user_task_prompt=user_task_prompt,
            api_key=api_key
        )
        # 不要な句読点や空白を除去して返す
        return category.strip().replace('"', '').replace("'", '')
    except Exception as e:
        print(f"[WARN] カテゴリ推薦中にエラーが発生しました: {e}")
        return ""


def format_summary_with_llm(summary_text: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """要約結果をHTMLで見やすく整形する"""
    system_hint = (
        "あなたは優秀なコンテンツエディターです。"
        "提供された要約テキストを、最も重要な結論が目立つようにHTML形式で再構成してください。"
        "出力はHTMLタグのみとし、余計な説明は含めないでください。"
    )
    user_task_prompt = (
        "以下の要約テキストについて、次のHTML構造で整形してください:\n"
        "1. 最も重要な結論、または中心的なメッセージを`<h1>`タグで囲んでください。"
        "2. その他の重要なポイントやキーワードを`<strong>`タグで強調してください。"
        "3. 全体は適切な段落`<p>`で構成してください。"
        "4. 必ずHTMLタグのみを出力してください。"
    )
    return call_llm_summarize(
        text=summary_text,
        model=model,
        system_hint=system_hint,
        user_task_prompt=user_task_prompt,
        api_key=api_key
    )


# -------- 出力先決定 --------
def make_summary_output_path(src: str, output_dir: Optional[Path]) -> Path:
    if is_url(src):
        base = sanitize_filename(src)
    else:
        base = Path(src).name
    if "." in base:
        stem = ".".join(base.split(".")[:-1]) or Path(base).stem
    else:
        stem = base
    fname = f"{stem}.summary.txt"
    return (output_dir / fname) if output_dir else Path(src).parent / fname


def summarize_multiple_inputs(
    srcs: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    summary_prompt: Optional[str] = None,
    system_hint: Optional[str] = None,
    synthesis: bool = False,
    highlight: bool = False,
    max_chars: int = 3500,
    tesseract_cmd: Optional[str] = None,
    ocr_lang: str = "jpn+eng"
) -> tuple[str, list[str], str]:
    """
    複数の入力（URL/ファイル）を処理して、(まとめ文字列, 専門用語リスト, 推奨カテゴリ) を返す。
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    source_texts = []
    for i, src in enumerate(srcs):
        # 取得
        if is_url(src):
            ctype, text = fetch_url(src, ocr_lang=ocr_lang)
            source_name = src
        else:
            p = Path(src)
            if not p.exists():
                raise FileNotFoundError(f"Not found: {src}")
            ctype, text = read_local(p, ocr_lang=ocr_lang)
            source_name = p.name

        if text and text.strip():
            # 各ソースに一意のIDを付与
            source_texts.append((f"source-{i+1}", source_name, text))
        else:
            print(f"[INFO] テキストが抽出できませんでした: {src} (type={ctype})")

    if not source_texts:
        return "[INFO] 全てのソースからテキストが抽出できませんでした。", [], ""

    # 各ソースを識別子付きで結合
    combined_text = "\n\n".join(f"[SOURCE_ID: {sid}, NAME: {name}]\n{text}" for sid, name, text in source_texts)

    # 専門用語を抽出
    terms = extract_technical_terms(combined_text, api_key=api_key, model=model)

    # カテゴリを推薦
    suggested_category = suggest_category(combined_text, api_key=api_key, model=model)

    # プロンプトを調整
    if system_hint is None:
        system_hint = (
            "あなたはプロの編集者です。"
            "複数のソースが[SOURCE_ID: ..., NAME: ...]形式で提供されます。"
            "各ソースの内容を個別の章としてまとめ、全体を一つのHTMLドキュメントとして構成してください。"
            "各章の区切りが明確になるようにしてください。"
        )
    if summary_prompt is None:
        summary_prompt = (
            "以下の複数ソースを、指定された形式に従って一つのHTMLとしてまとめてください。\n\n"
            "1. 全体のタイトルとして`<h1>今日のまとめ</h1>`を最初に追加してください。\n"
            "2. 次に、各ソースへのアンカーリンクを含む目次を`<ul>`リストで生成してください。リンクのテキストはソース名(`NAME`)を使用し、リンク先はソースID(`SOURCE_ID`)を`href`属性に使用してください（例: `<a href=\"#source-1\">...</a>`）。\n"
            "3. 各ソースのまとめは、`<h2>`タグで始まる章としてください。この`<h2>`タグには、目次のリンクに対応する`id`属性を必ず含めてください（例: `id=\"source-1\"`）。章のタイトルはソース名(`NAME`)にしてください。\n"
            "4. 各章の終わりには`<hr>`タグを挿入し、話の区切りを明確にしてください。\n"
            "5. 全体として、単一のHTML応答を生成してください。"
        )

    # チャンク分割
    chunks = chunk_text(combined_text, max_chars=max_chars)

    # 要約 (map)
    summaries = []
    for idx, ch in enumerate(chunks, 1):
        if idx > 1:
            time.sleep(0.4)
        summary = call_llm_summarize(
            ch,
            model=model,
            system_hint=system_hint,
            user_task_prompt=summary_prompt,
            api_key=api_key
        )
        summaries.append(summary)

    final_summary = "\n\n---\n\n".join(summaries).strip()

    # 最終統合 (reduce)
    if synthesis and len(summaries) > 1:
        synthesis_input = "\n\n".join(f"[Chunk {i+1}]\n{sm}" for i, sm in enumerate(summaries))
        synthesis_system_hint = (
            "あなたは優秀な編集者です。"
            "複数のまとめチャンクが提供されます。これらを一つの首尾一貫したHTMLドキュメントに統合してください。"
            "タイトル、目次、章立て、区切り線(`<hr>`)の構造は必ず維持してください。"
        )
        final_summary = call_llm_summarize(
            synthesis_input,
            model=model,
            system_hint=synthesis_system_hint,
            user_task_prompt="各チャンクの内容を統合し、重複を排除して、指定されたHTML形式（タイトル、目次、章立て、区切り線）の最終的なまとめを作成してください。",
            api_key=api_key
        )
    
    if highlight and not final_summary.strip().startswith("<h1>"):
        final_summary = format_summary_with_llm(final_summary, api_key=api_key, model=model)

    return final_summary, terms, suggested_category



# -------- メイン処理 --------

def summarize_input(
    src: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    summary_prompt: Optional[str] = None,
    system_hint: Optional[str] = None,
    synthesis: bool = False,
    highlight: bool = False,
    max_chars: int = 3500,
    tesseract_cmd: Optional[str] = None,
    ocr_lang: str = "jpn+eng"
) -> str:
    """
    単一の入力（URL/ファイル）を処理して要約文字列を返す。
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # 取得
    if is_url(src):
        ctype, text = fetch_url(src, ocr_lang=ocr_lang)
    else:
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"Not found: {src}")
        ctype, text = read_local(p, ocr_lang=ocr_lang)

    if not text or not text.strip():
        return f"[INFO] テキストが抽出できませんでした: {src} (type={ctype})"

    # チャンク分割
    chunks = chunk_text(text, max_chars=max_chars)

    # 要約 (map)
    summaries = []
    for idx, ch in enumerate(chunks, 1):
        if idx > 1:
            time.sleep(0.4)  # 軽いレート間隔
        summary = call_llm_summarize(
            ch,
            model=model,
            system_hint=system_hint,
            user_task_prompt=summary_prompt,
            api_key=api_key
        )
        summaries.append(summary)

    final_summary = "\n\n---\n\n".join(summaries).strip()

    # 最終統合 (reduce)
    if synthesis and len(summaries) > 1:
        synthesis_input = "\n\n".join(f"[Chunk {i+1}]\n{sm}" for i, sm in enumerate(summaries))
        final_summary = call_llm_summarize(
            synthesis_input,
            model=model,
            system_hint=system_hint or "You are a professional editor. Synthesize the chunks into one cohesive, non-redundant summary.",
            user_task_prompt=summary_prompt or "Create a single concise summary that merges all chunks without repetition.",
            api_key=api_key
        )
    
    # ハイライト表示
    if highlight:
        final_summary = format_summary_with_llm(final_summary, api_key=api_key, model=model)

    return final_summary


def process_source(src: str, args) -> Optional[Path]:
    """
    単一の入力（URL/ファイル）を処理して要約ファイルを保存。（CLI用ラッパー）
    """
    final_summary = summarize_input(
        src=src,
        api_key=args.api_key,
        model=args.model,
        summary_prompt=args.summary_prompt,
        system_hint=args.system_hint,
        synthesis=args.synthesis,
        max_chars=args.max_chars,
        tesseract_cmd=args.tesseract_cmd,
        ocr_lang=args.ocr_lang
    )
    
    # 保存
    out_path = make_summary_output_path(src, args.output_dir)
    ensure_dir(out_path.parent)
    out_path.write_text(final_summary, encoding="utf-8")
    print(f"[OK] Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize URLs / files (text, image(OCR), PDF) for review via LLM API."
    )
    parser.add_argument("inputs", nargs="+", help="URLまたはローカルファイルパス（複数可）")

    # LLM
    parser.add_argument("--model", default="gpt-4o-mini", help="LLMモデル名（OpenAI例：gpt-4o-mini）")
    parser.add_argument("--api-key", default=None, help="OpenAI API Key（環境変数 OPENAI_API_KEY も可/.envも可）")
    parser.add_argument("--summary-prompt", default=None, help="要約を指示するプロンプト（例：「箇条書きで要約」「重要語を列挙」など）")
    parser.add_argument("--system-hint", default=None, help="要約スタイルのシステムプロンプトを上書き")
    parser.add_argument("--synthesis", action="store_true", help="チャンク要約を最後に統合要約する")

    # I/O
    parser.add_argument("--output-dir", type=Path, default=None, help="出力ディレクトリ（未指定で入力と同じ場所）")
    parser.add_argument("--max-chars", type=int, default=3500, help="1チャンクの最大文字数")
    parser.add_argument("--tesseract-cmd", default=None, help="Tesseract 実行ファイルのパス（Windows等で必要な場合）")
    parser.add_argument("--ocr-lang", default="jpn+eng", help="Tesseract OCRの言語（例: 'jpn', 'eng', 'jpn+eng'）")

    # 先々の拡張のために予約
    parser.add_argument("--provider", default="openai", choices=["openai"], help="LLMプロバイダ")

    args = parser.parse_args()

    results = []
    for src in args.inputs:
        try:
            out = process_source(src, args)
            if out:
                results.append(out)
        except Exception as e:
            print(f"[ERROR] {src}: {e}")

    if not results:
        sys.exit(2)


if __name__ == "__main__":
    main()