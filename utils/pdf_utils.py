"""Markdown/HTML→PDF 変換ユーティリティ（WeasyPrint使用）。"""

from __future__ import annotations

from typing import Optional

import markdown as mdlib

from errors import internal


# 既定の簡易CSS（文字小さめ・余白控えめ）
DEFAULT_MD_CSS = """
  @page { size: A4; margin: 12mm; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans JP', Helvetica, Arial, sans-serif;
    font-size: 12px; /* was browser default (~16px) */
    line-height: 1.5; /* was 1.6 */
    color: #111;
  }
  p, ul, ol { margin: 0.6em 0; }
  h1, h2, h3 { margin: 0.8em 0 0.4em; }
  pre { background: #f5f5f5; padding: 8px; overflow: auto; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 4px 6px; }
  blockquote { color: #555; border-left: 4px solid #ddd; margin: 0.6em 0; padding: 0.4em 0.8em; }
"""


def markdown_to_html(md_text: str, *, css: Optional[str] = DEFAULT_MD_CSS, title: Optional[str] = None) -> str:
    """MarkdownをHTMLドキュメント文字列へ変換。"""
    html_content = mdlib.markdown(md_text, output_format="html5")
    doc_title = (title or "Document").replace("<", "&lt;").replace(">", "&gt;")
    css_block = f"<style>\n{css}\n</style>" if css else ""
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset=\"utf-8\" />
        <title>{doc_title}</title>
        {css_block}
      </head>
      <body>
        {html_content}
      </body>
    </html>
    """


def html_to_pdf_bytes(html_doc: str) -> bytes:
  """HTML文字列をPDFバイト列に変換（WeasyPrint）。"""
  try:
    from weasyprint import HTML  # lazy import; can fail on missing system libs
  except Exception as e:  # noqa: BLE001
    internal(
      "WeasyPrintの読み込みに失敗しました（Pango/Cairo/GDK-PixBuf/GLibなどの依存が必要）。詳細は epson.md/WeasyPrintのドキュメント参照。Error: "
      + str(e)
    )
  return HTML(string=html_doc).write_pdf()


def render_markdown_to_pdf(md_text: str, *, css: Optional[str] = DEFAULT_MD_CSS, title: Optional[str] = None) -> bytes:
  """Markdown→HTML→PDF を一括変換。"""
  html_doc = markdown_to_html(md_text, css=css, title=title)
  return html_to_pdf_bytes(html_doc)


def _derive_output_path(input_arg: str) -> str:
  """入力から出力PDFパスを推測。"""
  import os

  base = os.path.basename(input_arg)
  if base.lower().endswith(".md"):
    stem = base[:-3]
    return os.path.join(os.getcwd(), f"{stem}.pdf")
  return os.path.join(os.getcwd(), "out.pdf")


def main(argv: Optional[list[str]] = None) -> int:
  """CLI: Markdown（ファイル/標準入力）→PDF。"""
  import argparse
  import os
  import sys

  p = argparse.ArgumentParser(description="MarkdownをPDF化（WeasyPrint）")
  p.add_argument("input", help="Markdown file path or '-' to read from stdin")
  p.add_argument("-o", "--out", dest="out", help="Output PDF path (default: derive from input)")
  p.add_argument("--title", default=None, help="Document title")
  p.add_argument("--html", dest="html_out", default=None, help="Also write intermediate HTML to this path")
  args = p.parse_args(argv)

  # Load markdown text
  if args.input == "-":
    md_text = sys.stdin.read()
    default_out = os.path.join(os.getcwd(), "out.pdf")
  elif os.path.exists(args.input) and os.path.isfile(args.input):
    with open(args.input, "r", encoding="utf-8") as f:
      md_text = f.read()
    default_out = _derive_output_path(args.input)
  else:
    # Treat as literal markdown text
    md_text = args.input
    default_out = _derive_output_path("input.md")

  # Build HTML (optionally write it) and PDF
  html_doc = markdown_to_html(md_text, title=args.title)
  if args.html_out:
    with open(args.html_out, "w", encoding="utf-8") as f:
      f.write(html_doc)

  pdf_bytes = html_to_pdf_bytes(html_doc)
  out_path = args.out or default_out
  with open(out_path, "wb") as f:
    f.write(pdf_bytes)
  print(out_path)
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
