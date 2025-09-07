"""ファイル操作ユーティリティ（最小限）。"""

from __future__ import annotations

import base64
import os
from typing import Optional


def guess_content_type(filename: str) -> Optional[str]:
    """拡張子から簡易に Content-Type を推定。"""
    if "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in {"txt", "log", "md"}:
        return "text/plain"
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "pdf":
        return "application/pdf"
    return None


def display_filename(file_path: str) -> str:
    """保存名(タイムスタンプ_元名)から元のファイル名を取り出す。"""
    base = os.path.basename(file_path)
    return base.split("_", 1)[1] if "_" in base else base


def read_file_as_base64(file_path: str) -> str:
    """ファイル内容を Base64 文字列にして返す。"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
