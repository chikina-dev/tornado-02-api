import os
import io
from pathlib import Path
from typing import Optional

from PIL import Image
import pytesseract
from google.cloud import vision

def ocr_image(path: Path, tesseract_cmd: Optional[str] = None, lang: str = "jpn+eng", google_credentials_path: Optional[str] = None) -> str:
    if google_credentials_path:
        # Google Cloud Vision APIを使用
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
            client = vision.ImageAnnotatorClient()

            with io.open(path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            texts = response.text_annotations

            if texts:
                # 最初のテキストアノテーションが全体のテキスト
                return texts[0].description
            else:
                return ""
        except Exception as e:
            print(f"[WARN] Google Cloud Vision API OCR失敗: {path} ({e})")
            return ""
    else:
        # Tesseract OCRを使用 (フォールバックまたはデフォルト)
        if tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        try:
            # 手書き文字認識の精度向上のため、PSMとOEMを設定
            # 最適な設定は画像によって異なるため、必要に応じて調整してください。
            # 例: --psm 6 (単一の均一なテキストブロック) --oem 3 (LSTMとレガシーエンジンの両方)
            config = "--psm 6 --oem 3"
            with Image.open(path) as im:
                return pytesseract.image_to_string(im, lang=lang, config=config)
        except Exception as e:
            print(f"[WARN] Tesseract OCR失敗: {path} ({e})")
            return ""
