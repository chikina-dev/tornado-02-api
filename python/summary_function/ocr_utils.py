import os
import io
from pathlib import Path
from typing import Optional

from PIL import Image
import pytesseract
try:
    from google.cloud import vision
except ImportError:
    vision = None

def ocr_image(path: Path, lang: str = "jpn+eng") -> str:
    """
    画像からテキストを抽出します。
    GOOGLE_APPLICATION_CREDENTIALS 環境変数があれば Google Cloud Vision を、
    なければ Tesseract OCR を使用します。
    """
    # Google Cloud Vision API を優先
    if vision and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            client = vision.ImageAnnotatorClient()
            with io.open(path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            if response.error.message:
                raise Exception(response.error.message)
            
            texts = response.text_annotations
            return texts[0].description if texts else ""
        except Exception as e:
            print(f"[WARN] Google Cloud Vision API OCR失敗: {path} ({e})")
            return ""
    
    # Tesseract OCR をフォールバックとして使用
    try:
        # PSMとOEMを設定して精度向上を図る
        config = "--psm 6 --oem 3"
        with Image.open(path) as im:
            return pytesseract.image_to_string(im, lang=lang, config=config)
    except Exception as e:
        print(f"[WARN] Tesseract OCR失敗: {path} ({e})")
        return ""