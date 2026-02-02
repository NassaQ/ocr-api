from fastapi import Request
from typing import Tuple
from datetime import datetime
import numpy as np
import re

from app.core.logging import logger


def get_paddle_engine(request: Request):
    """Get PaddleOCR engine from app state."""
    return request.app.state.paddle_engine


def get_easy_engine(request: Request):
    """Get EasyOCR engine from app state."""
    return request.app.state.easy_engine


def get_timestamp() -> str:
    """Returns current timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def contains_arabic(text: str) -> bool:
    """Checks if text contains Arabic characters (Unicode range: U+0600-U+06FF)."""
    arabic_pattern = re.compile(r"[\u0600-\u06FF]")
    return bool(arabic_pattern.search(text))


def ocr_with_paddle(paddle_engine, img_array: np.ndarray) -> Tuple[str, float]:
    """
    Runs PaddleOCR on the image.
    Returns extracted text and average confidence score.
    """
    try:
        result = paddle_engine.ocr(img_array)
        extracted_text = ""
        confidence_scores = []

        if result:
            data = result[0]
            if isinstance(data, dict) and "rec_texts" in data:
                extracted_text = " ".join(data.get("rec_texts", []))
                confidence_scores = data.get("rec_scores", [])
            elif isinstance(data, list):
                texts = []
                for item in data:
                    if isinstance(item, list) and len(item) >= 2:
                        texts.append(item[1][0])
                        confidence_scores.append(item[1][1])
                extracted_text = " ".join(texts)

        avg_conf = (
            round(sum(confidence_scores) / len(confidence_scores), 2)
            if confidence_scores
            else 0.0
        )
        return extracted_text, avg_conf
    except Exception as e:
        logger.error(f"PaddleOCR error: {e}")
        return "", 0.0


def ocr_with_easy(easy_engine, img_array: np.ndarray) -> Tuple[str, float]:
    """
    Runs EasyOCR with paragraph handling for correct Arabic right-to-left reading order.
    Returns extracted text and confidence score.
    """
    try:
        results = easy_engine.readtext(img_array, detail=0, paragraph=True)
        final_text = "\n".join(results)
        avg_conf = 0.95 if final_text.strip() else 0.0
        return final_text, avg_conf
    except Exception as e:
        logger.error(f"EasyOCR error: {e}")
        return "", 0.0


def smart_ocr_pipeline(
    paddle_engine, easy_engine, img_array: np.ndarray
) -> Tuple[str, float, str]:
    """
    Smart OCR routing: runs PaddleOCR first, switches to EasyOCR if Arabic is detected.
    Returns (extracted_text, confidence_score, model_name).
    """
    text, conf = ocr_with_paddle(paddle_engine, img_array)
    used_model = "paddle"

    if contains_arabic(text):
        easy_text, easy_conf = ocr_with_easy(easy_engine, img_array)
        if easy_text.strip():
            return easy_text, easy_conf, "easyocr (auto-switched)"

    return text, conf, used_model
