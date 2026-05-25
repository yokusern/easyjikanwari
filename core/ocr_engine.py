"""
OCR stub — EasyOCR removed (too heavy for Streamlit Community Cloud free tier).
Group names are entered manually by the user.
"""

import numpy as np


def run_ocr(img_bgr: np.ndarray, *args, **kwargs) -> list[dict]:
    return []


def assign_ocr_to_groups(ocr_items: list[dict], groups: list[dict]) -> list[dict]:
    for group in groups:
        group["name_ocr"] = ""
    return groups
