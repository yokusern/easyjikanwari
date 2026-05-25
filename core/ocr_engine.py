"""
OCR engine using EasyOCR.

Strategy:
  1. Run EasyOCR once on the full (upscaled) image
  2. For each detected text region, find which color-group block contains it
  3. Pre-fill group names with the matched text
  4. User corrects anything wrong in the UI

EasyOCR advantages over pytesseract for this use case:
  - Handles rotated / vertical text (common in narrow yearly calendar columns)
  - Better Japanese + English mixed accuracy
  - No system packages required (pure pip install)
"""

import cv2
import numpy as np
from collections import Counter


def load_reader():
    """
    Load EasyOCR reader (Japanese + English).
    Call this inside st.cache_resource so the model loads only once.
    """
    import easyocr  # imported here to avoid loading at module level
    return easyocr.Reader(["ja", "en"], gpu=False, verbose=False)


def run_ocr(img_bgr: np.ndarray, reader, upscale: float = 2.5) -> list[dict]:
    """
    Run OCR on the full image and return a list of detected text regions.

    Each item: {text, confidence, cx, cy}
    cx/cy are the center coordinates in the ORIGINAL (pre-upscale) image.
    """
    h, w = img_bgr.shape[:2]

    # Upscale for better accuracy on small text
    up = cv2.resize(img_bgr,
                    (int(w * upscale), int(h * upscale)),
                    interpolation=cv2.INTER_CUBIC)

    # CLAHE on L channel to improve contrast
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    up = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    results = reader.readtext(up, detail=1, paragraph=False)

    items = []
    for (bbox, text, conf) in results:
        if conf < 0.25 or not text.strip():
            continue
        # bbox: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] in upscaled coords
        cx_up = sum(pt[0] for pt in bbox) / 4
        cy_up = sum(pt[1] for pt in bbox) / 4
        # Map back to original coords
        cx = cx_up / upscale
        cy = cy_up / upscale
        items.append({
            "text": text.strip(),
            "confidence": conf,
            "cx": cx,
            "cy": cy,
        })

    return items


def assign_ocr_to_groups(ocr_items: list[dict], groups: list[dict]) -> list[dict]:
    """
    For each color group, collect all OCR text whose center falls inside
    one of the group's blocks. The most frequent text becomes the suggested name.
    """
    for group in groups:
        found_texts: list[str] = []

        for item in ocr_items:
            cx, cy = item["cx"], item["cy"]
            for block in group["blocks"]:
                bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    found_texts.append(item["text"])
                    break  # count each OCR item once per group

        if found_texts:
            # Pick the most common text across all blocks in this group
            most_common = Counter(found_texts).most_common(1)[0][0]
            group["name_ocr"] = most_common
        else:
            group["name_ocr"] = ""

    return groups
