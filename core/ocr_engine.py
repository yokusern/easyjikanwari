"""
OCR engine with multi-pass preprocessing.
Handles low-quality / LINE-compressed images by aggressively upscaling
and applying CLAHE + sharpening before passing to Tesseract.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image


# Tesseract configs (Japanese + English, different page-segment modes)
_CFG_BLOCK = "--oem 3 --psm 6 -l jpn+eng"   # uniform block (multi-line cell)
_CFG_LINE = "--oem 3 --psm 7 -l jpn+eng"    # single text line (header cell)
_CFG_SPARSE = "--oem 3 --psm 11 -l jpn+eng"  # sparse text (small noisy cells)


def is_tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _preprocess(cell_bgr: np.ndarray, target_min_dim: int = 80) -> np.ndarray:
    """
    Prepare a single-cell BGR image for OCR:
      1. Upscale to at least target_min_dim on each side
      2. CLAHE on L channel (handles uneven lighting)
      3. Unsharp mask
      4. Otsu threshold → clean binary image
      5. White border padding (avoids edge-clip in Tesseract)
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return None

    h, w = cell_bgr.shape[:2]
    if h < 1 or w < 1:
        return None

    # 1. Upscale
    scale = max(target_min_dim / h, target_min_dim / w, 2.5)
    cell_bgr = cv2.resize(cell_bgr,
                          (max(int(w * scale), 1), max(int(h * scale), 1)),
                          interpolation=cv2.INTER_CUBIC)

    # 2. CLAHE
    lab = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    cell_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3. Grayscale + unsharp mask
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # 4. Otsu threshold
    _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Padding
    binary = cv2.copyMakeBorder(binary, 15, 15, 15, 15,
                                 cv2.BORDER_CONSTANT, value=255)
    return binary


def ocr_cell(cell_bgr: np.ndarray, mode: str = "block") -> str:
    """
    OCR a single cell image.
    mode: "block" (default) | "line" | "sparse"
    Returns cleaned text string.
    """
    processed = _preprocess(cell_bgr)
    if processed is None:
        return ""

    pil_img = Image.fromarray(processed)
    cfg = {"block": _CFG_BLOCK, "line": _CFG_LINE, "sparse": _CFG_SPARSE}.get(
        mode, _CFG_BLOCK
    )

    try:
        text = pytesseract.image_to_string(pil_img, config=cfg)
        return _clean(text)
    except Exception:
        return ""


def ocr_cell_best(cell_bgr: np.ndarray) -> str:
    """
    Run OCR with multiple configs and return the result with the most content.
    Use for cells where mode is uncertain (e.g., color blocks in nursing timetables).
    """
    results = []
    for mode in ("block", "sparse"):
        t = ocr_cell(cell_bgr, mode)
        if t:
            results.append(t)
    if not results:
        return ""
    # Prefer the result with more meaningful characters
    return max(results, key=lambda s: len(s.replace(" ", "").replace("\n", "")))


def ocr_large_region(img_bgr: np.ndarray) -> str:
    """OCR a large image region (e.g., a colored block spanning multiple weeks)."""
    processed = _preprocess(img_bgr, target_min_dim=120)
    if processed is None:
        return ""
    pil_img = Image.fromarray(processed)
    try:
        text = pytesseract.image_to_string(pil_img, config=_CFG_BLOCK)
        return _clean(text)
    except Exception:
        return ""


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    import re
    text = text.strip()
    text = re.sub(r"\n{2,}", "\n", text)   # collapse blank lines
    text = re.sub(r" {2,}", " ", text)      # collapse spaces
    # Remove common OCR junk characters
    text = re.sub(r"[|｜┃│\[\]{}\\]", "", text)
    text = text.strip()
    return text
