"""
Image processing for timetable detection.

Grid mode  : standard weekly timetables (uniform cells)
Color mode : nursing/education yearly calendars (colored multi-span blocks)
"""

import cv2
import numpy as np
from PIL import Image


# ── Shared ────────────────────────────────────────────────────────────────────

def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def enhance_for_ocr(img_bgr: np.ndarray, scale: float = 3.0) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, k)


# ── Color-block detection ─────────────────────────────────────────────────────

def detect_color_blocks(img_bgr: np.ndarray,
                         min_saturation: int = 30,
                         min_area_ratio: float = 0.0005,
                         merge_gap: int = 10) -> list[dict]:
    """
    Detect non-white/gray colored regions (class blocks in nursing-style timetables).
    Returns list of {x, y, w, h, area, color_bgr} sorted top-left to bottom-right.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > min_saturation) & (hsv[:, :, 2] > 30)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_gap, merge_gap))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    blocks = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_ratio * img_area:
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < 8 or bh < 8:
            continue
        m = (labels == i).astype(np.uint8) * 255
        color = cv2.mean(img_bgr, mask=m)[:3]
        blocks.append({"x": bx, "y": by, "w": bw, "h": bh,
                        "area": area, "color_bgr": color, "text": ""})

    return sorted(blocks, key=lambda b: (b["y"], b["x"]))


def draw_numbered_blocks(img_bgr: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """Draw numbered orange rectangles on each detected block."""
    out = img_bgr.copy()
    for i, b in enumerate(blocks):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 110, 255), 3)
        label = str(i + 1)
        fs = max(min(h / 35, 1.4), 0.45)
        fw = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, fs, fw)
        cv2.rectangle(out, (x, y), (x + tw + 8, y + th + 10), (0, 110, 255), -1)
        cv2.putText(out, label, (x + 4, y + th + 5), font, fs, (255, 255, 255), fw)
    return out


def auto_detect_mode(img_bgr: np.ndarray) -> str:
    blocks = detect_color_blocks(img_bgr)
    h, w = img_bgr.shape[:2]
    ratio = sum(b["area"] for b in blocks) / (h * w)
    return "color" if ratio > 0.08 else "grid"


# ── Grid-cell detection ───────────────────────────────────────────────────────

def detect_grid_cells(img_bgr: np.ndarray) -> list[dict]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 10, 20), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_k, iterations=2)

    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 20)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_k, iterations=2)

    grid = cv2.add(h_lines, v_lines)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    cells_mask = cv2.erode(cv2.bitwise_not(grid),
                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=2)
    _, cells_mask = cv2.threshold(cells_mask, 128, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(cells_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    raw = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if not (0.0003 * img_area < area < 0.40 * img_area):
            continue
        if not (0.04 < cw / max(ch, 1) < 25):
            continue
        raw.append({"x": cx, "y": cy, "w": cw, "h": ch})
    return _nms(raw)


def sort_to_grid(cells: list[dict]) -> list[list[dict]]:
    if not cells:
        return []
    cells_s = sorted(cells, key=lambda c: (c["y"], c["x"]))
    heights = sorted(c["h"] for c in cells_s)
    tol = max(heights[len(heights) // 4] * 0.4, 6)
    rows, cur = [], [cells_s[0]]
    for cell in cells_s[1:]:
        if abs(cell["y"] - cur[-1]["y"]) <= tol:
            cur.append(cell)
        else:
            rows.append(sorted(cur, key=lambda c: c["x"]))
            cur = [cell]
    rows.append(sorted(cur, key=lambda c: c["x"]))
    return rows


def extract_cell_img(img_bgr: np.ndarray, cell: dict, pad: int = 4):
    x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
    ih, iw = img_bgr.shape[:2]
    x1, y1 = max(x + pad, 0), max(y + pad, 0)
    x2, y2 = min(x + w - pad, iw), min(y + h - pad, ih)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


# ── Internal ──────────────────────────────────────────────────────────────────

def _nms(cells: list[dict], threshold: float = 0.5) -> list[dict]:
    cells = sorted(cells, key=lambda c: c["w"] * c["h"], reverse=True)
    kept = []
    for cell in cells:
        x1, y1, x2, y2 = cell["x"], cell["y"], cell["x"] + cell["w"], cell["y"] + cell["h"]
        ok = True
        for k in kept:
            kx2, ky2 = k["x"] + k["w"], k["y"] + k["h"]
            ix1, iy1 = max(x1, k["x"]), max(y1, k["y"])
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller = min(cell["w"] * cell["h"], k["w"] * k["h"])
                if smaller > 0 and inter / smaller > threshold:
                    ok = False
                    break
        if ok:
            kept.append(cell)
    return kept
