"""
Image processing for timetable detection.

Two modes:
- grid_mode: Standard weekly timetables with uniform cells
- color_mode: Nursing/education yearly calendars with colored multi-span blocks
"""

import cv2
import numpy as np
from PIL import Image


# ── Shared utilities ──────────────────────────────────────────────────────────

def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def enhance_for_ocr(img_bgr: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """
    Upscale + CLAHE + sharpen.
    Improves OCR accuracy on low-res / LINE-compressed images.
    """
    h, w = img_bgr.shape[:2]
    upscaled = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, kernel)


def draw_cells_on_image(img_bgr: np.ndarray, cells: list) -> np.ndarray:
    debug = img_bgr.copy()
    palette = [(220, 50, 50), (50, 180, 50), (50, 50, 220),
               (180, 130, 50), (130, 50, 180), (50, 180, 180)]
    for i, cell in enumerate(cells):
        x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
        color = palette[i % len(palette)]
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
    return debug


# ── Grid mode ────────────────────────────────────────────────────────────────

def detect_grid_cells(img_bgr: np.ndarray) -> list[dict]:
    """
    Detect uniform table cells using morphological line detection.
    Suited for standard weekly timetable screenshots.
    Returns list of {x, y, w, h} dicts.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal lines: at least 10% of image width
    h_len = max(w // 10, 20)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Vertical lines: at least 8% of image height
    v_len = max(h // 8, 20)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

    grid = cv2.add(h_lines, v_lines)
    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grid = cv2.dilate(grid, dil_k, iterations=1)

    # Cells are the regions NOT covered by grid lines
    cell_mask = cv2.erode(cv2.bitwise_not(grid),
                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                          iterations=2)
    _, cell_mask = cv2.threshold(cell_mask, 128, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL,
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
    """Sort flat cell list into 2-D grid (list of rows)."""
    if not cells:
        return []
    cells_sorted = sorted(cells, key=lambda c: (c["y"], c["x"]))
    heights = sorted(c["h"] for c in cells_sorted)
    p25 = heights[len(heights) // 4]
    tol = max(p25 * 0.4, 6)

    rows: list[list[dict]] = []
    current: list[dict] = [cells_sorted[0]]
    for cell in cells_sorted[1:]:
        if abs(cell["y"] - current[-1]["y"]) <= tol:
            current.append(cell)
        else:
            rows.append(sorted(current, key=lambda c: c["x"]))
            current = [cell]
    rows.append(sorted(current, key=lambda c: c["x"]))
    return rows


def extract_cell_img(img_bgr: np.ndarray, cell: dict, pad: int = 4) -> np.ndarray | None:
    x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
    ih, iw = img_bgr.shape[:2]
    x1, y1 = max(x + pad, 0), max(y + pad, 0)
    x2, y2 = min(x + w - pad, iw), min(y + h - pad, ih)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


# ── Color-block mode ──────────────────────────────────────────────────────────

def detect_color_blocks(img_bgr: np.ndarray,
                         min_saturation: int = 25,
                         min_area_ratio: float = 0.0003,
                         merge_gap: int = 8) -> list[dict]:
    """
    Detect colored (non-white/non-gray) regions using HSV color space.
    Suited for nursing/education yearly calendars where each class
    is a multi-column colored block.

    Returns list of {x, y, w, h, color_bgr, label} dicts.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Mask: colored (not white/gray/black)
    color_mask = (saturation.astype(np.int32) > min_saturation) & (value > 30)
    color_mask = color_mask.astype(np.uint8) * 255

    # Close small gaps within the same block
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (merge_gap, merge_gap))
    closed = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )

    blocks = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_ratio * img_area:
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Aspect ratio filter (avoid thin noise lines)
        if bw < 5 or bh < 5:
            continue

        mask = (labels == i).astype(np.uint8) * 255
        mean_color = cv2.mean(img_bgr, mask=mask)[:3]  # BGR

        blocks.append({
            "x": bx, "y": by, "w": bw, "h": bh,
            "area": area,
            "color_bgr": mean_color,
            "label": i,
        })

    return sorted(blocks, key=lambda b: (b["y"], b["x"]))


def crop_timetable_region(img_bgr: np.ndarray,
                           color_blocks: list[dict]) -> tuple[np.ndarray, int, int]:
    """
    Crop the image to the bounding box that tightly wraps all color blocks,
    plus a margin for headers.
    Returns (cropped_image, x_offset, y_offset).
    """
    if not color_blocks:
        return img_bgr, 0, 0

    min_x = min(b["x"] for b in color_blocks)
    min_y = min(b["y"] for b in color_blocks)
    max_x = max(b["x"] + b["w"] for b in color_blocks)
    max_y = max(b["y"] + b["h"] for b in color_blocks)

    ih, iw = img_bgr.shape[:2]
    # Add header margin (header rows sit above the first colored block)
    margin_top = max(min_y - 60, 0)
    margin_left = max(min_x - 40, 0)
    x2 = min(max_x + 20, iw)
    y2 = min(max_y + 20, ih)

    return img_bgr[margin_top:y2, margin_left:x2], margin_left, margin_top


def auto_detect_mode(img_bgr: np.ndarray) -> str:
    """
    Heuristic: if lots of colored blocks exist relative to image area,
    use color_mode (nursing/yearly calendar style).
    Otherwise use grid_mode.
    """
    blocks = detect_color_blocks(img_bgr)
    h, w = img_bgr.shape[:2]
    total_colored = sum(b["area"] for b in blocks)
    ratio = total_colored / (h * w)
    # If more than 8% of the image is colored blocks → color mode
    return "color" if ratio > 0.08 else "grid"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _nms(cells: list[dict], threshold: float = 0.5) -> list[dict]:
    """Non-maximum suppression: remove highly overlapping cells."""
    cells = sorted(cells, key=lambda c: c["w"] * c["h"], reverse=True)
    kept = []
    for cell in cells:
        x1, y1 = cell["x"], cell["y"]
        x2, y2 = x1 + cell["w"], y1 + cell["h"]
        dominated = False
        for k in kept:
            kx1, ky1 = k["x"], k["y"]
            kx2, ky2 = kx1 + k["w"], ky1 + k["h"]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                smaller = min(cell["w"] * cell["h"], k["w"] * k["h"])
                if smaller > 0 and inter / smaller > threshold:
                    dominated = True
                    break
        if not dominated:
            kept.append(cell)
    return kept
