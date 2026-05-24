"""
Image processing for yearly variable timetable calendars.
Primary use case: nursing/education school yearly schedules
where colored blocks = specific courses on specific dates.
"""

import cv2
import numpy as np
from PIL import Image


def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_color_blocks(img_bgr: np.ndarray,
                         min_saturation: int = 35,
                         min_area_ratio: float = 0.0004,
                         merge_gap: int = 12) -> list[dict]:
    """
    Detect colored (non-white/gray) regions in the image.
    Returns sorted list of {x, y, w, h, area, color_bgr}.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] > min_saturation) & (hsv[:, :, 2] > 40)
    ).astype(np.uint8) * 255

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
        if bw < 6 or bh < 6:
            continue
        m = (labels == i).astype(np.uint8) * 255
        color_bgr = cv2.mean(img_bgr, mask=m)[:3]
        blocks.append({
            "x": bx, "y": by, "w": bw, "h": bh,
            "area": area, "color_bgr": color_bgr,
        })

    return sorted(blocks, key=lambda b: (b["y"], b["x"]))


def group_blocks_by_color(blocks: list[dict],
                           hue_tol: int = 14,
                           sat_tol: int = 45) -> list[dict]:
    """
    Group blocks with visually similar colors.
    Each group = one course. User labels the group once.

    Returns list of groups sorted by block count (most common first):
      {color_bgr, rgb_css, h, s, v, blocks: [...], name: ""}
    """
    groups: list[dict] = []

    for block in blocks:
        bgr = block["color_bgr"]
        pixel = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        matched = False
        for g in groups:
            hd = min(abs(h - g["h"]), 180 - abs(h - g["h"]))
            if hd < hue_tol and abs(s - g["s"]) < sat_tol:
                g["blocks"].append(block)
                matched = True
                break

        if not matched:
            r, gg, b2 = int(bgr[2]), int(bgr[1]), int(bgr[0])
            groups.append({
                "h": h, "s": s, "v": v,
                "color_bgr": bgr,
                "rgb_css": f"rgb({r},{gg},{b2})",
                "blocks": [block],
                "name": "",
                "skip": False,
            })

    return sorted(groups, key=lambda g: len(g["blocks"]), reverse=True)


def draw_group_preview(img_bgr: np.ndarray,
                        groups: list[dict]) -> np.ndarray:
    """
    Outline each color group's blocks with a white border + group number.
    Helps users identify which blocks belong to each group.
    """
    out = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for gi, group in enumerate(groups):
        if group.get("skip"):
            continue
        label = str(gi + 1)
        for block in group["blocks"]:
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 3)
            cv2.rectangle(out, (x, y), (x + w, y + h), (30, 30, 30), 1)
            fs = max(min(h / 30, 1.2), 0.4)
            (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
            cv2.rectangle(out, (x, y), (x + tw + 6, y + th + 6), (30, 30, 30), -1)
            cv2.putText(out, label, (x + 3, y + th + 3), font, fs, (255, 255, 255), 2)

    return out
