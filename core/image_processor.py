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
                         min_saturation: int = 40,
                         min_area_ratio: float = 0.0006,
                         merge_gap: int = 8) -> list[dict]:
    """
    Detect colored (non-white/gray) regions in the image.
    Returns sorted list of {x, y, w, h, area, color_bgr}.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] > min_saturation) & (hsv[:, :, 2] > 50)
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
        if bw < 8 or bh < 8:
            continue
        m = (labels == i).astype(np.uint8) * 255
        color_bgr = cv2.mean(img_bgr, mask=m)[:3]
        blocks.append({
            "x": bx, "y": by, "w": bw, "h": bh,
            "area": area, "color_bgr": color_bgr,
        })

    return sorted(blocks, key=lambda b: (b["y"], b["x"]))


def _block_hsv(block: dict) -> tuple[int, int, int]:
    bgr = block["color_bgr"]
    pixel = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _hue_dist(h1: int, h2: int) -> int:
    return min(abs(h1 - h2), 180 - abs(h1 - h2))


def _color_dist(h1, s1, h2, s2) -> float:
    return _hue_dist(h1, h2) * 1.5 + abs(s1 - s2) * 0.5


def _nearest_group_idx(groups: list[dict], h: int, s: int) -> int:
    best_i, best_d = 0, float('inf')
    for i, g in enumerate(groups):
        d = _color_dist(h, s, g["h"], g["s"])
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def group_blocks_by_color(blocks: list[dict],
                           hue_tol: int = 22,
                           sat_tol: int = 65,
                           max_groups: int = 14) -> list[dict]:
    """
    Group blocks with visually similar colors into course groups.
    Uses nearest-match with running center update, then merges small/excess groups.

    Returns list sorted by block count (most common first).
    Each group: {color_bgr, rgb_css, h, s, v, blocks, name, skip}
    """
    groups: list[dict] = []

    for block in blocks:
        h, s, v = _block_hsv(block)

        best_i, best_d = None, float('inf')
        for gi, g in enumerate(groups):
            if _hue_dist(h, g["h"]) < hue_tol and abs(s - g["s"]) < sat_tol:
                d = _color_dist(h, s, g["h"], g["s"])
                if d < best_d:
                    best_d, best_i = d, gi

        if best_i is not None:
            g = groups[best_i]
            g["blocks"].append(block)
            # Update center with running average (weighted toward larger groups)
            n = len(g["blocks"])
            g["h"] = (g["h"] * (n - 1) + h) // n
            g["s"] = (g["s"] * (n - 1) + s) // n
            g["v"] = (g["v"] * (n - 1) + v) // n
        else:
            bgr = block["color_bgr"]
            r, gg, b2 = int(bgr[2]), int(bgr[1]), int(bgr[0])
            groups.append({
                "h": h, "s": s, "v": v,
                "color_bgr": bgr,
                "rgb_css": f"rgb({r},{gg},{b2})",
                "blocks": [block],
                "name": "",
                "skip": False,
            })

    # Sort: largest groups first
    groups.sort(key=lambda g: len(g["blocks"]), reverse=True)

    # Merge singletons into nearest group
    main, small = [], []
    for g in groups:
        (main if len(g["blocks"]) >= 3 else small).append(g)

    for sg in small:
        if main:
            main[_nearest_group_idx(main, sg["h"], sg["s"])]["blocks"].extend(sg["blocks"])
        else:
            main.append(sg)

    # If still too many groups, keep merging smallest into nearest
    main.sort(key=lambda g: len(g["blocks"]), reverse=True)
    while len(main) > max_groups:
        main.sort(key=lambda g: len(g["blocks"]))
        victim = main.pop(0)
        main[_nearest_group_idx(main, victim["h"], victim["s"])]["blocks"].extend(victim["blocks"])

    # Recompute representative color from actual block pixels (most common hue)
    for g in main:
        hs_list = [_block_hsv(b) for b in g["blocks"]]
        med_h = int(np.median([x[0] for x in hs_list]))
        med_s = int(np.median([x[1] for x in hs_list]))
        med_v = int(np.median([x[2] for x in hs_list]))
        pixel = np.uint8([[[med_h, med_s, med_v]]])
        bgr_arr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0][0]
        r, gg, b2 = int(bgr_arr[2]), int(bgr_arr[1]), int(bgr_arr[0])
        g["rgb_css"] = f"rgb({r},{gg},{b2})"

    main.sort(key=lambda g: len(g["blocks"]), reverse=True)
    return main


def draw_group_preview(img_bgr: np.ndarray,
                        groups: list[dict]) -> np.ndarray:
    """
    Outline each color group's blocks with a colored border + group number badge.
    """
    out = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Generate distinct label colors
    label_colors = [
        (255, 255, 255),  # white text on dark badge
    ]

    for gi, group in enumerate(groups):
        if group.get("skip"):
            continue
        label = str(gi + 1)

        # Badge background: derive a darkened version of the group color
        bgr = group["color_bgr"]
        px = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
        hsv_px = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]
        dark_px = np.uint8([[[hsv_px[0], min(255, int(hsv_px[1]) + 60), max(0, int(hsv_px[2]) - 80)]]])
        badge_bgr = cv2.cvtColor(dark_px, cv2.COLOR_HSV2BGR)[0][0]
        badge_color = (int(badge_bgr[0]), int(badge_bgr[1]), int(badge_bgr[2]))

        for block in group["blocks"]:
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), 3)
            cv2.rectangle(out, (x, y), (x + w, y + h), badge_color, 1)
            fs = max(min(h / 28, 1.1), 0.5)
            (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
            pad = 4
            cv2.rectangle(out, (x, y), (x + tw + pad * 2, y + th + pad * 2), badge_color, -1)
            cv2.putText(out, label, (x + pad, y + th + pad), font, fs, (255, 255, 255), 2, cv2.LINE_AA)

    return out
