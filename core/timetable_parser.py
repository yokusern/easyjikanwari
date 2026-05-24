"""
Parse structured schedule data from image analysis or manual input.
"""

import re
from datetime import date, time, timedelta

import pandas as pd

_DAY_MAP: dict[str, int] = {
    "月": 0, "月曜": 0, "月曜日": 0, "MON": 0, "MONDAY": 0,
    "火": 1, "火曜": 1, "火曜日": 1, "TUE": 1, "TUESDAY": 1,
    "水": 2, "水曜": 2, "水曜日": 2, "WED": 2, "WEDNESDAY": 2,
    "木": 3, "木曜": 3, "木曜日": 3, "THU": 3, "THURSDAY": 3,
    "金": 4, "金曜": 4, "金曜日": 4, "FRI": 4, "FRIDAY": 4,
    "土": 5, "土曜": 5, "土曜日": 5, "SAT": 5, "SATURDAY": 5,
    "日": 6, "日曜": 6, "日曜日": 6, "SUN": 6, "SUNDAY": 6,
}

_KANJI_NUM: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_DEFAULT_PERIOD_TIMES: dict[int, tuple[time, time]] = {
    1: (time(8, 50), time(10, 20)),
    2: (time(10, 30), time(12, 0)),
    3: (time(13, 0), time(14, 30)),
    4: (time(14, 40), time(16, 10)),
    5: (time(16, 20), time(17, 50)),
    6: (time(18, 0), time(19, 30)),
    7: (time(19, 40), time(21, 10)),
    8: (time(21, 20), time(22, 50)),
}


def parse_time_slot(text: str) -> dict:
    r = {"period": None, "start_time": None, "end_time": None, "raw": text}
    m = re.search(r"(\d{1,2}):(\d{2})\s*[~\-〜～]\s*(\d{1,2}):(\d{2})", text)
    if m:
        r["start_time"] = time(int(m.group(1)), int(m.group(2)))
        r["end_time"]   = time(int(m.group(3)), int(m.group(4)))
    m2 = re.search(r"(\d{1,2})[限時]", text)
    if m2:
        p = int(m2.group(1))
        r["period"] = p
        if r["start_time"] is None and p in _DEFAULT_PERIOD_TIMES:
            r["start_time"], r["end_time"] = _DEFAULT_PERIOD_TIMES[p]
    if r["period"] is None:
        for k, n in _KANJI_NUM.items():
            if f"{k}限" in text:
                r["period"] = n
                if r["start_time"] is None and n in _DEFAULT_PERIOD_TIMES:
                    r["start_time"], r["end_time"] = _DEFAULT_PERIOD_TIMES[n]
                break
    if r["start_time"] is None:
        if "午前" in text:
            r["start_time"], r["end_time"] = time(9, 0), time(12, 0)
        elif "午後" in text:
            r["start_time"], r["end_time"] = time(13, 0), time(17, 0)
    return r


def parse_day_of_week(text: str) -> int | None:
    upper = text.strip().upper()
    for key, val in _DAY_MAP.items():
        if key.upper() in upper:
            return val
    return None


def parse_specific_date(text: str, default_year: int | None = None) -> date | None:
    year = default_year or date.today().year
    m = re.search(r"(\d{1,2})/(\d{1,2})", text)
    if m:
        try:
            return date(year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})月(\d{1,2})日", text)
    if m:
        try:
            return date(year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    return None


def detect_layout_type(grid_texts: list[list[str]]) -> str:
    if not grid_texts:
        return "weekly"
    header = " ".join(grid_texts[0])
    if re.search(r"\d{1,2}/\d{1,2}", header):
        return "variable"
    if re.search(r"\d{1,2}月\d{1,2}日", header):
        return "variable"
    for day in ["月", "火", "水", "木", "金"]:
        if day in header:
            return "weekly"
    return "weekly"


def parse_grid_schedule(grid_texts: list[list[str]]) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    if not grid_texts or len(grid_texts) < 2:
        return pd.DataFrame(), [], []
    header_row = grid_texts[0]
    layout = detect_layout_type(grid_texts)
    col_headers: list[dict] = []
    for i, cell in enumerate(header_row):
        if i == 0:
            col_headers.append({"type": "time_label", "text": cell})
            continue
        if layout == "weekly":
            col_headers.append({"type": "day",
                                 "day_of_week": parse_day_of_week(cell), "text": cell})
        else:
            col_headers.append({"type": "date",
                                 "date": parse_specific_date(cell), "text": cell})
    time_slots: list[dict] = []
    data_rows: list[list[str]] = []
    for row in grid_texts[1:]:
        if not row:
            continue
        time_slots.append(parse_time_slot(row[0]))
        data_rows.append(row[1:] if len(row) > 1 else [])
    row_labels = []
    for ts in time_slots:
        if ts["period"]:
            lbl = f"{ts['period']}限"
            if ts["start_time"]:
                lbl += f" ({ts['start_time'].strftime('%H:%M')})"
        elif ts["start_time"]:
            lbl = ts["start_time"].strftime("%H:%M")
        else:
            lbl = ts["raw"] or "?"
        row_labels.append(lbl)
    col_labels = [h["text"] for h in col_headers[1:]]
    n_cols = max((len(r) for r in data_rows), default=len(col_labels))
    for row in data_rows:
        while len(row) < n_cols:
            row.append("")
    df = pd.DataFrame(
        data_rows,
        index=row_labels[:len(data_rows)],
        columns=col_labels[:n_cols] if col_labels else list(range(n_cols)),
    )
    return df, time_slots, col_headers[1:]


def map_blocks_to_schedule(
    blocks: list[dict],
    semester_start: date,
    semester_end: date,
    periods_per_day: int = 6,
    period_times: dict | None = None,
) -> list[dict]:
    """
    Map color blocks to events using their relative x/y position.
    Uses the actual bounding box of all blocks (not full image size)
    for more accurate date/period mapping.
    """
    if not blocks:
        return []

    pt = period_times or _DEFAULT_PERIOD_TIMES

    min_x = min(b["x"] for b in blocks)
    max_x = max(b["x"] + b["w"] for b in blocks)
    min_y = min(b["y"] for b in blocks)
    max_y = max(b["y"] + b["h"] for b in blocks)
    x_range = max(max_x - min_x, 1)
    y_range = max(max_y - min_y, 1)
    total_days = max((semester_end - semester_start).days, 1)

    events = []
    for block in blocks:
        name = block.get("text", "").strip()
        if not name:
            continue

        xs = (block["x"] - min_x) / x_range
        xe = (block["x"] + block["w"] - min_x) / x_range
        ev_start = semester_start + timedelta(days=int(xs * total_days))
        ev_end   = semester_start + timedelta(days=int(xe * total_days))
        if ev_end < ev_start:
            ev_end = ev_start

        yc = (block["y"] + block["h"] / 2 - min_y) / y_range
        period = max(1, min(int(yc * periods_per_day) + 1, periods_per_day))
        ts = pt.get(period, (time(9, 0), time(10, 30)))

        events.append({
            "summary":    name,
            "start_date": ev_start,
            "end_date":   ev_end,
            "start_time": ts[0],
            "end_time":   ts[1],
            "period":     period,
        })
    return events
