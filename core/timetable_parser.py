"""
Parse OCR output into structured schedule data.

Handles two layout types:
  - "grid"  : Standard weekly timetable (Mon-Fri columns, period rows)
  - "color" : Yearly/monthly calendar with colored multi-span blocks
              (typical in nursing/education schools)
"""

import re
from datetime import date, time, timedelta

import pandas as pd


# ── Day / period constants ────────────────────────────────────────────────────

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

# Fallback period times (Japanese standard)
_DEFAULT_PERIOD_TIMES: dict[int, tuple[time, time]] = {
    1: (time(8, 50), time(10, 20)),
    2: (time(10, 30), time(12, 0)),
    3: (time(13, 0), time(14, 30)),
    4: (time(14, 40), time(16, 10)),
    5: (time(16, 20), time(17, 50)),
    6: (time(18, 0), time(19, 30)),
}


# ── Public helpers ────────────────────────────────────────────────────────────

def parse_time_slot(text: str) -> dict:
    """
    Parse a time-slot cell.
    Returns {period, start_time, end_time, raw}.
    Any field may be None if undetectable.
    """
    result = {"period": None, "start_time": None, "end_time": None, "raw": text}
    text = text.strip()

    # "9:00~10:30" / "9:00-10:30" / "9:00〜10:30"
    m = re.search(r"(\d{1,2}):(\d{2})\s*[~\-〜～]\s*(\d{1,2}):(\d{2})", text)
    if m:
        result["start_time"] = time(int(m.group(1)), int(m.group(2)))
        result["end_time"] = time(int(m.group(3)), int(m.group(4)))

    # "N限" (digit)
    m2 = re.search(r"(\d{1,2})[限時]", text)
    if m2:
        p = int(m2.group(1))
        result["period"] = p
        if result["start_time"] is None and p in _DEFAULT_PERIOD_TIMES:
            result["start_time"], result["end_time"] = _DEFAULT_PERIOD_TIMES[p]

    # "一限" (kanji)
    if result["period"] is None:
        for kanji, num in _KANJI_NUM.items():
            if f"{kanji}限" in text or f"{kanji}時限" in text:
                result["period"] = num
                if result["start_time"] is None and num in _DEFAULT_PERIOD_TIMES:
                    result["start_time"], result["end_time"] = _DEFAULT_PERIOD_TIMES[num]
                break

    # "午前" / "午後"
    if result["start_time"] is None:
        if "午前" in text:
            result["start_time"], result["end_time"] = time(9, 0), time(12, 0)
        elif "午後" in text:
            result["start_time"], result["end_time"] = time(13, 0), time(17, 0)

    return result


def parse_day_of_week(text: str) -> int | None:
    """Return 0-6 (Mon-Sun) or None."""
    upper = text.strip().upper()
    for key, val in _DAY_MAP.items():
        if key.upper() in upper:
            return val
    return None


def parse_specific_date(text: str, default_year: int | None = None) -> date | None:
    """Parse 'M/D', 'M月D日', etc. Returns date or None."""
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
    """
    'weekly'  → header has 月火水木金 style day names
    'variable'→ header has explicit dates (4/7, 4月7日…)
    """
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


# ── Grid-mode parsing ─────────────────────────────────────────────────────────

def parse_grid_schedule(grid_texts: list[list[str]]) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """
    Parse a 2-D grid of OCR text into a structured schedule.

    Returns:
        df            – DataFrame[row=period_label, col=day/date_label] = class name
        time_slots    – list of parse_time_slot() results (one per data row)
        col_headers   – list of {type, day_of_week/date, text} (one per data col)
    """
    if not grid_texts or len(grid_texts) < 2:
        return pd.DataFrame(), [], []

    header_row = grid_texts[0]

    # Build column header metadata
    col_headers: list[dict] = []
    layout = detect_layout_type(grid_texts)

    for i, cell in enumerate(header_row):
        if i == 0:
            col_headers.append({"type": "time_label", "text": cell})
            continue
        if layout == "weekly":
            col_headers.append({"type": "day", "day_of_week": parse_day_of_week(cell),
                                 "text": cell})
        else:
            col_headers.append({"type": "date", "date": parse_specific_date(cell),
                                 "text": cell})

    # Parse time-slot column (first column of each data row)
    time_slots: list[dict] = []
    data_rows: list[list[str]] = []
    for row in grid_texts[1:]:
        if not row:
            continue
        time_slots.append(parse_time_slot(row[0]))
        data_rows.append(row[1:] if len(row) > 1 else [])

    # Build row labels
    row_labels: list[str] = []
    for ts in time_slots:
        if ts["period"]:
            label = f"{ts['period']}限"
            if ts["start_time"]:
                label += f" ({ts['start_time'].strftime('%H:%M')})"
        elif ts["start_time"]:
            label = ts["start_time"].strftime("%H:%M")
        else:
            label = ts["raw"] or "?"
        row_labels.append(label)

    # Column labels (data columns only, i.e. skip index-0 time label)
    col_labels = [h["text"] for h in col_headers[1:]]

    # Normalise row lengths
    n_cols = max(len(r) for r in data_rows) if data_rows else len(col_labels)
    for row in data_rows:
        while len(row) < n_cols:
            row.append("")

    df = pd.DataFrame(
        data_rows,
        index=row_labels[:len(data_rows)],
        columns=col_labels[:n_cols] if col_labels else list(range(n_cols)),
    )
    return df, time_slots, col_headers[1:]


# ── Color-block mode parsing ──────────────────────────────────────────────────

def map_blocks_to_schedule(
    color_blocks: list[dict],
    img_width: int,
    img_height: int,
    semester_start: date,
    semester_end: date,
    periods_per_day: int = 6,
) -> list[dict]:
    """
    Map detected color blocks to (class_name, start_date, end_date, period) tuples
    using their relative x/y position in the image.

    This is a heuristic approach: the x position maps linearly to the date
    range [semester_start … semester_end], and y maps to period 1-N.

    Returns list of event dicts ready for cal_generator.
    """
    if not color_blocks:
        return []

    total_days = (semester_end - semester_start).days + 1

    events = []
    for block in color_blocks:
        x, y, w, h = block["x"], block["y"], block["w"], block["h"]
        name = block.get("text", "").strip()
        if not name:
            continue

        # x → date mapping (linear across image width)
        start_day_offset = int((x / img_width) * total_days)
        end_day_offset = int(((x + w) / img_width) * total_days)
        ev_start = semester_start + timedelta(days=start_day_offset)
        ev_end = semester_start + timedelta(days=end_day_offset)

        # y → period mapping (linear across image height)
        period_idx = int((y / img_height) * periods_per_day) + 1
        period_idx = max(1, min(period_idx, periods_per_day))
        ts = _DEFAULT_PERIOD_TIMES.get(period_idx, (time(9, 0), time(10, 30)))

        events.append({
            "summary": name,
            "start_date": ev_start,
            "end_date": ev_end,
            "start_time": ts[0],
            "end_time": ts[1],
            "period": period_idx,
        })

    return events
