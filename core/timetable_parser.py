"""
Map color-grouped blocks to calendar events.
Primary use case: nursing/education yearly calendar timetables.
"""

from datetime import date, time, timedelta

# Standard Japanese university period times (customizable in UI)
DEFAULT_PERIOD_TIMES: dict[int, tuple[time, time]] = {
    1: (time(8, 50),  time(10, 20)),
    2: (time(10, 30), time(12,  0)),
    3: (time(13,  0), time(14, 30)),
    4: (time(14, 40), time(16, 10)),
    5: (time(16, 20), time(17, 50)),
    6: (time(18,  0), time(19, 30)),
    7: (time(19, 40), time(21, 10)),
    8: (time(21, 20), time(22, 50)),
}


def groups_to_events(
    groups: list[dict],
    semester_start: date,
    semester_end: date,
    periods_per_day: int = 6,
    period_times: dict | None = None,
) -> list[dict]:
    """
    Convert color groups → event list for cal_generator.

    For each group:
      - name: user-supplied class name
      - Each block → one event with date range from x-position
                      and period from y-position
    """
    pt = period_times or DEFAULT_PERIOD_TIMES
    total_days = max((semester_end - semester_start).days, 1)

    # Use the bounding box of ALL blocks across all groups for mapping
    all_blocks = [b for g in groups for b in g["blocks"] if not g.get("skip")]
    if not all_blocks:
        return []

    min_x = min(b["x"] for b in all_blocks)
    max_x = max(b["x"] + b["w"] for b in all_blocks)
    min_y = min(b["y"] for b in all_blocks)
    max_y = max(b["y"] + b["h"] for b in all_blocks)
    x_range = max(max_x - min_x, 1)
    y_range = max(max_y - min_y, 1)

    events = []
    for group in groups:
        if group.get("skip"):
            continue
        name = group.get("name", "").strip()
        if not name:
            continue

        for block in group["blocks"]:
            xs = (block["x"] - min_x) / x_range
            xe = (block["x"] + block["w"] - min_x) / x_range
            ev_start = semester_start + timedelta(days=int(xs * total_days))
            ev_end   = semester_start + timedelta(days=int(xe * total_days))
            if ev_end < ev_start:
                ev_end = ev_start

            # y-center → period
            yc = (block["y"] + block["h"] / 2 - min_y) / y_range
            period = max(1, min(int(yc * periods_per_day) + 1, periods_per_day))
            ts = pt.get(period, (time(9, 0), time(10, 30)))

            events.append({
                "summary":    name,
                "start_date": ev_start,
                "end_date":   ev_end,
                "start_time": ts[0],
                "end_time":   ts[1],
            })

    return events
