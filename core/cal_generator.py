"""
Generate valid .ics files compatible with iOS Calendar, Google Calendar, TimeTree.

Key requirements for iOS Calendar:
  - DTSTART/DTEND with explicit TZID=Asia/Tokyo
  - Valid VTIMEZONE component for JST
  - RRULE with correct BYDAY format
  - PRODID and VERSION required
"""

import uuid
from datetime import date, datetime, time, timedelta

import pandas as pd
from icalendar import Calendar, Event, Timezone, TimezoneStandard

_WEEKDAY_BYDAY = {0: "MO", 1: "TU", 2: "WE", 3: "TH", 4: "FR", 5: "SA", 6: "SU"}

JST_OFFSET = timedelta(hours=9)


def _jst(d: date, t: time) -> datetime:
    """Create a JST-aware datetime from date + time."""
    from datetime import timezone
    return datetime(d.year, d.month, d.day, t.hour, t.minute,
                    tzinfo=timezone(JST_OFFSET))


def _base_cal() -> Calendar:
    cal = Calendar()
    cal.add("prodid", "-//EasyJikanwari//easyjikanwari//JP")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("method", "PUBLISH")

    # VTIMEZONE component for Asia/Tokyo (required for iOS Calendar compatibility)
    tz = Timezone()
    tz.add("tzid", "Asia/Tokyo")
    std = TimezoneStandard()
    std.add("dtstart", datetime(1970, 1, 1))
    std.add("tzoffsetfrom", timedelta(hours=9))
    std.add("tzoffsetto", timedelta(hours=9))
    std.add("tzname", "JST")
    tz.add_component(std)
    cal.add_component(tz)

    return cal


# ── Weekly (Mon-Fri grid) ─────────────────────────────────────────────────────

def weekly_ics(
    df: pd.DataFrame,
    time_slots: list[dict],
    col_headers: list[dict],
    semester_start: date,
    semester_end: date,
) -> bytes:
    cal = _base_cal()
    count = 0

    for col_idx, header in enumerate(col_headers):
        day_of_week: int | None = header.get("day_of_week")
        if day_of_week is None:
            continue

        for row_idx, ts in enumerate(time_slots):
            if row_idx >= len(df) or col_idx >= len(df.columns):
                break
            class_name = str(df.iloc[row_idx, col_idx]).strip()
            if not class_name or class_name in ("nan", ""):
                continue

            start_t: time | None = ts.get("start_time")
            end_t: time | None = ts.get("end_time")
            if not start_t or not end_t:
                continue

            days_ahead = (day_of_week - semester_start.weekday()) % 7
            first = semester_start + timedelta(days=days_ahead)
            if first > semester_end:
                continue

            from datetime import timezone
            until = datetime(semester_end.year, semester_end.month,
                             semester_end.day, 23, 59, 59,
                             tzinfo=timezone(JST_OFFSET))

            ev = Event()
            ev.add("summary", class_name)
            ev.add("dtstart", _jst(first, start_t))
            ev.add("dtend",   _jst(first, end_t))
            ev.add("rrule", {
                "freq": "weekly",
                "until": until,
                "byday": _WEEKDAY_BYDAY[day_of_week],
            })
            ev.add("uid", str(uuid.uuid4()))
            cal.add_component(ev)
            count += 1

    return cal.to_ical(), count


# ── Specific events (variable / color-block mode) ─────────────────────────────

def events_ics(events: list[dict]) -> tuple[bytes, int]:
    cal = _base_cal()
    count = 0

    for ev_data in events:
        summary = str(ev_data.get("summary", "")).strip()
        if not summary or summary in ("nan", ""):
            continue

        start_date: date = ev_data.get("start_date", date.today())
        end_date: date   = ev_data.get("end_date", start_date)
        start_t: time | None = ev_data.get("start_time")
        end_t: time | None   = ev_data.get("end_time")

        if start_t and end_t:
            current = start_date
            while current <= end_date:
                ev = Event()
                ev.add("summary", summary)
                ev.add("dtstart", _jst(current, start_t))
                ev.add("dtend",   _jst(current, end_t))
                ev.add("uid", str(uuid.uuid4()))
                cal.add_component(ev)
                count += 1
                current += timedelta(days=1)
        else:
            ev = Event()
            ev.add("summary", summary)
            ev.add("dtstart", start_date)
            ev.add("dtend",   end_date + timedelta(days=1))
            ev.add("uid", str(uuid.uuid4()))
            cal.add_component(ev)
            count += 1

    return cal.to_ical(), count


def df_variable_to_events(
    df: pd.DataFrame,
    time_slots: list[dict],
    col_headers: list[dict],
) -> list[dict]:
    events = []
    for col_idx, header in enumerate(col_headers):
        d: date | None = header.get("date")
        if d is None:
            continue
        for row_idx, ts in enumerate(time_slots):
            if row_idx >= len(df) or col_idx >= len(df.columns):
                break
            class_name = str(df.iloc[row_idx, col_idx]).strip()
            if not class_name or class_name in ("nan", ""):
                continue
            events.append({
                "summary":    class_name,
                "start_date": d,
                "end_date":   d,
                "start_time": ts.get("start_time"),
                "end_time":   ts.get("end_time"),
            })
    return events
