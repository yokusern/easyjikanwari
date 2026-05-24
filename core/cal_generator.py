"""
Generate .ics (iCalendar) files from structured schedule data.

Two generators:
  - weekly_ics  : Fixed repeating weekly schedule (standard university)
  - events_ics  : List of specific one-time or span events (nursing/education)
"""

import uuid
from datetime import date, datetime, time, timedelta

import pandas as pd
from icalendar import Calendar, Event

_WEEKDAY_BYDAY = {0: "MO", 1: "TU", 2: "WE", 3: "TH", 4: "FR", 5: "SA", 6: "SU"}


def _base_cal() -> Calendar:
    cal = Calendar()
    cal.add("prodid", "-//Smart JIKANWARI//smart-jikanwari//JP")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    return cal


def _make_event(summary: str, dtstart: datetime, dtend: datetime,
                description: str = "") -> Event:
    ev = Event()
    ev.add("summary", summary)
    ev.add("dtstart", dtstart)
    ev.add("dtend", dtend)
    if description:
        ev.add("description", description)
    ev.add("uid", str(uuid.uuid4()))
    return ev


# ── Weekly (grid-mode) ────────────────────────────────────────────────────────

def weekly_ics(
    df: pd.DataFrame,
    time_slots: list[dict],
    col_headers: list[dict],
    semester_start: date,
    semester_end: date,
) -> bytes:
    """
    Generate .ics with RRULE:FREQ=WEEKLY events.

    df          – rows=periods, cols=days
    time_slots  – list of {period, start_time, end_time} per row
    col_headers – list of {day_of_week, text} per column
    """
    cal = _base_cal()

    for col_idx, header in enumerate(col_headers):
        day_of_week: int | None = header.get("day_of_week")
        if day_of_week is None:
            continue

        for row_idx, ts in enumerate(time_slots):
            if row_idx >= len(df):
                break
            if col_idx >= len(df.columns):
                break

            class_name = str(df.iloc[row_idx, col_idx]).strip()
            if not class_name or class_name in ("nan", ""):
                continue

            start_t: time | None = ts.get("start_time")
            end_t: time | None = ts.get("end_time")
            if not start_t or not end_t:
                continue

            # Find first occurrence of this weekday on/after semester_start
            days_ahead = (day_of_week - semester_start.weekday()) % 7
            first = semester_start + timedelta(days=days_ahead)
            if first > semester_end:
                continue

            until = datetime(semester_end.year, semester_end.month,
                             semester_end.day, 23, 59, 59)

            ev = _make_event(
                class_name,
                datetime(first.year, first.month, first.day, start_t.hour, start_t.minute),
                datetime(first.year, first.month, first.day, end_t.hour, end_t.minute),
            )
            ev.add("rrule", {
                "freq": "weekly",
                "until": until,
                "byday": _WEEKDAY_BYDAY[day_of_week],
            })
            cal.add_component(ev)

    return cal.to_ical()


# ── Variable (specific dates / color-block mode) ──────────────────────────────

def events_ics(events: list[dict]) -> bytes:
    """
    Generate .ics from a list of event dicts:
      {summary, start_date, end_date, start_time, end_time}

    If start_date == end_date → single-day timed event.
    If start_date < end_date  → multi-day event (one event per day with the time slot).
    """
    cal = _base_cal()

    for ev_data in events:
        summary = str(ev_data.get("summary", "")).strip()
        if not summary:
            continue

        start_date: date = ev_data.get("start_date", date.today())
        end_date: date = ev_data.get("end_date", start_date)
        start_t: time | None = ev_data.get("start_time")
        end_t: time | None = ev_data.get("end_time")

        if start_t and end_t:
            # Timed event (possibly spanning multiple days)
            current = start_date
            while current <= end_date:
                # Skip weekends unless explicitly included
                ev = _make_event(
                    summary,
                    datetime(current.year, current.month, current.day,
                             start_t.hour, start_t.minute),
                    datetime(current.year, current.month, current.day,
                             end_t.hour, end_t.minute),
                )
                cal.add_component(ev)
                current += timedelta(days=1)
        else:
            # All-day event
            ev = Event()
            ev.add("summary", summary)
            ev.add("dtstart", start_date)
            ev.add("dtend", end_date + timedelta(days=1))
            ev.add("uid", str(uuid.uuid4()))
            cal.add_component(ev)

    return cal.to_ical()


# ── DataFrame → events list (for variable/grid variable mode) ─────────────────

def df_variable_to_events(
    df: pd.DataFrame,
    time_slots: list[dict],
    col_headers: list[dict],
) -> list[dict]:
    """Convert a variable-mode DataFrame to events list for events_ics()."""
    events = []
    for col_idx, header in enumerate(col_headers):
        specific_date: date | None = header.get("date")
        if specific_date is None:
            continue
        for row_idx, ts in enumerate(time_slots):
            if row_idx >= len(df):
                break
            if col_idx >= len(df.columns):
                break
            class_name = str(df.iloc[row_idx, col_idx]).strip()
            if not class_name or class_name in ("nan", ""):
                continue
            events.append({
                "summary": class_name,
                "start_date": specific_date,
                "end_date": specific_date,
                "start_time": ts.get("start_time"),
                "end_time": ts.get("end_time"),
            })
    return events
