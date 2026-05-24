"""
EasyJikanwari — 時間割をカレンダーアプリに自動登録
スマホ完結 / 完全無料 / API不要
"""

import base64
import io
from datetime import date, time, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.image_processor import (
    auto_detect_mode,
    detect_color_blocks,
    detect_grid_cells,
    enhance_for_ocr,
    extract_cell_img,
    pil_to_bgr,
    sort_to_grid,
)
from core.ocr_engine import (
    is_tesseract_available,
    ocr_cell,
    ocr_large_region,
)
from core.timetable_parser import (
    detect_layout_type,
    map_blocks_to_schedule,
    parse_grid_schedule,
    _DEFAULT_PERIOD_TIMES,
)
from core.cal_generator import (
    df_variable_to_events,
    events_ics,
    weekly_ics,
)

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EasyJikanwari",
    page_icon="📅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F2F2F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

/* ── Card ── */
.card {
    background: #fff;
    border-radius: 16px;
    padding: 1.25rem 1.25rem 1rem;
    margin: 0.6rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ── Section heading ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8E8E93;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 1.4rem 0 0.3rem 0.3rem;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    height: 3.2rem;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 12px;
    border: none;
}
.stButton > button[kind="primary"] {
    background: #007AFF !important;
    color: #fff !important;
}
.stButton > button[kind="secondary"] {
    background: #E5E5EA !important;
    color: #1C1C1E !important;
}
.stDownloadButton > button {
    width: 100%;
    height: 3.2rem;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    background: #34C759 !important;
    color: #fff !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #C7C7CC;
    border-radius: 16px;
    padding: 0.5rem;
    background: #fff;
}

/* ── Radio ── */
[data-testid="stRadio"] label {
    font-size: 0.95rem;
}

/* ── Data editor ── */
[data-testid="stDataEditor"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Alert tweaks ── */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ── Export button (custom HTML) ── */
.export-btn {
    display: block;
    text-align: center;
    background: #34C759;
    color: #fff !important;
    padding: 1rem;
    border-radius: 14px;
    text-decoration: none !important;
    font-size: 1.05rem;
    font-weight: 700;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(52,199,89,0.35);
}
.export-btn:hover { background: #28a745; }

/* ── Badge ── */
.badge {
    display: inline-block;
    background: #007AFF;
    color: #fff;
    border-radius: 999px;
    padding: 0.1rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.4rem;
}
.badge-green { background: #34C759; }
.badge-orange { background: #FF9500; }

/* ── Count chip ── */
.count-chip {
    display: inline-block;
    background: #E5E5EA;
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #3A3A3C;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def badge(text: str, color: str = "") -> str:
    cls = f"badge {color}".strip()
    return f'<span class="{cls}">{text}</span>'


def section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


def ics_html_link(ics_bytes: bytes, filename: str = "timetable.ics") -> str:
    """
    Data-URI download link — more reliable than st.download_button on iOS Safari.
    Tapping this link on iPhone triggers the 'Open in Calendar' dialog directly.
    """
    b64 = base64.b64encode(ics_bytes).decode()
    return (
        f'<a class="export-btn" '
        f'href="data:text/calendar;charset=utf-8;base64,{b64}" '
        f'download="{filename}">'
        f'📲&nbsp;&nbsp;カレンダーアプリに登録する (.ics)</a>'
    )


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("## 📅 EasyJikanwari")
st.caption("時間割の画像 → iOSカレンダー・Googleカレンダーに自動登録")

# ─── Tesseract check ──────────────────────────────────────────────────────────

tess_ok = is_tesseract_available()
if not tess_ok:
    st.warning(
        "OCRエンジン（Tesseract）が見つかりません。"
        "自動認識は使えませんが、手動入力モードは利用可能です。"
    )

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 1 　画像をアップロード")

uploaded = st.file_uploader(
    "時間割の画像を選択（スクリーンショット・写真どちらも対応）",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2rem 1rem;color:#8E8E93;">
        <div style="font-size:2.5rem;margin-bottom:0.5rem;">📷</div>
        <div style="font-weight:600;color:#3A3A3C;margin-bottom:0.3rem;">
            時間割の画像を選んでください
        </div>
        <div style="font-size:0.85rem;">
            LINE転送などで画質が落ちた画像にも対応しています
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img_bgr = pil_to_bgr(img_pil)
st.image(img_pil, use_column_width=True, caption="アップロードされた画像")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Input method
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 2 　入力方法を選択")

input_method = st.radio(
    "入力方法",
    ["🤖 画像から自動認識（OCR）", "✏️ 手動で時間割を入力する"],
    label_visibility="collapsed",
    horizontal=False,
)
use_ocr = input_method.startswith("🤖")

if use_ocr and not tess_ok:
    st.warning("OCRエンジンが使えないため、手動入力モードに切り替えてください。")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# MANUAL INPUT MODE
# ══════════════════════════════════════════════════════════════════════════════

if not use_ocr:
    section("STEP 3 　時間割を入力する")
    st.caption("表のセルをクリックして授業名を入力してください")

    col_left, col_right = st.columns(2)
    with col_left:
        n_days = st.selectbox("曜日数", [5, 6, 7], index=0)
    with col_right:
        n_periods = st.selectbox("時限数", [4, 5, 6, 7, 8], index=2)

    day_labels_all = ["月", "火", "水", "木", "金", "土", "日"]
    day_labels = day_labels_all[:n_days]

    period_defaults = {
        1: ("08:50", "10:20"), 2: ("10:30", "12:00"),
        3: ("13:00", "14:30"), 4: ("14:40", "16:10"),
        5: ("16:20", "17:50"), 6: ("18:00", "19:30"),
        7: ("19:40", "21:10"), 8: ("21:20", "22:50"),
    }

    period_labels = []
    for i in range(1, n_periods + 1):
        d = period_defaults.get(i, ("09:00", "10:30"))
        period_labels.append(f"{i}限 {d[0]}〜{d[1]}")

    empty = {d: [""] * n_periods for d in day_labels}
    default_df = pd.DataFrame(empty, index=period_labels)

    manual_df = st.data_editor(
        default_df,
        use_container_width=True,
        num_rows="fixed",
        key="manual_input",
    )

    # Convert to time_slots / col_headers format
    manual_time_slots = []
    for i in range(1, n_periods + 1):
        d = period_defaults.get(i, ("09:00", "10:30"))
        sh, sm = map(int, d[0].split(":"))
        eh, em = map(int, d[1].split(":"))
        manual_time_slots.append({
            "period": i,
            "start_time": time(sh, sm),
            "end_time": time(eh, em),
            "raw": f"{i}限",
        })

    manual_col_headers = [
        {"type": "day",
         "day_of_week": ["月", "火", "水", "木", "金", "土", "日"].index(d),
         "text": d}
        for d in day_labels
    ]

    st.session_state["manual_df"] = manual_df
    st.session_state["manual_time_slots"] = manual_time_slots
    st.session_state["manual_col_headers"] = manual_col_headers
    st.session_state["input_ready"] = True
    st.session_state["input_mode"] = "manual"

# ══════════════════════════════════════════════════════════════════════════════
# OCR MODE
# ══════════════════════════════════════════════════════════════════════════════

else:
    section("STEP 3 　画像を解析する")

    auto_mode = auto_detect_mode(img_bgr)
    mode_map = {
        "grid":  "グリッド型（通常の大学の週次時間割）",
        "color": "カラーブロック型（看護・教育系の年間カレンダー）",
    }

    detected_label = mode_map[auto_mode]
    mode_choice = st.radio(
        "検出された時間割の種類",
        [f"自動（{detected_label}）", mode_map["grid"], mode_map["color"]],
        label_visibility="collapsed",
    )
    if mode_choice.startswith("自動"):
        active_mode = auto_mode
    elif "グリッド" in mode_choice:
        active_mode = "grid"
    else:
        active_mode = "color"

    if st.button("🔍 画像を解析する", type="primary"):
        st.session_state.pop("input_ready", None)

        # ── Grid mode ──────────────────────────────────────────────────────
        if active_mode == "grid":
            with st.spinner("表のセルを検出中..."):
                cells = detect_grid_cells(img_bgr)

            if not cells:
                st.error("セルを検出できませんでした。カラーブロック型に変更するか、手動入力をお試しください。")
                st.stop()

            grid = sort_to_grid(cells)
            if len(grid) < 2:
                st.error("行が少なすぎます。手動入力モードを試してください。")
                st.stop()

            with st.spinner(f"文字を認識中… ({sum(len(r) for r in grid)} セル)"):
                prog = st.progress(0)
                total = sum(len(r) for r in grid)
                done = 0
                grid_texts: list[list[str]] = []
                for row in grid:
                    row_t = []
                    for cell in row:
                        ci = extract_cell_img(img_bgr, cell)
                        row_t.append(ocr_cell(ci) if ci is not None else "")
                        done += 1
                        prog.progress(done / total)
                    grid_texts.append(row_t)
                prog.empty()

            layout = detect_layout_type(grid_texts)
            df, time_slots, col_headers = parse_grid_schedule(grid_texts)

            if df.empty:
                st.error("時間割を解析できませんでした。手動入力モードをお試しください。")
                st.stop()

            st.session_state.update({
                "input_ready": True,
                "input_mode": "grid",
                "layout": layout,
                "df": df,
                "time_slots": time_slots,
                "col_headers": col_headers,
            })

        # ── Color-block mode ────────────────────────────────────────────────
        else:
            with st.spinner("カラーブロックを検出中..."):
                blocks = detect_color_blocks(img_bgr)

            if not blocks:
                st.error("色付きブロックを検出できませんでした。手動入力モードをお試しください。")
                st.stop()

            with st.spinner(f"各ブロックの文字を認識中… ({len(blocks)} ブロック)"):
                prog = st.progress(0)
                enhanced = enhance_for_ocr(img_bgr, scale=3.0)
                h, w = img_bgr.shape[:2]
                for i, block in enumerate(blocks):
                    sx = enhanced.shape[1] / w
                    sy = enhanced.shape[0] / h
                    bx = int(block["x"] * sx); by = int(block["y"] * sy)
                    bw = int(block["w"] * sx); bh = int(block["h"] * sy)
                    ih2, iw2 = enhanced.shape[:2]
                    region = enhanced[
                        max(by, 0):min(by + bh, ih2),
                        max(bx, 0):min(bx + bw, iw2),
                    ]
                    block["text"] = ocr_large_region(region) if region.size > 0 else ""
                    prog.progress((i + 1) / len(blocks))
                prog.empty()

            st.session_state.update({
                "input_ready": True,
                "input_mode": "color",
                "blocks": blocks,
                "img_shape": img_bgr.shape,
            })

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Review / Edit
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("input_ready"):
    st.stop()

section("STEP 4 　内容を確認・修正する")
st.caption("セルをタップして直接編集できます。OCRのミスはここで直してください。")

mode = st.session_state.get("input_mode")

if mode == "manual":
    # Already shown above; just pass through
    df_edit = st.session_state["manual_df"]
    ts_edit = st.session_state["manual_time_slots"]
    ch_edit = st.session_state["manual_col_headers"]
    layout = "weekly"

elif mode == "grid":
    df = st.session_state["df"]
    layout = st.session_state["layout"]
    ts_edit = st.session_state["time_slots"]
    ch_edit = st.session_state["col_headers"]

    df_edit = st.data_editor(df, use_container_width=True, num_rows="fixed",
                              key="grid_edit")

elif mode == "color":
    blocks = st.session_state["blocks"]
    rows = [{"授業名": b.get("text", ""),
             "幅(週数の目安)": round(b["w"] / img_bgr.shape[1] * 52),
             "X位置": b["x"], "Y位置": b["y"],
             "W": b["w"], "H": b["h"]}
            for b in blocks]
    blk_df = pd.DataFrame(rows)
    blk_edited = st.data_editor(
        blk_df[["授業名", "幅(週数の目安)"]],
        use_container_width=True,
        key="color_edit",
    )
    for i, row in blk_edited.iterrows():
        blocks[i]["text"] = row["授業名"]
    st.session_state["blocks"] = blocks

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Calendar settings
# ══════════════════════════════════════════════════════════════════════════════

section("STEP 5 　カレンダーの設定")

ics_bytes: bytes | None = None
n_events = 0

if mode in ("manual", "grid"):
    if layout == "weekly":
        st.write("毎週繰り返す予定を登録します。学期の期間を入力してください。")
        col1, col2 = st.columns(2)
        with col1:
            sem_start = st.date_input("学期開始日", value=date.today(), key="ws")
        with col2:
            sem_end = st.date_input("学期終了日",
                                    value=date.today() + timedelta(weeks=16), key="we")

        if sem_start >= sem_end:
            st.warning("終了日は開始日より後に設定してください。")
        else:
            if st.button("カレンダーデータを作成する", type="primary", key="gen_w"):
                result = weekly_ics(
                    df_edit, ts_edit, ch_edit, sem_start, sem_end
                )
                ics_bytes, n_events = result
    else:
        st.write("日付指定の予定として登録します。")
        if st.button("カレンダーデータを作成する", type="primary", key="gen_v"):
            ev_list = df_variable_to_events(df_edit, ts_edit, ch_edit)
            ics_bytes, n_events = events_ics(ev_list)

elif mode == "color":
    st.write("ブロックの位置から日付を自動計算します。学年度の期間を入力してください。")
    col1, col2 = st.columns(2)
    with col1:
        sem_start = st.date_input("学年度開始日（例: 4/1）", value=date.today(), key="cs")
    with col2:
        yr = date.today().year
        sem_end = st.date_input("学年度終了日（例: 翌3/31）",
                                value=date(yr + 1, 3, 31), key="ce")
    periods = st.number_input("1日の時限数", 1, 10, 6, key="cp")

    if sem_start >= sem_end:
        st.warning("終了日は開始日より後に設定してください。")
    else:
        if st.button("カレンダーデータを作成する", type="primary", key="gen_c"):
            h, w = st.session_state["img_shape"][:2]
            ev_list = map_blocks_to_schedule(
                st.session_state["blocks"], w, h,
                sem_start, sem_end, int(periods)
            )
            ics_bytes, n_events = events_ics(ev_list)

if ics_bytes:
    st.session_state["ics_bytes"] = ics_bytes
    st.session_state["n_events"] = n_events

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Export
# ══════════════════════════════════════════════════════════════════════════════

if "ics_bytes" not in st.session_state:
    st.stop()

section("STEP 6 　カレンダーに登録する")

n = st.session_state["n_events"]
st.markdown(
    f'<div class="card" style="text-align:center;">'
    f'<div style="font-size:2.2rem;margin-bottom:0.4rem;">✅</div>'
    f'<div style="font-weight:700;font-size:1.1rem;color:#1C1C1E;">'
    f'{n} 件の予定を作成しました</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# HTML data-URI link (iOS Safari compatible)
st.markdown(
    ics_html_link(st.session_state["ics_bytes"], "timetable.ics"),
    unsafe_allow_html=True,
)

with st.expander("📌 カレンダーへの登録方法"):
    st.markdown("""
**iPhone（iOS標準カレンダー）**
1. 上の緑ボタンをタップ
2. 「"カレンダー"で開く」をタップ
3. 「全てのイベントを追加」→ 完了 ✅

**Googleカレンダー（スマホ）**
1. 上のボタンからファイルをダウンロード
2. Googleカレンダーアプリを開く
3. 設定 → インポート → ダウンロードしたファイルを選択

**TimeTree**
1. ファイルをダウンロード
2. TimeTree → カレンダー設定 → インポート
""")

st.divider()
if st.button("最初からやり直す", type="secondary"):
    st.session_state.clear()
    st.rerun()
