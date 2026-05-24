"""
スマート時間割インポーター
Streamlit web app: upload a timetable image → export .ics for any calendar app.

Supports:
  - Standard weekly timetables (Mon-Fri grid)
  - Complex yearly/monthly calendars (nursing, education: colored multi-span blocks)
  - Low-quality / LINE-compressed images
"""

import io
from datetime import date, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.image_processor import (
    auto_detect_mode,
    detect_color_blocks,
    detect_grid_cells,
    draw_cells_on_image,
    enhance_for_ocr,
    extract_cell_img,
    pil_to_bgr,
    sort_to_grid,
)
from core.ocr_engine import (
    is_tesseract_available,
    ocr_cell,
    ocr_cell_best,
    ocr_large_region,
)
from core.timetable_parser import (
    detect_layout_type,
    map_blocks_to_schedule,
    parse_grid_schedule,
    parse_time_slot,
)
from core.cal_generator import (
    df_variable_to_events,
    events_ics,
    weekly_ics,
)

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="スマート時間割",
    page_icon="📅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .stButton > button, .stDownloadButton > button {
      width: 100%; height: 3.2rem; font-size: 1.05rem; border-radius: 10px;
  }
  .step-chip {
      display: inline-block;
      background: #EFF6FF; color: #1D4ED8;
      padding: 0.25rem 0.75rem; border-radius: 20px;
      font-weight: 600; font-size: 0.9rem; margin: 1rem 0 0.4rem 0;
  }
</style>
""", unsafe_allow_html=True)

st.title("📅 スマート時間割")
st.caption("時間割の画像をアップロード → カレンダーアプリに自動登録")

# ── Tesseract check ───────────────────────────────────────────────────────────

if not is_tesseract_available():
    st.error(
        "**Tesseract OCRが見つかりません。**\n\n"
        "- Mac: `brew install tesseract tesseract-lang`\n"
        "- Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-jpn`"
    )
    st.stop()

# ── Helper: section header ────────────────────────────────────────────────────

def step(n: int, label: str):
    st.markdown(f'<div class="step-chip">STEP {n}　{label}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Upload
# ═══════════════════════════════════════════════════════════════════════════════

step(1, "時間割の画像を選択")

uploaded = st.file_uploader(
    "スクリーンショットまたは写真（JPG / PNG）",
    type=["jpg", "jpeg", "png"],
    help="LINE転送などで画質が落ちた画像にも対応します",
)

if not uploaded:
    st.info("↑ 上のエリアから画像をアップロードしてください")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img_bgr = pil_to_bgr(img_pil)
st.image(img_pil, caption="アップロードされた画像", use_column_width=True)

# ── Mode selector ─────────────────────────────────────────────────────────────

step(2, "時間割の種類を確認")

auto_mode = auto_detect_mode(img_bgr)
mode_labels = {
    "auto": f"自動検出（推奨: {'カラーブロック型' if auto_mode == 'color' else 'グリッド型'}）",
    "grid": "グリッド型（通常の大学の週次時間割）",
    "color": "カラーブロック型（看護・教育系の年間カレンダー）",
}
chosen_label = st.radio(
    "処理モードを選択してください",
    list(mode_labels.values()),
    index=0,
    help="自動検出で上手くいかない場合は手動で変更してください",
)
chosen_mode = list(mode_labels.keys())[list(mode_labels.values()).index(chosen_label)]
active_mode = auto_mode if chosen_mode == "auto" else chosen_mode

st.caption(f"▶ 選択中のモード: **{'カラーブロック型（看護・教育系）' if active_mode == 'color' else 'グリッド型（週次）'}**")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – Analyze
# ═══════════════════════════════════════════════════════════════════════════════

step(3, "時間割を解析")

if st.button("🔍 解析開始", type="primary"):
    st.session_state.clear()   # reset previous results

    # ── Grid mode ─────────────────────────────────────────────────────────────
    if active_mode == "grid":
        with st.spinner("表のセルを検出中..."):
            cells = detect_grid_cells(img_bgr)

        if not cells:
            st.error("セルを検出できませんでした。カラーブロック型に切り替えてみてください。")
            st.stop()

        grid = sort_to_grid(cells)
        if len(grid) < 2:
            st.error("行が少なすぎて時間割として認識できませんでした。")
            st.stop()

        st.success(f"表を検出しました（{len(grid)} 行 × {len(grid[0])} 列）")

        with st.expander("検出されたセルを確認（タップで展開）"):
            flat_cells = [c for row in grid for c in row]
            debug_img = draw_cells_on_image(img_bgr, flat_cells)
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        with st.spinner("文字を認識中... （処理に時間がかかります）"):
            prog = st.progress(0)
            total = sum(len(row) for row in grid)
            done = 0
            grid_texts: list[list[str]] = []
            for row in grid:
                row_texts = []
                for cell in row:
                    ci = extract_cell_img(img_bgr, cell)
                    row_texts.append(ocr_cell(ci) if ci is not None else "")
                    done += 1
                    prog.progress(done / total)
                grid_texts.append(row_texts)
            prog.empty()

        layout = detect_layout_type(grid_texts)
        df, time_slots, col_headers = parse_grid_schedule(grid_texts)

        if df.empty:
            st.error("時間割データを解析できませんでした。画像を確認してください。")
            st.stop()

        st.session_state.update({
            "mode": "grid",
            "layout": layout,
            "df": df,
            "time_slots": time_slots,
            "col_headers": col_headers,
        })
        if layout == "weekly":
            st.info("固定週次型（毎週繰り返し）として認識しました")
        else:
            st.info("日付指定型（変則・看護系）として認識しました")

    # ── Color-block mode ──────────────────────────────────────────────────────
    else:
        with st.spinner("カラーブロックを検出中..."):
            blocks = detect_color_blocks(img_bgr)

        if not blocks:
            st.error("色付きのブロックを検出できませんでした。グリッド型に切り替えてみてください。")
            st.stop()

        st.success(f"{len(blocks)} 個のブロックを検出しました")

        with st.expander("検出されたブロックを確認（タップで展開）"):
            debug_img = draw_cells_on_image(img_bgr, blocks)
            st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        with st.spinner("各ブロックの文字を認識中..."):
            prog = st.progress(0)
            enhanced = enhance_for_ocr(img_bgr, scale=3.0)
            h, w = img_bgr.shape[:2]
            for i, block in enumerate(blocks):
                # Scale block coordinates to enhanced image
                scale_x = enhanced.shape[1] / w
                scale_y = enhanced.shape[0] / h
                bx = int(block["x"] * scale_x)
                by = int(block["y"] * scale_y)
                bw = int(block["w"] * scale_x)
                bh = int(block["h"] * scale_y)
                ih, iw = enhanced.shape[:2]
                region = enhanced[
                    max(by, 0):min(by + bh, ih),
                    max(bx, 0):min(bx + bw, iw),
                ]
                block["text"] = ocr_large_region(region) if region.size > 0 else ""
                prog.progress((i + 1) / len(blocks))
            prog.empty()

        st.session_state.update({
            "mode": "color",
            "blocks": blocks,
            "img_shape": img_bgr.shape,
        })

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – Review & Edit
# ═══════════════════════════════════════════════════════════════════════════════

if "mode" not in st.session_state:
    st.stop()

step(4, "解析結果の確認・修正")
st.caption("OCRのミスがあれば直接セルをタップして修正できます")

if st.session_state["mode"] == "grid":
    edited_df = st.data_editor(
        st.session_state["df"],
        use_container_width=True,
        num_rows="fixed",
        key="grid_editor",
    )
    st.session_state["edited_df"] = edited_df

else:  # color mode
    blocks = st.session_state["blocks"]
    # Build a simple editable table: one row per block
    block_rows = [
        {"ブロック番号": i + 1,
         "授業名（OCR結果）": b.get("text", ""),
         "位置X": b["x"], "位置Y": b["y"],
         "幅": b["w"], "高さ": b["h"]}
        for i, b in enumerate(blocks)
    ]
    block_df = pd.DataFrame(block_rows).set_index("ブロック番号")
    edited_block_df = st.data_editor(
        block_df[["授業名（OCR結果）", "位置X", "位置Y", "幅", "高さ"]],
        use_container_width=True,
        key="color_editor",
    )
    # Write back edited names
    for i, row in edited_block_df.iterrows():
        blocks[int(i) - 1]["text"] = row["授業名（OCR結果）"]
    st.session_state["blocks"] = blocks

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – Calendar settings
# ═══════════════════════════════════════════════════════════════════════════════

step(5, "カレンダーの設定")

ics_bytes: bytes | None = None

if st.session_state["mode"] == "grid":
    layout = st.session_state["layout"]
    df = st.session_state["edited_df"]
    time_slots = st.session_state["time_slots"]
    col_headers = st.session_state["col_headers"]

    if layout == "weekly":
        st.write("学期の期間を入力してください（毎週繰り返しの予定が作られます）")
        col1, col2 = st.columns(2)
        with col1:
            sem_start = st.date_input("学期開始日", value=date.today(), key="ws")
        with col2:
            sem_end = st.date_input("学期終了日",
                                    value=date.today() + timedelta(weeks=15), key="we")
        if sem_start >= sem_end:
            st.warning("終了日は開始日より後に設定してください")
        else:
            if st.button("カレンダーデータを生成する", type="primary", key="gen_w"):
                with st.spinner("生成中..."):
                    ics_bytes = weekly_ics(df, time_slots, col_headers, sem_start, sem_end)
    else:
        st.write("日付が入った時間割として認識しました。そのまま登録します。")
        if st.button("カレンダーデータを生成する", type="primary", key="gen_v"):
            with st.spinner("生成中..."):
                ev_list = df_variable_to_events(df, time_slots, col_headers)
                ics_bytes = events_ics(ev_list)

else:  # color mode
    st.write("各ブロックの日付範囲を設定します（カラーブロックのX位置で自動計算）")
    col1, col2 = st.columns(2)
    with col1:
        sem_start = st.date_input("学年度開始日（例: 4/1）", value=date.today(), key="cs")
    with col2:
        sem_end = st.date_input("学年度終了日（例: 翌3/31）",
                                value=date(date.today().year + 1, 3, 31), key="ce")
    periods = st.number_input("1日の時限数", min_value=1, max_value=10, value=6, key="cp")

    if sem_start >= sem_end:
        st.warning("終了日は開始日より後に設定してください")
    else:
        if st.button("カレンダーデータを生成する", type="primary", key="gen_c"):
            with st.spinner("生成中..."):
                h, w = st.session_state["img_shape"][:2]
                ev_list = map_blocks_to_schedule(
                    st.session_state["blocks"], w, h,
                    sem_start, sem_end, int(periods)
                )
                ics_bytes = events_ics(ev_list)

if ics_bytes:
    st.session_state["ics_bytes"] = ics_bytes

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 – Download
# ═══════════════════════════════════════════════════════════════════════════════

if "ics_bytes" in st.session_state:
    step(6, "カレンダーアプリに登録")
    st.success("カレンダーデータの生成が完了しました！")

    st.download_button(
        label="📲 カレンダーに登録（.ics ダウンロード）",
        data=st.session_state["ics_bytes"],
        file_name="timetable.ics",
        mime="text/calendar",
        type="primary",
    )

    with st.expander("カレンダーアプリへの登録方法"):
        st.markdown("""
**iPhone / iPad（iOS標準カレンダー）**
1. 上のボタンをタップしてダウンロード
2. ダウンロードしたファイルをタップ
3. カレンダーアプリが自動起動 → 「全てのイベントを追加」

**Googleカレンダー**
- PCから: [calendar.google.com](https://calendar.google.com) → 設定 → インポート
- スマホ: ダウンロードファイルを「Googleカレンダー」アプリで共有

**TimeTree / その他のアプリ**
- 各アプリの「インポート」機能から .ics ファイルを読み込む
        """)

    st.divider()
    if st.button("最初からやり直す"):
        st.session_state.clear()
        st.rerun()
