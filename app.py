"""
EasyJikanwari — 時間割をカレンダーアプリに自動登録
スマホ完結 / 完全無料 / OCR不要
"""

import base64
from datetime import date, time, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.image_processor import (
    auto_detect_mode,
    detect_color_blocks,
    draw_numbered_blocks,
    detect_grid_cells,
    extract_cell_img,
    pil_to_bgr,
    sort_to_grid,
)
from core.ocr_engine import is_tesseract_available, ocr_cell
from core.timetable_parser import (
    _DEFAULT_PERIOD_TIMES,
    detect_layout_type,
    map_blocks_to_schedule,
    parse_grid_schedule,
)
from core.cal_generator import df_variable_to_events, events_ics, weekly_ics

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EasyJikanwari",
    page_icon="📅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #F2F2F7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stHeader"], [data-testid="stDecoration"] { display: none !important; }

.card {
    background: #fff;
    border-radius: 16px;
    padding: 1.2rem 1.2rem 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}
.sec {
    font-size: 0.7rem;
    font-weight: 700;
    color: #8E8E93;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    margin: 1.3rem 0 0.25rem 0.2rem;
}
.stButton > button {
    width: 100%; height: 3.2rem;
    font-size: 1rem; font-weight: 700;
    border-radius: 12px; border: none;
}
.stButton > button[kind="primary"]   { background:#007AFF!important; color:#fff!important; }
.stButton > button[kind="secondary"] { background:#E5E5EA!important; color:#1C1C1E!important; }
.stDownloadButton > button {
    width: 100%; height: 3.2rem;
    font-size: 1rem; font-weight: 700;
    border-radius: 12px; border: none;
    background: #34C759 !important; color: #fff !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #C7C7CC; border-radius: 14px;
    padding: 0.5rem; background: #fff;
}
.export-btn {
    display: block; text-align: center;
    background: #34C759; color: #fff !important;
    padding: 1rem; border-radius: 14px;
    text-decoration: none !important;
    font-size: 1.05rem; font-weight: 700;
    margin: 0.4rem 0;
    box-shadow: 0 3px 10px rgba(52,199,89,0.3);
}
.ok-box {
    background:#fff; border-radius:16px;
    padding:1.5rem 1rem; text-align:center;
    box-shadow:0 1px 6px rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def sec(t): st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def ics_link(ics_bytes: bytes, filename: str = "timetable.ics") -> str:
    b64 = base64.b64encode(ics_bytes).decode()
    return (
        f'<a class="export-btn" '
        f'href="data:text/calendar;charset=utf-8;base64,{b64}" '
        f'download="{filename}">📲&nbsp; カレンダーアプリに登録する</a>'
    )

def period_label(i: int, pt: dict) -> str:
    ts = pt.get(i)
    if ts:
        return f"{i}限　{ts[0].strftime('%H:%M')}〜{ts[1].strftime('%H:%M')}"
    return f"{i}限"


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("## 📅 EasyJikanwari")
st.caption("時間割 → カレンダーアプリに登録")

# ══════════════════════════════════════════════════════════════════════════════
# モード選択
# ══════════════════════════════════════════════════════════════════════════════

sec("STEP 1　時間割の種類を選ぶ")

mode = st.radio(
    "種類",
    ["📅 週次時間割（毎週同じ）",
     "🗓️ 年間・日程カレンダー（看護・教育系など）"],
    label_visibility="collapsed",
)
is_yearly = mode.startswith("🗓️")

# ══════════════════════════════════════════════════════════════════════════════
# ── 週次モード ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if not is_yearly:
    sec("STEP 2　時間割を入力する")
    st.caption("授業名をセルに直接入力してください")

    c1, c2 = st.columns(2)
    with c1:
        n_days = st.selectbox("曜日数", [5, 6, 7], index=0, key="nd")
    with c2:
        n_per  = st.selectbox("時限数", [4, 5, 6, 7, 8], index=2, key="np")

    # Custom period times
    with st.expander("⏰ 時限の時間を変更する（任意）"):
        custom_pt: dict[int, tuple[time, time]] = {}
        defaults = {
            1:("08:50","10:20"), 2:("10:30","12:00"), 3:("13:00","14:30"),
            4:("14:40","16:10"), 5:("16:20","17:50"), 6:("18:00","19:30"),
            7:("19:40","21:10"), 8:("21:20","22:50"),
        }
        for i in range(1, n_per + 1):
            d = defaults.get(i, ("09:00","10:30"))
            cc1, cc2, cc3 = st.columns([1, 2, 2])
            with cc1: st.markdown(f"**{i}限**")
            with cc2: s = st.text_input(f"開始{i}", d[0], label_visibility="collapsed", key=f"s{i}")
            with cc3: e = st.text_input(f"終了{i}", d[1], label_visibility="collapsed", key=f"e{i}")
            try:
                sh, sm = map(int, s.split(":"))
                eh, em = map(int, e.split(":"))
                custom_pt[i] = (time(sh, sm), time(eh, em))
            except Exception:
                custom_pt[i] = _DEFAULT_PERIOD_TIMES.get(i, (time(9,0), time(10,30)))

    day_names = ["月","火","水","木","金","土","日"][:n_days]
    prow_labels = [period_label(i, custom_pt or _DEFAULT_PERIOD_TIMES)
                   for i in range(1, n_per + 1)]
    empty = pd.DataFrame(
        [[""] * n_days for _ in range(n_per)],
        index=prow_labels, columns=day_names
    )
    df_in = st.data_editor(empty, use_container_width=True, num_rows="fixed", key="week_grid")

    sec("STEP 3　学期の期間を設定する")
    c1, c2 = st.columns(2)
    with c1: sem_s = st.date_input("開始日", date.today(), key="wss")
    with c2: sem_e = st.date_input("終了日", date.today()+timedelta(weeks=16), key="wse")

    if sem_s >= sem_e:
        st.warning("終了日は開始日より後にしてください")
        st.stop()

    if st.button("カレンダーデータを作成する", type="primary", key="wgen"):
        pt_final = custom_pt or _DEFAULT_PERIOD_TIMES
        ts_list = [{"period": i,
                    "start_time": pt_final.get(i, (time(9,0), time(10,30)))[0],
                    "end_time":   pt_final.get(i, (time(9,0), time(10,30)))[1],
                    "raw": f"{i}限"}
                   for i in range(1, n_per + 1)]
        ch_list = [{"type":"day",
                    "day_of_week": ["月","火","水","木","金","土","日"].index(d),
                    "text": d}
                   for d in day_names]
        ics_bytes, n_ev = weekly_ics(df_in, ts_list, ch_list, sem_s, sem_e)
        st.session_state["ics_bytes"] = ics_bytes
        st.session_state["n_ev"] = n_ev

# ══════════════════════════════════════════════════════════════════════════════
# ── 年間モード ─────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

else:
    sec("STEP 2　時間割の画像をアップロード")
    uploaded = st.file_uploader(
        "スクリーンショットまたは写真",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed",
        key="yearly_img",
    )

    if not uploaded:
        st.markdown("""
        <div class="card" style="text-align:center;padding:1.8rem 1rem;color:#8E8E93;">
          <div style="font-size:2.2rem;margin-bottom:0.4rem;">📷</div>
          <div style="font-weight:600;color:#3A3A3C;margin-bottom:0.3rem;">画像を選んでください</div>
          <div style="font-size:0.83rem;">LINE転送などで画質が落ちた画像でも利用できます</div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    img_pil = Image.open(uploaded).convert("RGB")
    img_bgr = pil_to_bgr(img_pil)

    sec("STEP 3　学年度の期間と時限数を設定する")
    c1, c2, c3 = st.columns(3)
    with c1: yr_s = st.number_input("開始年", 2024, 2030, date.today().year, key="yrs")
    with c2: mo_s = st.number_input("開始月", 1, 12, 4, key="yms")
    with c3: n_per2 = st.number_input("1日の時限数", 1, 10, 6, key="ynp")
    try:
        ay_start = date(int(yr_s), int(mo_s), 1)
        mo_e = mo_s - 1 if mo_s > 1 else 12
        yr_e = int(yr_s) + 1 if mo_s > 1 else int(yr_s)
        # End of the last month of the academic year
        import calendar as cal_mod
        last_day = cal_mod.monthrange(yr_e, mo_e)[1]
        ay_end = date(yr_e, mo_e, last_day)
    except Exception:
        st.error("期間の設定が正しくありません")
        st.stop()

    st.caption(f"学年度: {ay_start.strftime('%Y/%m/%d')} 〜 {ay_end.strftime('%Y/%m/%d')}")

    if st.button("🔍 カラーブロックを検出する", type="primary", key="ydet"):
        with st.spinner("画像からブロックを検出中..."):
            blocks = detect_color_blocks(img_bgr)
        if not blocks:
            st.error("カラーブロックを検出できませんでした。画像を確認してください。")
            st.stop()
        st.session_state["blocks"] = blocks
        st.session_state["img_bgr"] = img_bgr
        st.session_state["img_pil"] = img_pil

    if "blocks" not in st.session_state:
        st.stop()

    blocks = st.session_state["blocks"]
    img_bgr_saved = st.session_state["img_bgr"]

    sec(f"STEP 4　ブロックを確認して授業名を入力する（{len(blocks)} 件検出）")

    numbered_img = draw_numbered_blocks(img_bgr_saved, blocks)
    st.image(cv2.cvtColor(numbered_img, cv2.COLOR_BGR2RGB),
             caption="検出されたブロック（番号付き）",
             use_column_width=True)

    st.caption("上の画像の番号を参照しながら、各ブロックの授業名を入力してください。不要な行は空欄のままでOKです。")

    # Compute estimated date ranges for display
    min_x = min(b["x"] for b in blocks)
    max_x = max(b["x"] + b["w"] for b in blocks)
    x_range = max(max_x - min_x, 1)
    total_days = max((ay_end - ay_start).days, 1)

    rows = []
    for i, b in enumerate(blocks):
        xs = (b["x"] - min_x) / x_range
        xe = (b["x"] + b["w"] - min_x) / x_range
        d_s = ay_start + timedelta(days=int(xs * total_days))
        d_e = ay_start + timedelta(days=int(xe * total_days))
        rows.append({
            "No": i + 1,
            "授業名": b.get("text", ""),
            "開始日(目安)": d_s.strftime("%m/%d"),
            "終了日(目安)": d_e.strftime("%m/%d"),
        })

    edit_df = pd.DataFrame(rows).set_index("No")
    edited = st.data_editor(
        edit_df,
        use_container_width=True,
        disabled=["開始日(目安)", "終了日(目安)"],
        key="yearly_edit",
    )

    # Write back names
    for idx, row in edited.iterrows():
        blocks[int(idx) - 1]["text"] = row["授業名"]

    if st.button("カレンダーデータを作成する", type="primary", key="ygen"):
        ev_list = map_blocks_to_schedule(blocks, ay_start, ay_end, int(n_per2))
        if not ev_list:
            st.warning("授業名が入力されているブロックがありません。入力してから再度お試しください。")
            st.stop()
        ics_bytes, n_ev = events_ics(ev_list)
        st.session_state["ics_bytes"] = ics_bytes
        st.session_state["n_ev"] = n_ev

# ══════════════════════════════════════════════════════════════════════════════
# Export (shared)
# ══════════════════════════════════════════════════════════════════════════════

if "ics_bytes" not in st.session_state:
    st.stop()

sec("STEP 5　カレンダーに登録する")

n = st.session_state["n_ev"]
st.markdown(
    f'<div class="ok-box">'
    f'<div style="font-size:2rem;margin-bottom:0.3rem;">✅</div>'
    f'<div style="font-weight:700;font-size:1.05rem;color:#1C1C1E;">'
    f'{n} 件の予定を作成しました</div></div>',
    unsafe_allow_html=True,
)

st.markdown(ics_link(st.session_state["ics_bytes"]), unsafe_allow_html=True)

with st.expander("📌 登録方法（iPhone / Google / TimeTree）"):
    st.markdown("""
**iPhone（iOS標準カレンダー）**
1. 緑のボタンをタップ
2. 「"カレンダー"で開く」をタップ
3. 「全てのイベントを追加」→ 完了 ✅

**Googleカレンダー**
- PC: [calendar.google.com](https://calendar.google.com) → 設定 → インポート
- スマホ: ダウンロード後、Googleカレンダーで共有

**TimeTree / その他**
- 各アプリの「インポート」機能から .ics を読み込む
""")

st.divider()
if st.button("最初からやり直す", type="secondary"):
    st.session_state.clear()
    st.rerun()
