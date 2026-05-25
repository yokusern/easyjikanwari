"""
EasyJikanwari
看護・教育系の年間時間割画像 → カレンダーアプリに一括登録
"""

import base64
import calendar as cal_mod
from datetime import date, time, timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.image_processor import (
    detect_color_blocks,
    draw_group_preview,
    group_blocks_by_color,
    pil_to_bgr,
)
from core.ocr_engine import assign_ocr_to_groups
from core.timetable_parser import DEFAULT_PERIOD_TIMES, groups_to_events
from core.cal_generator import events_ics

# ── Page config ────────────────────────────────────────────────────────────────

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
[data-testid="stHeader"],[data-testid="stDecoration"]{ display:none!important; }

.card {
    background:#fff; border-radius:16px;
    padding:1.1rem 1.2rem; margin:0.45rem 0;
    box-shadow:0 1px 6px rgba(0,0,0,0.08);
}
.sec {
    font-size:.7rem; font-weight:700; color:#8E8E93;
    letter-spacing:.07em; text-transform:uppercase;
    margin:1.3rem 0 .3rem .2rem;
}
.stButton>button {
    width:100%; height:3.2rem;
    font-size:1rem; font-weight:700;
    border-radius:12px; border:none;
}
.stButton>button[kind="primary"]  { background:#007AFF!important;color:#fff!important; }
.stButton>button[kind="secondary"]{ background:#E5E5EA!important;color:#1C1C1E!important; }
[data-testid="stFileUploader"]{
    border:2px dashed #C7C7CC; border-radius:14px;
    padding:.4rem; background:#fff;
}
.export-btn {
    display:block; text-align:center;
    background:#34C759; color:#fff!important;
    padding:1rem; border-radius:14px;
    text-decoration:none!important;
    font-size:1.05rem; font-weight:700;
    margin:.4rem 0; box-shadow:0 3px 10px rgba(52,199,89,.3);
}
.ocr-badge {
    display:inline-block;
    background:#FF9500; color:#fff;
    font-size:.65rem; font-weight:700;
    padding:.1rem .4rem; border-radius:4px;
    margin-left:.3rem; vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def sec(t: str):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def ics_link(ics_bytes: bytes) -> str:
    b64 = base64.b64encode(ics_bytes).decode()
    return (
        f'<a class="export-btn" '
        f'href="data:text/calendar;charset=utf-8;base64,{b64}" '
        f'download="timetable.ics">📲&nbsp; カレンダーアプリに登録する</a>'
    )

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("## 📅 EasyJikanwari")
st.caption("看護・教育系の年間時間割 → iOSカレンダー / Googleカレンダーに一括登録")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — 画像アップロード
# ══════════════════════════════════════════════════════════════════════════════

sec("STEP 1　時間割の画像を選ぶ")

uploaded = st.file_uploader(
    "年間時間割・日程表の画像（JPG / PNG）",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if not uploaded:
    st.markdown("""
    <div class="card" style="text-align:center;padding:2rem 1rem;color:#8E8E93;">
      <div style="font-size:2.5rem;margin-bottom:.5rem;">📷</div>
      <div style="font-weight:600;color:#3A3A3C;margin-bottom:.3rem;">
        時間割の画像をアップロードしてください
      </div>
      <div style="font-size:.85rem;">
        スクリーンショット・写真どちらも対応。LINE転送画像もOK。
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img_bgr = pil_to_bgr(img_pil)
st.image(img_pil, use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 学年度設定
# ══════════════════════════════════════════════════════════════════════════════

sec("STEP 2　学年度・時限の設定")

c1, c2, c3 = st.columns(3)
with c1: yr_s = st.number_input("開始年", 2024, 2030, date.today().year, key="yrs")
with c2: mo_s = st.number_input("開始月", 1, 12, 4, key="yms")
with c3: n_per = st.number_input("1日の時限数", 1, 10, 6, key="ynp")

mo_e = int(mo_s) - 1 if int(mo_s) > 1 else 12
yr_e = int(yr_s) + 1 if int(mo_s) > 1 else int(yr_s)
ay_start = date(int(yr_s), int(mo_s), 1)
ay_end   = date(yr_e, mo_e, cal_mod.monthrange(yr_e, mo_e)[1])
st.caption(f"学年度: **{ay_start.strftime('%Y年%m月')}** 〜 **{ay_end.strftime('%Y年%m月')}**")

with st.expander("⏰ 時限の時間帯を変更する（任意）"):
    custom_pt: dict[int, tuple[time, time]] = {}
    defaults_str = {
        1:("08:50","10:20"), 2:("10:30","12:00"), 3:("13:00","14:30"),
        4:("14:40","16:10"), 5:("16:20","17:50"), 6:("18:00","19:30"),
        7:("19:40","21:10"), 8:("21:20","22:50"),
    }
    for i in range(1, int(n_per) + 1):
        d = defaults_str.get(i, ("09:00","10:30"))
        cc1, cc2, cc3 = st.columns([1, 2, 2])
        with cc1: st.markdown(f"**{i}限**")
        with cc2: s_str = st.text_input(f"s{i}", d[0], label_visibility="collapsed", key=f"ts{i}")
        with cc3: e_str = st.text_input(f"e{i}", d[1], label_visibility="collapsed", key=f"te{i}")
        try:
            sh, sm = map(int, s_str.split(":"))
            eh, em = map(int, e_str.split(":"))
            custom_pt[i] = (time(sh, sm), time(eh, em))
        except Exception:
            custom_pt[i] = DEFAULT_PERIOD_TIMES.get(i, (time(9, 0), time(10, 30)))

period_times = custom_pt if custom_pt else DEFAULT_PERIOD_TIMES

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 検出 + OCR（まとめて実行）
# ══════════════════════════════════════════════════════════════════════════════

sec("STEP 3　授業ブロックを検出する")

if st.button("🔍 ブロック検出", type="primary", key="detect"):
    with st.spinner("授業ブロックを検出中…"):
        blocks = detect_color_blocks(img_bgr)
        groups = group_blocks_by_color(blocks)

    if not groups:
        st.error("カラーブロックを検出できませんでした。別の画像を試してください。")
        st.stop()

    groups = assign_ocr_to_groups([], groups)

    st.session_state["groups"]  = groups
    st.session_state["img_bgr"] = img_bgr

if "groups" not in st.session_state:
    st.stop()

groups: list[dict] = st.session_state["groups"]
img_bgr_ref        = st.session_state["img_bgr"]

total_blocks = sum(len(g["blocks"]) for g in groups)
st.success(f"{len(groups)} 種類の色・{total_blocks} ブロックを検出しました")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — 授業名の確認・修正
# ══════════════════════════════════════════════════════════════════════════════

sec("STEP 4　授業名を確認・修正する")

with st.expander("🔍 検出プレビュー（数字 = グループ番号）", expanded=True):
    preview_img = draw_group_preview(img_bgr_ref, groups)
    st.image(cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB), use_column_width=True)

st.caption("各色に授業名を入力してください。休日・空きはスキップ。")

# 2列レイアウトでグループを表示
cols_left, cols_right = st.columns(2)

for gi, group in enumerate(groups):
    rgb   = group["rgb_css"]
    count = len(group["blocks"])
    col   = cols_left if gi % 2 == 0 else cols_right

    with col:
        swatch_label = (
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">'
            f'<div style="width:18px;height:18px;background:{rgb};'
            f'border-radius:4px;border:1px solid rgba(0,0,0,.15);flex-shrink:0;"></div>'
            f'<span style="font-size:.72rem;color:#8E8E93;">#{gi+1} · {count}ブロック</span>'
            f'</div>'
        )
        st.markdown(swatch_label, unsafe_allow_html=True)
        name = st.text_input(
            f"g{gi}",
            value=group.get("name", ""),
            placeholder="授業名",
            label_visibility="collapsed",
            key=f"name_{gi}",
        )
        group["name"] = name
        skip = st.checkbox("スキップ", value=group.get("skip", False), key=f"skip_{gi}")
        group["skip"] = skip

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — カレンダー生成 & エクスポート
# ══════════════════════════════════════════════════════════════════════════════

sec("STEP 5　カレンダーに登録する")

labeled = sum(1 for g in groups if g.get("name", "").strip() and not g.get("skip"))
if labeled == 0:
    st.info("STEP 4 で少なくとも1つの授業名を入力してください")
    st.stop()

st.caption(f"{labeled} 種類の授業が入力済みです")

if st.button("📅 カレンダーデータを作成する", type="primary", key="gen"):
    ev_list = groups_to_events(groups, ay_start, ay_end, int(n_per), period_times)
    if not ev_list:
        st.warning("予定を作成できませんでした。授業名の入力を確認してください。")
        st.stop()
    ics_bytes, n_ev = events_ics(ev_list)
    st.session_state["ics_bytes"] = ics_bytes
    st.session_state["n_ev"]      = n_ev
    st.session_state["ev_list"]   = ev_list

if "ics_bytes" not in st.session_state:
    st.stop()

n_ev    = st.session_state["n_ev"]
ev_list = st.session_state["ev_list"]

st.markdown(
    f'<div class="card" style="text-align:center;">'
    f'<div style="font-size:2rem;margin-bottom:.3rem;">✅</div>'
    f'<div style="font-weight:700;font-size:1.05rem;">{n_ev} 件の予定を作成しました</div>'
    f'</div>',
    unsafe_allow_html=True,
)

with st.expander(f"📋 作成される予定の一覧（最初の20件）"):
    rows = [{
        "授業名":  e["summary"],
        "開始日":  e["start_date"].strftime("%Y/%m/%d"),
        "終了日":  e["end_date"].strftime("%Y/%m/%d"),
        "開始時刻": e["start_time"].strftime("%H:%M"),
        "終了時刻": e["end_time"].strftime("%H:%M"),
    } for e in ev_list[:20]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if n_ev > 20:
        st.caption(f"… 他 {n_ev - 20} 件")

st.markdown(ics_link(st.session_state["ics_bytes"]), unsafe_allow_html=True)

with st.expander("📌 カレンダーへの登録方法"):
    st.markdown("""
**iPhone（iOS標準カレンダー）**
1. 緑のボタンをタップ
2. 「"カレンダー"で開く」をタップ
3. 「全てのイベントを追加」→ 完了 ✅

**Googleカレンダー**
- PC: [calendar.google.com](https://calendar.google.com) → 設定 → インポート

**TimeTree・その他**
- 各アプリの「インポート」機能から `.ics` ファイルを読み込む
""")

st.divider()
if st.button("最初からやり直す", type="secondary"):
    st.session_state.clear()
    st.rerun()
