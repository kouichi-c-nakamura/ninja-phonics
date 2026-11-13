"""
app.py: NinjaPhonetics Biofeedback Scope
"""
import os
import re
import soundfile as sf
import streamlit as st
import streamlit.components.v1 as components

from constants import PRESETS, PRESET_DIR, E2_PASSAGE_BENCHMARKS
from ui_styles import SHURIKEN_ICON_MAIN, SHURIKEN_ICON_SM, APP_CUSTOM_CSS
from audio_processing import (
    decode_audio_bytes,
    extract_scope_data_from_bytes,
    evaluate_user_rhythm,
    get_benchmark_word_raw_distributions
)
from views_a1 import build_spike2_html
from views_e2 import build_e2_multiview_canvas_html, build_clickable_word_map_html

st.set_page_config(page_title="NinjaPhonetics: Biofeedback Scope", layout="wide", initial_sidebar_state="collapsed")
st.markdown(APP_CUSTOM_CSS, unsafe_allow_html=True)

# セッション状態初期化
if "app_mode" not in st.session_state: st.session_state.app_mode = "RL"
if "current_label" not in st.session_state: st.session_state.current_label = "お手本: Right (US)"
if "current_audio_bytes" not in st.session_state:
    init_path = os.path.join("presets", "right_us.wav")
    st.session_state.current_audio_bytes = open(init_path, "rb").read() if os.path.exists(init_path) else None
if "last_recorded_bytes" not in st.session_state: st.session_state.last_recorded_bytes = None
if "auto_play" not in st.session_state: st.session_state.auto_play = False

if "e2_category" not in st.session_state: st.session_state.e2_category = "passage"
if "e2_selected_passage" not in st.session_state: st.session_state.e2_selected_passage = "P1_northwind"
if "e2_custom_text" not in st.session_state:
    st.session_state.e2_custom_text = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
if "e2_last_analyzed_text" not in st.session_state:
    st.session_state.e2_last_analyzed_text = st.session_state.e2_custom_text
if "e2_audio_bytes" not in st.session_state: st.session_state.e2_audio_bytes = None
if "e2_eval_history" not in st.session_state: st.session_state.e2_eval_history = []
if "e2_plot_view" not in st.session_state: st.session_state.e2_plot_view = "scatter2d"

# ブランドヘッダー
st.markdown(
    f'<div class="brand-bar">{SHURIKEN_ICON_MAIN}'
    f'<span class="brand-title">NinjaPhonetics: Biofeedback Scope</span>'
    f'<span class="brand-sub">ACOUSTIC & PROSODY DIAGNOSTICS</span></div>',
    unsafe_allow_html=True
)

tab_m1, tab_m2, _ = st.columns([1.5, 1.5, 3.0])
with tab_m1:
    if st.button("A1: 音素ホルマント・スコープ (R / L)", type="primary" if st.session_state.app_mode == "RL" else "secondary", use_container_width=True):
        st.session_state.app_mode = "RL"; st.rerun()
with tab_m2:
    if st.button("E2: プロソディ＆ストレス拍診断", type="primary" if st.session_state.app_mode == "RHYTHM" else "secondary", use_container_width=True):
        st.session_state.app_mode = "RHYTHM"; st.rerun()

# -------------------------------------------------------------
# A1 モード
# -------------------------------------------------------------
if st.session_state.app_mode == "RL":
    cols = st.columns(len(PRESETS))
    preset_clicked = False
    active_preset_info = None

    for idx, col in enumerate(cols):
        p_info = PRESETS[idx]
        is_active = (st.session_state.current_label == f"お手本: {p_info['label']}")
        if is_active:
            active_preset_info = p_info

        if col.button(p_info["label"], type="primary" if is_active else "secondary", use_container_width=True):
            fpath = os.path.join(PRESET_DIR, p_info["file"])
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    st.session_state.current_audio_bytes = f.read()
                    st.session_state.current_label = f"お手本: {p_info['label']}"
                    st.session_state.auto_play = True
                preset_clicked = True
                st.rerun()

    audio_input = st.audio_input("または自分の声を録音して解析 (R / L)")
    if not preset_clicked and audio_input is not None:
        new_bytes = audio_input.read()
        if new_bytes != st.session_state.last_recorded_bytes:
            st.session_state.last_recorded_bytes = new_bytes
            st.session_state.current_audio_bytes = new_bytes
            st.session_state.current_label = "あなたの発音録音"
            st.session_state.auto_play = False

    if st.session_state.current_audio_bytes is not None:
        if st.session_state.current_label:
            st.markdown(f"<div class='sec-header'>{SHURIKEN_ICON_SM} 表示中: <span style='color: #F8FAFC;'>{st.session_state.current_label}</span></div>", unsafe_allow_html=True)

        clean_name = re.sub(r'[^a-zA-Z0-9_\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', '_', st.session_state.current_label).strip('_')
        scope_data = extract_scope_data_from_bytes(st.session_state.current_audio_bytes)
        html_code = build_spike2_html(scope_data, auto_play=st.session_state.auto_play, export_name=clean_name or "sample")
        st.session_state.auto_play = False
        components.html(html_code, height=590, scrolling=False)

        # 選択中プリセットの解説カード表示
        if active_preset_info and active_preset_info.get("desc"):
            st.markdown(f"""
            <div style="background: #0F172A; border: 1px solid #1E293B; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 12px 18px; margin-top: 6px;">
              <div style="font-size: 0.82rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; margin-bottom: 2px;">
                {active_preset_info.get('phonetics', '')}
              </div>
              <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 4px;">
                {active_preset_info.get('title', '')}
              </div>
              <div style="font-size: 0.88rem; color: #94A3B8; line-height: 1.5;">
                {active_preset_info.get('desc', '')}
              </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# E2 モード
# -------------------------------------------------------------
else:
    sub_t1, sub_t2, _ = st.columns([1.4, 1.4, 3.2])
    with sub_t1:
        if st.button("短文フォーカス・ドリル (S1〜S3)", type="primary" if st.session_state.e2_category == "drill" else "secondary", use_container_width=True):
            if st.session_state.e2_category != "drill":
                st.session_state.e2_category = "drill"
                st.session_state.e2_selected_passage = "S1_stress"
                t_txt = E2_PASSAGE_BENCHMARKS["S1_stress"]["target_text"]
                st.session_state.e2_custom_text = st.session_state.e2_last_analyzed_text = st.session_state["e2_text_input_widget"] = t_txt
                st.session_state.e2_eval_history, st.session_state.e2_audio_bytes = [], None
                st.rerun()
    with sub_t2:
        if st.button("総合アセスメント (長文・標準規格)", type="primary" if st.session_state.e2_category == "passage" else "secondary", use_container_width=True):
            if st.session_state.e2_category != "passage":
                st.session_state.e2_category = "passage"
                st.session_state.e2_selected_passage = "P1_northwind"
                t_txt = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
                st.session_state.e2_custom_text = st.session_state.e2_last_analyzed_text = st.session_state["e2_text_input_widget"] = t_txt
                st.session_state.e2_eval_history, st.session_state.e2_audio_bytes = [], None
                st.rerun()

    if st.session_state.e2_category == "drill":
        d_col1, d_col2, d_col3, _ = st.columns([1.2, 1.2, 1.2, 2.4])
        for col, key, lbl in [(d_col1, "S1_stress", "S1: 語彙強弱拍"), (d_col2, "S2_clusters", "S2: 子音連結保持"), (d_col3, "S3_connected", "S3: 機能語弱化")]:
            with col:
                if st.button(lbl, type="primary" if st.session_state.e2_selected_passage == key else "secondary", use_container_width=True):
                    st.session_state.e2_selected_passage = key
                    t_txt = E2_PASSAGE_BENCHMARKS[key]["target_text"]
                    st.session_state.e2_custom_text = st.session_state.e2_last_analyzed_text = st.session_state["e2_text_input_widget"] = t_txt
                    st.session_state.e2_eval_history = []
                    st.rerun()
    else:
        p_col1, p_col2, p_col3, _ = st.columns([1.3, 1.3, 1.3, 2.1])
        with p_col1:
            if st.button("P1: 北風と太陽 (113語)", type="primary" if st.session_state.e2_selected_passage == "P1_northwind" else "secondary", use_container_width=True):
                st.session_state.e2_selected_passage = "P1_northwind"
                t_txt = E2_PASSAGE_BENCHMARKS["P1_northwind"]["target_text"]
                st.session_state.e2_custom_text = st.session_state.e2_last_analyzed_text = st.session_state["e2_text_input_widget"] = t_txt
                st.session_state.e2_eval_history = []
                st.rerun()
        with p_col2: st.button("P2: 虹のパッセージ", disabled=True, use_container_width=True)
        with p_col3: st.button("P3: 祖父のパッセージ", disabled=True, use_container_width=True)

    active_bm = E2_PASSAGE_BENCHMARKS[st.session_state.e2_selected_passage]

    txt_col1, txt_col2 = st.columns([4.8, 1.0], vertical_alignment="bottom")
    with txt_col1:
        # 高さを従来の2倍に拡大 (長文: 170px, 短文: 110px)
        th = 170 if st.session_state.e2_category == "passage" else 110
        user_target_text = st.text_area(
            "課題テキスト (自由に書き換え可能)", 
            value=st.session_state.e2_custom_text, 
            height=th, 
            key="e2_text_input_widget"
        )
    with txt_col2:
        if st.button("例文に戻す", use_container_width=True):
            t_txt = active_bm["target_text"]
            st.session_state.e2_custom_text = st.session_state.e2_last_analyzed_text = st.session_state["e2_text_input_widget"] = t_txt
            st.session_state.e2_eval_history, st.session_state.e2_audio_bytes = [], None
            st.rerun()

    if user_target_text.strip() != st.session_state.e2_last_analyzed_text.strip():
        st.session_state.e2_custom_text = st.session_state.e2_last_analyzed_text = user_target_text.strip()
        st.session_state.e2_eval_history, st.session_state.e2_audio_bytes = [], None

    is_custom_now = (" ".join(st.session_state.e2_custom_text.lower().split()) != " ".join(active_bm["target_text"].lower().split()))

    card_col1, card_col2 = st.columns([1.45, 1.05])
    with card_col1:
        e2_input = st.audio_input("この英文を録音してプロソディ診断を実行", key="e2_audio_in")
        e2_uploaded_file = st.file_uploader("または音声ファイルをアップロード (WAV / MP3 / M4A)", type=["wav", "mp3", "m4a"], key="e2_file_in")

        # 録音・アップロードデータの抽出
        active_bytes = None
        if e2_input is not None:
            active_bytes = e2_input.getvalue()
        elif e2_uploaded_file is not None:
            active_bytes = e2_uploaded_file.getvalue()

        # 新規データが届いた場合のみ解析を実行
        if active_bytes and active_bytes != st.session_state.get("e2_last_seen_bytes"):
            st.session_state["e2_last_seen_bytes"] = active_bytes
            st.session_state.e2_audio_bytes = active_bytes

            audio_arr, sr = decode_audio_bytes(active_bytes)
            tmp_wav = f"tmp_e2_active.wav"
            sf.write(tmp_wav, audio_arr, sr, format="WAV")

            eval_res = evaluate_user_rhythm(tmp_wav, st.session_state.e2_selected_passage, st.session_state.e2_custom_text)
            if os.path.exists(tmp_wav):
                try: os.remove(tmp_wav)
                except Exception: pass

            st.session_state.e2_eval_history.append(eval_res)

        if st.session_state.e2_audio_bytes is not None:
            _, dl_col2 = st.columns([1.8, 1.2])
            with dl_col2:
                st.download_button(label="録音音声を保存 (.wav)", data=st.session_state.e2_audio_bytes, file_name=f"NinjaPhonetics_{st.session_state.e2_selected_passage}_recording.wav", mime="audio/wav", use_container_width=True)

        word_bench_dist = get_benchmark_word_raw_distributions(st.session_state.e2_selected_passage) if not is_custom_now else {}
        radar_html = build_e2_multiview_canvas_html(
            st.session_state.e2_selected_passage, st.session_state.e2_eval_history,
            is_custom_mode=is_custom_now, view_mode=st.session_state.e2_plot_view, word_bench_dist=word_bench_dist
        )
        components.html(radar_html, height=465, scrolling=False)

    with card_col2:
        st.markdown(f"<div class='sec-header'>{SHURIKEN_ICON_SM} 多次元プロソディ診断結果 (プロット表示切替)</div>", unsafe_allow_html=True)
        v_btn1, v_btn2, v_btn3 = st.columns(3)
        with v_btn1:
            if st.button("1. 2D 総合楕円", type="primary" if st.session_state.e2_plot_view == "scatter2d" else "secondary", use_container_width=True):
                st.session_state.e2_plot_view = "scatter2d"; st.rerun()
        with v_btn2:
            if st.button("2. 指標 1D 分布", type="primary" if st.session_state.e2_plot_view == "metrics_1d" else "secondary", use_container_width=True):
                st.session_state.e2_plot_view = "metrics_1d"; st.rerun()
        with v_btn3:
            if st.button("3. 機能語 Swarm", type="primary" if st.session_state.e2_plot_view == "word_swarm" else "secondary", use_container_width=True):
                st.session_state.e2_plot_view = "word_swarm"; st.rerun()

        if st.session_state.e2_eval_history:
            latest_ev = st.session_state.e2_eval_history[-1]
            trial_no = len(st.session_state.e2_eval_history)
            dm_str = f"{latest_ev.get('d_m_sq', 'N/A')} (≤9.21 合格)" if latest_ev.get("d_m_sq") is not None else "自由文判定"
            c_diag = "#10B981" if latest_ev.get("diag_median_ms", 99) <= 48.0 else ("#F59E0B" if latest_ev.get("diag_median_ms", 99) <= 65.0 else "#EF4444")
            c_schwa = "#10B981" if latest_ev.get("schwa_p75_ms", 99) <= 65.0 else "#EF4444"
            c_rcf = "#10B981" if latest_ev.get("r_cf", 0) >= 1.30 else "#EF4444"
            c_npvi = "#10B981" if 40.0 <= latest_ev.get("npvi_clipped", 0) <= 62.0 else "#F59E0B"

            summary_html = (
                f'<div style="background: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; margin-top: 6px; margin-bottom: 10px;">'
                f'<div style="display: flex; justify-content: space-between; align-items: baseline;">'
                f'<span style="font-size: 0.82rem; color: #94A3B8; font-weight: 600;">VERDICT (Trial #{trial_no})</span>'
                f'<span style="font-size: 0.78rem; color: #64748B;">蓄積データ: {trial_no} 件</span></div>'
                f'<div style="font-size: 1.45rem; font-weight: 800; color: {latest_ev.get("status_color", "#EF4444")}; margin-bottom: 8px;">{latest_ev.get("verdict", "OFF TARGET")}</div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #BAE6FD; font-size: 0.86rem; font-weight: 700;">弱化必須語 中央値 (V<sub>diag</sub>):</span>'
                f'<span style="font-weight: 800; color: {c_diag};">{latest_ev.get("diag_median_ms", 0)} ms <small style="font-size:0.75rem; color:#94A3B8;">(目標 ≤45ms)</small></span></div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">弱母音持続時間 (Schwa P75):</span>'
                f'<span style="font-weight: bold; color: {c_schwa};">{latest_ev.get("schwa_p75_ms", 0):.1f} ms <small style="font-size:0.75rem; color:#94A3B8;">(目標 ≤65ms)</small></span></div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">内容語/機能語比 (R<sub>CF</sub>):</span>'
                f'<span style="font-weight: bold; color: {c_rcf};">{latest_ev.get("r_cf", 0)}x <small style="font-size:0.75rem; color:#94A3B8;">(内容 {latest_ev.get("mean_c_ms", 0)}ms / 機能 {latest_ev.get("mean_f_ms", 0)}ms)</small></span></div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 4px; margin-bottom: 4px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">リズム伸縮指数 (nPVI-V):</span>'
                f'<span style="font-weight: bold; color: {c_npvi};">{latest_ev.get("npvi_clipped", 0)} <small style="font-size:0.75rem; color:#94A3B8;">(適正 40〜60)</small></span></div>'
                f'<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">'
                f'<span style="color: #94A3B8; font-size: 0.86rem;">2D Mahalanobis (D<sub>M</sub><sup>2</sup>):</span>'
                f'<span style="font-weight: bold; color: #F8FAFC;">{dm_str}</span></div>'
                f'<div style="background: #090D16; border-radius: 6px; padding: 8px; font-size: 0.83rem; color: #E2E8F0; line-height: 1.4;">'
                f'💡 <b>フィードバック:</b> {latest_ev.get("feedback", "")}</div></div>'
            )
            st.markdown(summary_html, unsafe_allow_html=True)

            wb_list = latest_ev.get("word_breakdown", [])
            diag_words_meas = [w for w in wb_list if w.get("is_diag")]
            if diag_words_meas:
                st.markdown("<div style='font-size: 0.88rem; font-weight: bold; color: #BAE6FD; margin-top: 2px; margin-bottom: 4px;'>弱化必須語 (冠詞除く) の実測母音長一覧</div>", unsafe_allow_html=True)
                rows_html = ""
                for idx_w, dw in enumerate(diag_words_meas):
                    v_val = dw["vowel_ms"]
                    if v_val <= 48.0: status_txt, bar_c, badge_style = "合格", "#10B981", "background:#064E3B; color:#A7F3D0; border:1px solid #10B981;"
                    elif v_val <= 65.0: status_txt, bar_c, badge_style = "やや長", "#F59E0B", "background:#78350F; color:#FDE68A; border:1px solid #F59E0B;"
                    else: status_txt, bar_c, badge_style = "引き伸ばし", "#EF4444", "background:#7F1D1D; color:#FECACA; border:1px solid #EF4444;"
                    bar_w = min(100, int((v_val / 130.0) * 100))
                    rows_html += (
                        f'<tr style="border-bottom: 1px solid #1E293B; font-size: 12px;">'
                        f'<td style="padding: 4px 6px; color: #64748B;">#{idx_w+1}</td>'
                        f'<td style="padding: 4px 6px; font-weight: 700; color: #F8FAFC;">{dw["word"]}</td>'
                        f'<td style="padding: 4px 6px; font-weight: 700; color: {bar_c}; font-variant-numeric: tabular-nums;">{v_val:.0f} ms</td>'
                        f'<td style="padding: 4px 6px; width: 34%;"><div style="background:#1E293B; height:7px; border-radius:4px; overflow:hidden;"><div style="width:{bar_w}%; background:{bar_c}; height:100%;"></div></div></td>'
                        f'<td style="padding: 4px 6px; text-align: right;"><span style="{badge_style} padding: 1px 5px; border-radius: 4px; font-size: 10.5px; font-weight: bold;">{status_txt}</span></td></tr>'
                    )
                table_html = f"""<!DOCTYPE html><html><head><style>
body {{ margin: 0; padding: 0; background: #0F172A; color: #E2E8F0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Hiragino Sans", sans-serif; }}
.tbl-box {{ max-height: 175px; overflow-y: auto; border: 1px solid #1E293B; border-radius: 6px; padding: 2px 6px; background: #0F172A; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ position: sticky; top: 0; background: #0F172A; color: #94A3B8; font-size: 11px; text-align: left; padding: 4px 6px; border-bottom: 1px solid #334155; }}
</style></head><body>
<div class="tbl-box"><table><thead><tr><th>順</th><th>単語</th><th>母音長</th><th>ゲージ (目標≤45ms)</th><th style="text-align:right;">判定</th></tr></thead><tbody>{rows_html}</tbody></table></div></body></html>"""
                components.html(table_html, height=185, scrolling=False)

            if st.button("この課題の試行履歴をクリア", use_container_width=True):
                st.session_state.e2_eval_history, st.session_state.e2_audio_bytes, st.session_state["e2_last_seen_bytes"] = [], None, None; st.rerun()
        else:
            st.markdown("""<div style="background: #0F172A; border: 1px dashed #334155; border-radius: 8px; padding: 24px; text-align: center; color: #94A3B8;">
左側のマイクボタンを押して、英文を読み上げてください。<br><br>
2D ストライクゾーン (R_CF × Schwa P75) による客観的な判定が行われます。
</div>""", unsafe_allow_html=True)

        if not is_custom_now and os.path.exists(active_bm["ref_audio"]):
            st.markdown("<div style='font-size: 0.9rem; font-weight: bold; color: #94A3B8; margin-top: 8px; margin-bottom: 4px;'>ネイティブお手本音声</div>", unsafe_allow_html=True)
            with open(active_bm["ref_audio"], "rb") as rf: st.audio(rf.read(), format="audio/wav")

    # 最下段：発話実測マップ
    if st.session_state.e2_eval_history and st.session_state.e2_audio_bytes:
        latest_wb = st.session_state.e2_eval_history[-1].get("word_breakdown", [])
        trial_num = len(st.session_state.e2_eval_history)
        map_html = build_clickable_word_map_html(latest_wb, st.session_state.e2_audio_bytes, trial_num)
        components.html(map_html, height=155, scrolling=False)