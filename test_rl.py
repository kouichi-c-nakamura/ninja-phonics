import numpy as np
import parselmouth
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def classify_right_light_raito(wav_path: str, max_formant_hz: float = 5000.0):
    snd = parselmouth.Sound(wav_path)
    
    pitch = snd.to_pitch()
    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5.0, maximum_formant=max_formant_hz)
    intensity = snd.to_intensity()

    times = formant.xs()
    f1_list, f2_list, f3_list, delta_list = [], [], [], []
    valid_times = []

    # 最大音圧の取得（ノイズと真の発声開始を切り分けるため）
    max_intensity = np.max(intensity.values) if len(intensity.values) > 0 else 0
    intensity_threshold = max(35.0, max_intensity - 25.0)  # ピークから25dB下または35dB以上

    for t in times:
        f0 = pitch.get_value_at_time(t)
        inte = intensity.get_value(t)
        
        # 音圧が十分かつピッチが存在する区間を有声区間とする
        if not np.isnan(f0) and f0 > 0 and inte >= intensity_threshold:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if not (np.isnan(f2) or np.isnan(f3)):
                f1_list.append(f1)
                f2_list.append(f2)
                f3_list.append(f3)
                delta_list.append(f3 - f2)
                valid_times.append(t)
            else:
                f1_list.append(None); f2_list.append(None); f3_list.append(None); delta_list.append(None)
        else:
            f1_list.append(None); f2_list.append(None); f3_list.append(None); delta_list.append(None)

    if not valid_times:
        return {"error": "有効な発声が検出されませんでした。マイク音量を確認してください。"}, None

    # 発声開始から120msを子音解析ウィンドウとする
    t_start_voice = valid_times[0]
    t_consonant_window = t_start_voice + 0.12

    onset_deltas = []
    onset_f3s = []
    for t, d, f3 in zip(times, delta_list, f3_list):
        if t_start_voice <= t <= t_consonant_window and d is not None:
            onset_deltas.append(d)
            onset_f3s.append(f3)

    min_onset_delta = float(np.min(onset_deltas)) if onset_deltas else 9999.0
    min_onset_f3 = float(np.min(onset_f3s)) if onset_f3s else 9999.0

    # --- 調整後の3者判別ロジック ---
    # 1. 英語 R: F3-F2の急激な狭小化 (<= 650Hz) が最優先指標
    if min_onset_delta <= 650 and min_onset_f3 <= 2300:
        detected_category = "English /r/ (right)"
        verdict = "R_SOUND"
        advice = "合格です！舌の両側面が上顎奥歯に吸着（アンカー）し、F3がF2に綺麗に引き寄せられています。"

    # 2. 英語 L: F3が高く維持され (>2100Hz)、かつF3-F2が大きく分離 (>1100Hz)
    elif min_onset_delta >= 1100 and min_onset_f3 >= 2100:
        detected_category = "English /l/ (light)"
        verdict = "L_SOUND"
        advice = "クリアな /l/ です。舌先が歯茎に密着し、F3が高く保たれて側音化しています。"

    # 3. 日本語の「ライト」([ɾ] 弾き音): 中間の危険ゾーン (650Hz < diff < 1100Hz) に滞在
    elif 650 < min_onset_delta < 1100:
        detected_category = "Japanese Flap [ɾ] (日本語のライト)"
        verdict = "JAPANESE_FLAP"
        advice = "日本語のラ行（舌先で歯茎を弾く音）になっています。舌先をどこにも当てず、舌の両側面を上の奥歯に押し当ててF3-F2差分を緑ゾーン（<650Hz）に落としてください。"

    else:
        detected_category = "Ambiguous / In-between"
        verdict = "UNCERTAIN"
        advice = "境界線上の音です。Rを狙うなら奥歯のアンカーを意識し、Lを狙うなら舌先を上前歯の裏に当ててください。"

    diagnosis_json = {
        "detected_class": detected_category,
        "verdict": verdict,
        "metrics": {
            "onset_min_F3_F2_diff_hz": round(min_onset_delta, 1),
            "onset_min_F3_hz": round(min_onset_f3, 1),
            "voice_onset_time_s": round(t_start_voice, 3)
        },
        "feedback": advice
    }

    # Plotly描画部分はそのまま流用
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"判定結果: {detected_category}",
            "語頭ターゲットレーダー: Δ(F3 - F2)"
        )
    )

    fig.add_trace(go.Scatter(x=times, y=f3_list, mode="lines+markers", name="F3", line=dict(color="#2563EB", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=f2_list, mode="lines+markers", name="F2", line=dict(color="#10B981", width=2.5)), row=1, col=1)
    fig.add_vrect(x0=t_start_voice, x1=t_consonant_window, fillcolor="rgba(100, 116, 139, 0.15)", line_width=1, line_dash="dash", annotation_text="語頭解析ウィンドウ", row=1, col=1)

    fig.add_trace(go.Scatter(x=times, y=delta_list, mode="lines+markers", name="F3 - F2", line=dict(color="#111827", width=3)), row=2, col=1)
    fig.add_vrect(x0=t_start_voice, x1=t_consonant_window, fillcolor="rgba(100, 116, 139, 0.15)", line_width=1, line_dash="dash", row=2, col=1)

    fig.add_hrect(y0=0, y1=650, fillcolor="rgba(16, 185, 129, 0.2)", line_width=0, annotation_text="English R ゾーン (<650Hz)", row=2, col=1)
    fig.add_hrect(y0=650, y1=1100, fillcolor="rgba(239, 68, 68, 0.18)", line_width=0, annotation_text="日本語ラ行 / エラー (650-1100Hz)", row=2, col=1)
    fig.add_hrect(y0=1100, y1=2200, fillcolor="rgba(59, 130, 246, 0.2)", line_width=0, annotation_text="English L ゾーン (>1100Hz)", row=2, col=1)

    fig.update_yaxes(title_text="周波数 (Hz)", range=[500, 3500], row=1, col=1)
    fig.update_yaxes(title_text="F3 - F2 差分 (Hz)", range=[0, 2000], row=2, col=1)
    fig.update_xaxes(title_text="時間 (秒)", range=[max(0, t_start_voice - 0.05), t_start_voice + 0.5], row=2, col=1)
    fig.update_layout(height=650, template="plotly_white", hovermode="x unified")

    return diagnosis_json, fig