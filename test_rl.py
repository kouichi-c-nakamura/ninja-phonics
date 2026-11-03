import numpy as np
import parselmouth
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def classify_right_light_raito(wav_path: str, max_formant_hz: float = 5000.0):
    """
    right (/r/), light (/l/), 日本語「ライト」([ɾ]) を判別し、
    診断結果とPlotly図を返す。
    """
    snd = parselmouth.Sound(wav_path)
    
    # ピッチ、ホルマント、インテンシティ（音圧）を抽出
    pitch = snd.to_pitch()
    formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5.0, maximum_formant=max_formant_hz)
    intensity = snd.to_intensity()

    times = formant.xs()
    f1_list, f2_list, f3_list, delta_list = [], [], [], []
    voiced_times = []

    for t in times:
        f0 = pitch.get_value_at_time(t)
        if not np.isnan(f0) and f0 > 0:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if not (np.isnan(f2) or np.isnan(f3)):
                f1_list.append(f1)
                f2_list.append(f2)
                f3_list.append(f3)
                delta_list.append(f3 - f2)
                voiced_times.append(t)
            else:
                f1_list.append(None); f2_list.append(None); f3_list.append(None); delta_list.append(None)
        else:
            f1_list.append(None); f2_list.append(None); f3_list.append(None); delta_list.append(None)

    if not voiced_times:
        return {"error": "有声音が検出されませんでした。マイク音量を確認してください。"}, None

    # --- 語頭子音（Onset）区間の特定 ---
    # 発声開始（有声化）から最初の120ms以内を「語頭子音〜渡り」として解析対象にする
    t_start_voice = voiced_times[0]
    t_consonant_window = t_start_voice + 0.12  # 語頭120msにフォーカス

    onset_deltas = []
    onset_f3s = []
    for t, d, f3 in zip(times, delta_list, f3_list):
        if t_start_voice <= t <= t_consonant_window and d is not None:
            onset_deltas.append(d)
            onset_f3s.append(f3)

    min_onset_delta = float(np.min(onset_deltas)) if onset_deltas else 9999.0
    min_onset_f3 = float(np.min(onset_f3s)) if onset_f3s else 9999.0

    # 語頭における音圧立ち上がり（ステップ性の有無）
    int_start = intensity.get_value(t_start_voice)
    int_dip = intensity.get_value(t_start_voice + 0.03)

    # --- 3者判別ロジック ---
    # 1. 英語 R: F3-F2の急激な狭小化 (< 650Hz) かつ F3の著しい低下
    if min_onset_delta <= 650 and min_onset_f3 <= 2100:
        detected_category = "English /r/ (right)"
        verdict = "R_SOUND"
        confidence = "HIGH"
        advice = "合格です。舌の後ろ側面が上顎奥歯にしっかり吸着（アンカー）し、F3がF2に綺麗に引き寄せられています。"

    # 2. 日本語の「ライト」([ɾ] 弾き音): 
    # F3は下がらず高く維持(>2000Hz)、かつF3-F2が中途半端(700-1100Hz)、持続時間が極めて短い
    elif min_onset_f3 > 1950 and (650 < min_onset_delta < 1100):
        detected_category = "Japanese Flap [ɾ] (日本語のライト)"
        verdict = "JAPANESE_FLAP"
        confidence = "HIGH"
        advice = "日本語のラ行（舌先で歯茎を弾く音）になっています。舌先をどこにも当てず、舌の両側面を上の奥歯に押し当ててF3を落としてください。"

    # 3. 英語 L: F3が高く維持され (>2200Hz)、F3-F2の分離が大きい (>1100Hz)
    elif min_onset_f3 >= 2100 and min_onset_delta >= 1100:
        detected_category = "English /l/ (light)"
        verdict = "L_SOUND"
        confidence = "HIGH"
        advice = "クリアな /l/ の音です。舌先が上前歯の付け根に密着し、F3が高く保たれて側音化しています。"

    else:
        detected_category = "Ambiguous / In-between"
        verdict = "UNCERTAIN"
        confidence = "MEDIUM"
        advice = "RともLとも判別しづらい中間の音です。Rを狙うなら奥歯のアンカーを強め、Lを狙うなら舌先を歯茎にしっかり当ててください。"

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

    # --- Plotly 可視化 ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"判定結果: {detected_category}",
            "語頭ターゲットレーダー: Δ(F3 - F2)"
        )
    )

    # 上段: F2とF3の軌跡
    fig.add_trace(go.Scatter(x=times, y=f3_list, mode="lines+markers", name="F3", line=dict(color="#2563EB", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=f2_list, mode="lines+markers", name="F2", line=dict(color="#10B981", width=2.5)), row=1, col=1)
    # 判定対象区間（語頭120ms）を縦帯で可視化
    fig.add_vrect(x0=t_start_voice, x1=t_consonant_window, fillcolor="rgba(100, 116, 139, 0.15)", line_width=1, line_dash="dash", annotation_text="語頭解析ウィンドウ", row=1, col=1)

    # 下段: 差分レーダー
    fig.add_trace(go.Scatter(x=times, y=delta_list, mode="lines+markers", name="F3 - F2", line=dict(color="#111827", width=3)), row=2, col=1)
    fig.add_vrect(x0=t_start_voice, x1=t_consonant_window, fillcolor="rgba(100, 116, 139, 0.15)", line_width=1, line_dash="dash", row=2, col=1)

    # 判定ゾーン背景
    fig.add_hrect(y0=0, y1=650, fillcolor="rgba(16, 185, 129, 0.2)", line_width=0, annotation_text="English R ゾーン (<650Hz)", row=2, col=1)
    fig.add_hrect(y0=650, y1=1100, fillcolor="rgba(239, 68, 68, 0.18)", line_width=0, annotation_text="日本語ラ行 / エラー (650-1100Hz)", row=2, col=1)
    fig.add_hrect(y0=1100, y1=2200, fillcolor="rgba(59, 130, 246, 0.2)", line_width=0, annotation_text="English L ゾーン (>1100Hz)", row=2, col=1)

    fig.update_yaxes(title_text="周波数 (Hz)", range=[500, 3500], row=1, col=1)
    fig.update_yaxes(title_text="F3 - F2 差分 (Hz)", range=[0, 2000], row=2, col=1)
    fig.update_xaxes(title_text="時間 (秒)", range=[max(0, t_start_voice - 0.05), t_start_voice + 0.5], row=2, col=1)
    fig.update_layout(height=650, template="plotly_white", hovermode="x unified")

    return diagnosis_json, fig