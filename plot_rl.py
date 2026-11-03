import numpy as np
import parselmouth
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_rl_dynamics(wav_path: str, max_formant_hz: float = 5000.0):
    """音声ファイルからホルマント(F2, F3)と差分Δ(F3-F2)を抽出し、
    視認性を高めたPlotlyグラフを生成して返す。
    """
    snd = parselmouth.Sound(wav_path)

    # 音響特徴量の抽出
    pitch = snd.to_pitch()
    formant = snd.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5.0,
        maximum_formant=max_formant_hz,
    )
    intensity = snd.to_intensity()

    times = formant.xs()
    f2_list, f3_list, delta_list = [], [], []
    valid_times = []

    # 最大音圧を基準に、発話のある有効な有声区間のみを抽出
    max_int = np.max(intensity.values) if len(intensity.values) > 0 else 0
    threshold_int = max(30.0, max_int - 28.0)

    for t in times:
        f0 = pitch.get_value_at_time(t)
        inte = intensity.get_value(t)

        if not np.isnan(f0) and f0 > 0 and inte >= threshold_int:
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if not (np.isnan(f2) or np.isnan(f3)):
                f2_list.append(f2)
                f3_list.append(f3)
                delta_list.append(f3 - f2)
                valid_times.append(t)
            else:
                f2_list.append(None)
                f3_list.append(None)
                delta_list.append(None)
        else:
            f2_list.append(None)
            f3_list.append(None)
            delta_list.append(None)

    # 実線が存在する区間の前後0.04秒だけを残し、余白を自動カット
    if valid_times:
        x_min = max(0.0, min(valid_times) - 0.04)
        x_max = max(valid_times) + 0.04
    else:
        x_min, x_max = 0.0, snd.duration

    # サブプロット作成（タイトル全廃・マージン最適化）
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # 上段: F3, F2 実線描画
    fig.add_trace(
        go.Scatter(
            x=times,
            y=f3_list,
            mode="lines",
            name="F3",
            line=dict(color="#2563EB", width=3.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=f2_list,
            mode="lines",
            name="F2",
            line=dict(color="#059669", width=3.5),
        ),
        row=1,
        col=1,
    )

    # 下段: 差分カーブ描画
    fig.add_trace(
        go.Scatter(
            x=times,
            y=delta_list,
            mode="lines",
            name="Δ(F3-F2)",
            line=dict(color="#0F172A", width=4),
        ),
        row=2,
        col=1,
    )

    # 1000Hz 目安補助線（破線・20pxフォント）
    fig.add_hline(
        y=1000,
        line_dash="dash",
        line_color="#64748B",
        line_width=2,
        annotation_text="R Target (<1000Hz)",
        annotation_position="top right",
        annotation_font=dict(
            family="Arial, Helvetica, sans-serif",
            size=20,
            color="#475569",
        ),
        row=2,
        col=1,
    )

    # 座標軸の共通スタイル設定（軸線・目盛り線の可視化と2倍フォント）
    axis_common_style = dict(
        showline=True,            # 主軸線を表示
        linewidth=2,              # 軸線の太さ
        linecolor="#334155",      # くっきりとしたダークグレー
        ticks="outside",          # 目盛り線を外向きに表示
        ticklen=6,                # 目盛り線の長さ
        tickwidth=2,              # 目盛り線の太さ
        tickcolor="#334155",
        tickfont=dict(
            family="Arial, Helvetica, sans-serif",
            size=20,              # 目盛り数値（約2倍大）
            color="#0F172A",
        ),
        gridcolor="#E2E8F0",      # グリッド線（淡いグレー）
        gridwidth=1.5,
        zeroline=False,
    )

    # Y軸設定（タイトルのみ・24px）
    fig.update_yaxes(
        axis_common_style,
        title_text="F3 / F2 (Hz)",
        title_font=dict(
            family="Arial, Helvetica, sans-serif",
            size=24,
            color="#0F172A",
        ),
        range=[400, 3200],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        axis_common_style,
        title_text="Δ(F3 - F2) (Hz)",
        title_font=dict(
            family="Arial, Helvetica, sans-serif",
            size=24,
            color="#0F172A",
        ),
        range=[0, 1800],
        row=2,
        col=1,
    )

    # X軸設定（上段は目盛りのみ、下段にTimeラベル配置）
    fig.update_xaxes(
        axis_common_style,
        range=[x_min, x_max],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        axis_common_style,
        title_text="Time (s)",
        title_font=dict(
            family="Arial, Helvetica, sans-serif",
            size=24,
            color="#0F172A",
        ),
        range=[x_min, x_max],
        row=2,
        col=1,
    )

    # 全体レイアウト（フォント統一・大文字対応のマージン確保）
    fig.update_layout(
        height=650,
        margin=dict(l=110, r=50, t=40, b=80),
        template="plotly_white",
        hovermode="x unified",
        font=dict(family="Arial, Helvetica, sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(
                family="Arial, Helvetica, sans-serif",
                size=22,          # 凡例文字（約2倍大）
            ),
        ),
    )

    return fig