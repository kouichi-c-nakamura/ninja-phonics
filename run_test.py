import sounddevice as sd
import soundfile as sf
from plot_rl import plot_rl_dynamics


def record_and_plot():
    duration = 2.0  # 2秒間録音
    sample_rate = 44100
    wav_path = "temp_test.wav"

    print("\n発音する準備をしてください（例: right / light / ライト）...")
    input("Ready? Enterキーを押すと2秒間録音します > ")

    print("● 録音中...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(wav_path, audio, sample_rate)
    print("✔ 録音完了。波形を生成中...")

    # plot_rl からグラフを生成してブラウザで表示
    fig = plot_rl_dynamics(wav_path)
    if fig:
        fig.show()


if __name__ == "__main__":
    record_and_plot()