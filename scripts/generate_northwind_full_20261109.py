import io
import json
import os
from pathlib import Path
import requests
import soundfile as sf
import dotenv

dotenv.load_dotenv()

current_path = Path.cwd()
PROJECT_ROOT = (
    current_path.parent if current_path.name == "scripts" else current_path
)
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks" / "P1_northwind"

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not API_KEY:
  raise ValueError("ELEVENLABS_API_KEY が環境変数に設定されていません。")

# 北風と太陽 全文テキスト (113語)
PASSAGE_FULL_TEXT = (
    "The North Wind and the Sun were disputing which was the stronger, "
    "when a traveler came along wrapped in a warm cloak. "
    "They agreed that the one who first succeeded in making the traveler "
    "take his cloak off should be considered stronger than the other. "
    "Then the North Wind blew with all his might, but the more he blew, "
    "the more closely did the traveler fold his cloak around him, "
    "and at last the North Wind gave up the attempt. "
    "Then the Sun shone out warmly, and immediately the traveler took off his"
    " cloak. "
    "And so the North Wind was obliged to confess that the Sun was the stronger"
    " of the two."
)

# カタカナ全文 (等時拍モーラ誘導用)
JP_MORA_FULL_TEXT = (
    "Za No-su Windo ando za San wa- disupyu-tingu huicchi wazu za sutoronga-, "
    "wen e torabera- ke-mu arongu warapputo in e wa-mu kuro-ku. "
    "Zei aguri-do zatto za wan hu- fa-suto sakusi-deddo in me-kingu za"
    " torabera- "
    "te-ku hizu kuro-ku ohhu syuddo bi- konsida-do sutoronga- zan zi aza-. "
    "Zen za No-su Windo buryu- wizu o-ru hizu maito, batto za moa hi- buryu-, "
    "za moa kuro-suri- diddo za torabera- fo-rudo hizu kuro-ku araundo himu, "
    "ando atto rasuto za No-su Windo ge-bu appu zi atemputo. "
    "Zen za San syon auto wa-muri-, ando imi-dietori- za torabera- tukku ohhu"
    " hizu kuro-ku. "
    "Ando so- za No-su Windo wazu oburaijido tu- konfesu zatto za San wazu za"
    " sutoronga- obu za tu-."
)

SELECTED_SPEAKERS = {
    "US-Native": [
        {"name": "Roger", "voice_id": "CwhRBWXzGAHq8TQ4Fs17"},
        {"name": "Jessica", "voice_id": "cgSgspJ2msm6clMCkdW9"},
        {"name": "Chris", "voice_id": "iP95p4xoKVk53GoZ742B"},
    ],
    "UK-Native": [
        {"name": "Archer", "voice_id": "LcfcDJNUP1GQjkzn1xUU"},
        {"name": "Blondie", "voice_id": "JBFqnCBsd6RMkjVDRZzb"},
        {"name": "Tom", "voice_id": "bIHbv24MWmeRgasZH58o"},
    ],
    "JP-Accented": [
        {"name": "Mitami", "voice_id": "z9fAnlkpzviPz146aGWa"},
        {"name": "Believe", "voice_id": "TX3LPaxmHKxFdv7VOQHJ"},
        {"name": "Makoto", "voice_id": "AZnzlk1XvdvUeBnXmlld"},
    ],
    "JP-Mora-Forced": [
        {"name": "Mitami", "voice_id": "z9fAnlkpzviPz146aGWa"},
        {"name": "Believe", "voice_id": "TX3LPaxmHKxFdv7VOQHJ"},
        {"name": "Makoto", "voice_id": "AZnzlk1XvdvUeBnXmlld"},
    ],
}


def generate_voice(voice_id, text, model_id="eleven_turbo_v2_5"):
  url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
  headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
  payload = {
      "text": text,
      "model_id": model_id,
      "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
  }
  response = requests.post(url, json=payload, headers=headers)
  if response.status_code != 200:
    raise RuntimeError(
        f"ElevenLabs API Error ({response.status_code}): {response.text}"
    )
  return response.content


def main():
  print("北風と太陽【全文 113語】ベンチマーク生成開始...")
  print(f"文字数: {len(PASSAGE_FULL_TEXT)} 文字 × 12本")
  print("予想消費: 約 3,396 credits (残枠 3,858 内)")

  for cohort, speakers in SELECTED_SPEAKERS.items():
    cohort_dir = BENCHMARKS_DIR / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)

    for idx, spk in enumerate(speakers):
      out_file = cohort_dir / f"{cohort}_{idx+1:02d}_{spk['name']}.wav"
      use_text = (
          JP_MORA_FULL_TEXT if cohort == "JP-Mora-Forced" else PASSAGE_FULL_TEXT
      )

      print(f"生成中: [{cohort}] {spk['name']}...")
      audio_bytes = generate_voice(spk["voice_id"], use_text)

      data, sr = sf.read(io.BytesIO(audio_bytes))
      sf.write(str(out_file), data, sr)
      print(f"  -> 保存完了: {out_file.name}")

  print("\n✅ 北風と太陽【全文】12本の生成が完了しました！")


if __name__ == "__main__":
  main()