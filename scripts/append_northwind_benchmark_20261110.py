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
MANIFEST_FILE = BENCHMARKS_DIR / "voices_manifest.json"

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not API_KEY:
  raise ValueError("ELEVENLABS_API_KEY が環境変数に設定されていません。")

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

# voice_registry.json に登録されている正規 Voice ID
NEW_SPEAKERS = {
    "US-Native": [
        {
            "name": "Brian",
            "voice_id": "Gubgw9l4dtIoQA9YZHgx",
            "gender": "male",
            "idx": 4,
        },
        {
            "name": "Eric",
            "voice_id": "cjVigY5qzO86Huf0OWal",
            "gender": "male",
            "idx": 5,
        },
        {
            "name": "Elise",
            "voice_id": "EST9Ui6982FZPSi7gCHi",
            "gender": "female",
            "idx": 6,
        },
    ],
    "UK-Native": [
        {
            "name": "Leon",
            "voice_id": "1LECWyROimjWHAqjvjqa",
            "gender": "male",
            "idx": 4,
        },
        {
            "name": "Lucy",
            "voice_id": "lcMyyd2HUfFzxdCaC4Ta",
            "gender": "female",
            "idx": 5,
        },
        {
            "name": "Fraser",
            "voice_id": "MNUBQXVM4SBzrA4AbVeB",
            "gender": "male",
            "idx": 6,
        },
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
  print("北風と太陽【ネイティブ追加生成（registry検証済み）】開始...")

  manifest = {}
  if MANIFEST_FILE.exists():
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
      manifest = json.load(f)

  for cohort, speakers in NEW_SPEAKERS.items():
    cohort_dir = BENCHMARKS_DIR / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)
    if cohort not in manifest:
      manifest[cohort] = []

    for spk in speakers:
      out_file = cohort_dir / f"{cohort}_{spk['idx']:02d}_{spk['name']}.wav"

      if out_file.exists():
        print(f"⏩ スキップ（既存）: {out_file.name}")
        continue

      print(f"生成中: [{cohort}] {spk['name']} (ID: {spk['voice_id']})...")
      audio_bytes = generate_voice(spk["voice_id"], PASSAGE_FULL_TEXT)
      data, sr = sf.read(io.BytesIO(audio_bytes))
      sf.write(str(out_file), data, sr)
      print(f"  -> 保存完了: {out_file.name}")

      manifest[cohort].append({
          "file": out_file.name,
          "name": spk["name"],
          "voice_id": spk["voice_id"],
          "gender": spk["gender"],
      })

  with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
  print(f"\n✅ 全ネイティブ音声の補完とマニフェスト保存が完了しました: {MANIFEST_FILE}")


if __name__ == "__main__":
  main()