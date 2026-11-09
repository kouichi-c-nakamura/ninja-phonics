import io
import json
import os
from pathlib import Path
import requests
import soundfile as sf
import dotenv

dotenv.load_dotenv()
current_path = Path.cwd()
PROJECT_ROOT = current_path.parent if current_path.name == "scripts" else current_path
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
    "Then the Sun shone out warmly, and immediately the traveler took off his cloak. "
    "And so the North Wind was obliged to confess that the Sun was the stronger of the two."
)

JP_MORA_FULL_TEXT = (
    "Za No-su Windo ando za San wa- disupyu-tingu huicchi wazu za sutoronga-, "
    "wen e torabera- ke-mu arongu warapputo in e wa-mu kuro-ku. "
    "Zei aguri-do zatto za wan hu- fa-suto sakusi-deddo in me-kingu za torabera- "
    "te-ku hizu kuro-ku ohhu syuddo bi- konsida-do sutoronga- zan zi aza-. "
    "Zen za No-su Windo buryu- wizu o-ru hizu maito, batto za moa hi- buryu-, "
    "za moa kuro-suri- diddo za torabera- fo-rudo hizu kuro-ku araundo himu, "
    "ando atto rasuto za No-su Windo ge-bu appu zi atemputo. "
    "Zen za San syon auto wa-muri-, ando imi-dietori- za torabera- tukku ohhu hizu kuro-ku. "
    "Ando so- za No-su Windo wazu oburaijido tu- konfesu zatto za San wazu za sutoronga- obu za tu-."
)

# voice_registry.json の確定ボイス
ADDITIONAL_JP_SPEAKERS = [
    {"name": "Riu", "voice_id": "kDjNtb1AbMP0kqZ1BUxW", "gender": "male", "idx": 4},
    {"name": "Kaori", "voice_id": "30Bl9aRayVzd7rYeBNHc", "gender": "female", "idx": 5}
]

def generate_voice(voice_id, text, model_id="eleven_turbo_v2_5"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API Error ({response.status_code}): {response.text}")
    return response.content

def main():
    print("北風と太陽【日本人コーホート4本追加生成】開始...")
    print("消費予定: 1,132 credits (残 2,664 -> 約 1,532 残)")

    manifest = {}
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    for cohort in ["JP-Accented", "JP-Mora-Forced"]:
        cohort_dir = BENCHMARKS_DIR / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)
        if cohort not in manifest:
            manifest[cohort] = []

        for spk in ADDITIONAL_JP_SPEAKERS:
            out_file = cohort_dir / f"{cohort}_{spk['idx']:02d}_{spk['name']}.wav"
            if out_file.exists():
                print(f"⏩ スキップ（既存）: {out_file.name}")
                continue

            text_to_use = JP_MORA_FULL_TEXT if cohort == "JP-Mora-Forced" else PASSAGE_FULL_TEXT
            print(f"生成中: [{cohort}] {spk['name']} (ID: {spk['voice_id']})...")
            
            audio_bytes = generate_voice(spk["voice_id"], text_to_use)
            data, sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(str(out_file), data, sr)
            print(f"  -> 保存完了: {out_file.name}")

            manifest[cohort].append({
                "file": out_file.name,
                "name": spk["name"],
                "voice_id": spk["voice_id"],
                "gender": spk["gender"]
            })

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 日本人音声4本の追加完了。マニフェスト更新: {MANIFEST_FILE}")

if __name__ == "__main__":
    main()