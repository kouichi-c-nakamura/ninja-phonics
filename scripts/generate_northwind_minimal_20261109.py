from pathlib import Path
import os
import json
import requests
import soundfile as sf
import io
import dotenv

dotenv.load_dotenv()

# プロジェクトルートの特定
current_path = Path.cwd()
PROJECT_ROOT = current_path.parent if current_path.name == "scripts" else current_path
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks" / "P1_northwind"

# ElevenLabs APIキーの取得
API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not API_KEY:
    raise ValueError("ELEVENLABS_API_KEY が環境変数に設定されていません。")

# 53語抜粋テキスト
PASSAGE_TEXT = (
    "The North Wind and the Sun were disputing which was the stronger, "
    "when a traveler came along wrapped in a warm cloak. "
    "They agreed that the one who first succeeded in making the traveler "
    "take his cloak off should be considered stronger than the other."
)

# 各コーホート代表3名（計12本）
SELECTED_SPEAKERS = {
    "US-Native": [
        {"name": "Roger", "voice_id": "CwhRBWXzGAHq8TQ4Fs17"},       # Laid-back male
        {"name": "Jessica", "voice_id": "cgSgspJ2msm6clMCkdW9"},     # Playful female
        {"name": "Chris", "voice_id": "iP95p4xoKVk53GoZ742B"}        # Charming male
    ],
    "UK-Native": [
        {"name": "Archer", "voice_id": "LcfcDJNUP1GQjkzn1xUU"},      # Conversational male
        {"name": "Blondie", "voice_id": "JBFqnCBsd6RMkjVDRZzb"},     # Conversational female
        {"name": "Tom", "voice_id": "bIHbv24MWmeRgasZH58o"}          # British male
    ],
    "JP-Accented": [
        {"name": "Mitami", "voice_id": "z9fAnlkpzviPz146aGWa"},      # Japanese accent female
        {"name": "Believe", "voice_id": "TX3LPaxmHKxFdv7VOQHJ"},     # Japanese accent male
        {"name": "Makoto", "voice_id": "AZnzlk1XvdvUeBnXmlld"}       # Japanese accent male
    ],
    "JP-Mora-Forced": [
        {"name": "Mitami", "voice_id": "z9fAnlkpzviPz146aGWa"},
        {"name": "Believe", "voice_id": "TX3LPaxmHKxFdv7VOQHJ"},
        {"name": "Makoto", "voice_id": "AZnzlk1XvdvUeBnXmlld"}
    ]
}

# 日本語カタカナ読み用テキスト（Mora拍誘導用）
# 拗音・促音・長音を含め、均等モーラ拍を機械的に引き出すテキスト
JP_MORA_TEXT = (
    "Za No-su Windo ando za San wa- disupyu-tingu huicchi wazu za sutoronga-, "
    "wen e torabera- ke-mu arongu warapputo in e wa-mu kuro-ku. "
    "Zei aguri-do zatto za wan hu- fa-suto sakusi-deddo in me-kingu za torabera- "
    "te-ku hizu kuro-ku ohhu syuddo bi- konsida-do sutoronga- zan zi aza-."
)

def generate_voice(voice_id, text, model_id="eleven_turbo_v2_5"):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"ElevenLabs API Error ({response.status_code}): {response.text}")
    return response.content

def main():
    print(f"長文ベンチマーク (The North Wind and the Sun) 生成開始...")
    print(f"モデル: eleven_turbo_v2_5 (消費0.5 credit/文字)")
    print(f"対象文字数: {len(PASSAGE_TEXT)} 文字 × 12本 ＝ 約 1,550 credits")
    
    for cohort, speakers in SELECTED_SPEAKERS.items():
        cohort_dir = BENCHMARKS_DIR / cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, spk in enumerate(speakers):
            out_file = cohort_dir / f"{cohort}_{idx+1:02d}_{spk['name']}.wav"
            if out_file.exists():
                print(f"スキップ（既存）: {out_file.name}")
                continue
                
            use_text = JP_MORA_TEXT if cohort == "JP-Mora-Forced" else PASSAGE_TEXT
            print(f"生成中: [{cohort}] {spk['name']}...")
            
            audio_bytes = generate_voice(spk["voice_id"], use_text)
            
            # 16kHz WAV に変換して保存
            data, sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(str(out_file), data, sr)
            print(f"  -> 保存完了: {out_file.name}")

    print("\n✅ 全12本の生成が完了しました！")

if __name__ == "__main__":
    main()