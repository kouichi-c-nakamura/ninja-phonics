import os
import re
import json
import time
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not ELEVEN_API_KEY:
    raise ValueError("環境変数 ELEVENLABS_API_KEY を設定してください。")

HEADERS = {
    "xi-api-key": ELEVEN_API_KEY,
    "Content-Type": "application/json"
}

# お題スクリプト（英語原文 vs 強制モーラ化カタカナ原稿）
SCRIPT_EN = "A photograph of a famous photographer was taken in the station."
SCRIPT_KATAKANA = "ア フォトグラフ オブ ア フェイマス フォトグラファー ワズ テイクン イン ザ ステーション"

OUTPUT_DIR = "benchmarks"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")


def fetch_candidate_voices():
    """自分のアカウント内のボイス + 共有Voice Libraryから候補を取得"""
    candidates = {"uk": [], "us": [], "jp": []}

    # 1. デフォルト・マイボイスの取得
    r = requests.get("https://api.elevenlabs.io/v1/voices", headers=HEADERS)
    if r.status_code == 200:
        for v in r.json().get("voices", []):
            labels = v.get("labels", {})
            accent = str(labels.get("accent", "")).lower()
            info = {
                "voice_id": v["voice_id"],
                "name": v["name"],
                "gender": labels.get("gender", "unknown"),
                "age": labels.get("age", "unknown"),
                "use_case": labels.get("use_case", "general"),
                "description": labels.get("description", ""),
                "accent": accent
            }
            if "british" in accent or "uk" in accent:
                candidates["uk"].append(info)
            elif "american" in accent or "us" in accent:
                candidates["us"].append(info)
            elif "japanese" in accent:
                candidates["jp"].append(info)

    # 2. 不足分（特にJapaneseや多様なUK/US）を共有ライブラリから検索補完
    search_queries = [
        ("uk", {"accent": "british", "page_size": 30}),
        ("us", {"accent": "american", "page_size": 30}),
        ("jp", {"language": "ja", "page_size": 30}),
        ("jp", {"search": "japanese", "page_size": 30}),
    ]

    for key, params in search_queries:
        if len(candidates[key]) >= 25:
            continue
        r_shared = requests.get(
            "https://api.elevenlabs.io/v1/shared-voices",
            headers=HEADERS,
            params=params
        )
        if r_shared.status_code == 200:
            existing_ids = {x["voice_id"] for x in candidates[key]}
            for sv in r_shared.json().get("voices", []):
                vid = sv.get("voice_id")
                if vid and vid not in existing_ids:
                    candidates[key].append({
                        "voice_id": vid,
                        "name": sv.get("name", "shared"),
                        "gender": sv.get("gender", "unknown"),
                        "age": sv.get("age", "unknown"),
                        "use_case": sv.get("use_case", "general"),
                        "description": sv.get("description", "")[:80],
                        "accent": sv.get("accent", key)
                    })
                    existing_ids.add(vid)

    return candidates


def select_10_per_group_with_llm(candidates: dict, target_count: int = 10):
    """Gemini APIを用いて男女比・会話調のバランスが良い各10名を自動選抜（未設定時はルールベース）"""
    if GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""
あなたは音声学ベンチマーク構築のエキスパートです。
以下のElevenLabsボイス候補リスト（uk, us, jp）から、各グループ最大 {target_count} 個の `voice_id` を選抜してください。

【選抜基準】
1. 男女比（female / male）ができるだけ均等になるように選ぶこと。
2. 極端なキャラクターボイス（monster, child, robot等）は除外し、conversational または narrative な自然な声を選ぶこと。
3. JSON形式で以下のキー構成のみを出力すること:
{{"uk": ["id1", ...], "us": ["id1", ...], "jp": ["id1", ...]}}

候補データ:
{json.dumps(candidates, ensure_ascii=False)}
"""
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            selected_ids = json.loads(resp.text)
            result = {}
            for k in ["uk", "us", "jp"]:
                id_set = set(selected_ids.get(k, []))
                picked = [v for v in candidates[k] if v["voice_id"] in id_set][:target_count]
                if len(picked) < target_count:
                    # 足りない分を先頭から補完
                    for v in candidates[k]:
                        if v not in picked and len(picked) < target_count:
                            picked.append(v)
                result[k] = picked
            print("✅ Gemini Agentによる最適ボイス選抜が完了しました。")
            return result
        except Exception as e:
            print(f"⚠️ Gemini選抜スキップ（ルールベース選抜へ切替）: {e}")

    # フォールバック：男女交互にルールベース選抜
    result = {}
    for k in ["uk", "us", "jp"]:
        females = [v for v in candidates[k] if str(v["gender"]).lower() == "female"]
        males = [v for v in candidates[k] if str(v["gender"]).lower() == "male"]
        others = [v for v in candidates[k] if str(v["gender"]).lower() not in ("female", "male")]
        interleaved = []
        for i in range(max(len(females), len(males), len(others))):
            if i < len(females): interleaved.append(females[i])
            if i < len(males): interleaved.append(males[i])
            if i < len(others): interleaved.append(others[i])
        result[k] = interleaved[:target_count]
    return result


def synthesize_to_wav(voice_id: str, text: str, out_wav_path: str, lang_code: str = None):
    """ElevenLabs APIでMP3を生成し、ffmpegで44.1kHz 16bit mono WAVに変換"""
    if os.path.exists(out_wav_path):
        print(f"  [Skip] 既に存在します: {out_wav_path}")
        return True

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.40,
            "similarity_boost": 0.75,
            "style": 0.15
        }
    }
    if lang_code:
        payload["language_code"] = lang_code

    r = requests.post(url, json=payload, headers=HEADERS)
    if r.status_code != 200:
        print(f"  [Error] TTS失敗 ({voice_id}): {r.status_code} {r.text[:120]}")
        return False

    tmp_mp3 = out_wav_path.replace(".wav", "_tmp.mp3")
    with open(tmp_mp3, "wb") as f:
        f.write(r.content)

    # ffmpeg で 44.1kHz 16-bit mono WAV へ変換
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_mp3,
        "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le",
        out_wav_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(tmp_mp3)
    time.sleep(0.4)  # APIレートリミット配慮
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🔍 ElevenLabsからボイス候補を検索中...")
    candidates = fetch_candidate_voices()
    print(f"  候補数 -> UK: {len(candidates['uk'])}, US: {len(candidates['us'])}, JP: {len(candidates['jp'])}")

    selected = select_10_per_group_with_llm(candidates, target_count=10)

    manifest = []
    cohorts = [
        ("UK-Native", selected["uk"], SCRIPT_EN, "en"),
        ("US-Native", selected["us"], SCRIPT_EN, "en"),
        ("JP-Accented", selected["jp"], SCRIPT_EN, None),
        ("JP-Mora-Forced", selected["jp"], SCRIPT_KATAKANA, "ja"),
    ]

    for cohort_name, voice_list, script_text, lang_code in cohorts:
        cohort_dir = os.path.join(OUTPUT_DIR, cohort_name)
        os.makedirs(cohort_dir, exist_ok=True)
        print(f"\n🎙️ コホート生成中: {cohort_name} ({len(voice_list)} voices)")

        for idx, v in enumerate(voice_list, 1):
            safe_name = re.sub(r"[^a-zA-Z0-9]", "", v["name"])[:12] or f"v{idx}"
            filename = f"{cohort_name}_{idx:02d}_{safe_name}_{v['gender']}.wav"
            wav_path = os.path.join(cohort_dir, filename)

            print(f"  ({idx}/{len(voice_list)}) Generating {filename} ...")
            ok = synthesize_to_wav(v["voice_id"], script_text, wav_path, lang_code=lang_code)
            if ok:
                manifest.append({
                    "cohort": cohort_name,
                    "voice_id": v["voice_id"],
                    "name": v["name"],
                    "gender": v["gender"],
                    "wav_path": wav_path
                })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✨ 全ベンチマーク音声の生成完了！マニフェスト保存先: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()