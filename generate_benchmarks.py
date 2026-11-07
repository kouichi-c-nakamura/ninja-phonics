import os
import re
import json
import time
import subprocess
import requests

ELEVEN_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not ELEVEN_API_KEY:
    raise ValueError("Environment variable ELEVENLABS_API_KEY must be set.")

HEADERS = {
    "xi-api-key": ELEVEN_API_KEY,
    "Content-Type": "application/json"
}

# The 3 Diagnostic Sentences (English vs. Forced-Mora Katakana)
PASSAGES = {
    "S1_stress": {
        "en": "A photograph of a famous photographer was taken in the station.",
        "ja": "ア フォトグラフ オブ ア フェイマス フォトグラファー ワズ テイクン イン ザ ステーション"
    },
    "S2_clusters": {
        "en": "Strict instructors asked complex questions about the project tests.",
        "ja": "ストリクト インストラクターズ アスクト コンプレックス クエスチョンズ アバウト ザ プロジェクト テスツ"
    },
    "S3_connected": {
        "en": "Can you give him a cup of water from the kitchen at lunch?",
        "ja": "キャン ユー ギブ ヒム ア カップ オブ ウォーター フロム ザ キッチン アット ランチ"
    }
}

OUTPUT_DIR = "benchmarks"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")


def fetch_candidate_voices():
    """Retrieve available voices from personal library and shared voices."""
    candidates = {"uk": [], "us": [], "jp": []}

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


def select_10_per_group(candidates: dict, target_count: int = 10):
    """Select balanced speaker lists via Gemini (or rule-based fallback)."""
    if GEMINI_API_KEY:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            prompt = f"""
Select up to {target_count} natural conversational voices for 'uk', 'us', and 'jp' from the list.
Maintain equal gender balance where possible. Output strictly valid JSON:
{{"uk": ["id1", ...], "us": ["id1", ...], "jp": ["id1", ...]}}

Candidates:
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
                for v in candidates[k]:
                    if v not in picked and len(picked) < target_count:
                        picked.append(v)
                result[k] = picked
            return result
        except Exception as e:
            print(f"Fallback selection triggered: {e}")

    result = {}
    for k in ["uk", "us", "jp"]:
        females = [v for v in candidates[k] if str(v["gender"]).lower() == "female"]
        males = [v for v in candidates[k] if str(v["gender"]).lower() == "male"]
        interleaved = []
        for i in range(max(len(females), len(males))):
            if i < len(females): interleaved.append(females[i])
            if i < len(males): interleaved.append(males[i])
        result[k] = interleaved[:target_count]
    return result


def synthesize_to_wav(voice_id: str, text: str, out_wav_path: str, lang_code: str = None):
    """Synthesize speech via ElevenLabs and standardize to 44.1kHz mono WAV."""
    if os.path.exists(out_wav_path):
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
        print(f"  [TTS Error] {voice_id}: {r.status_code}")
        return False

    tmp_mp3 = out_wav_path.replace(".wav", "_tmp.mp3")
    with open(tmp_mp3, "wb") as f:
        f.write(r.content)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", tmp_mp3,
        "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le",
        out_wav_path
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(tmp_mp3):
        os.remove(tmp_mp3)
    time.sleep(0.35)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    candidates = fetch_candidate_voices()
    selected = select_10_per_group(candidates, target_count=10)

    # Persist selected voice IDs so the speaker cohort remains identical across runs
    voice_registry_path = os.path.join(OUTPUT_DIR, "voice_registry.json")
    with open(voice_registry_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    cohorts = [
        ("UK-Native", selected["uk"], "en", "en"),
        ("US-Native", selected["us"], "en", "en"),
        ("JP-Accented", selected["jp"], "en", None),
        ("JP-Mora-Forced", selected["jp"], "ja", "ja"),
    ]

    manifest = []

    for sentence_key, script_dict in PASSAGES.items():
        print(f"\n========================================================")
        print(f"🎬 Processing Sentence: {sentence_key}")
        print(f"========================================================")

        for cohort_name, voice_list, text_key, lang_code in cohorts:
            cohort_dir = os.path.join(OUTPUT_DIR, sentence_key, cohort_name)
            os.makedirs(cohort_dir, exist_ok=True)
            text_to_read = script_dict[text_key]

            print(f"  Generating {cohort_name} ({len(voice_list)} voices)...")

            for idx, v in enumerate(voice_list, 1):
                safe_name = re.sub(r"[^a-zA-Z0-9]", "", v["name"])[:12] or f"v{idx}"
                filename = f"{cohort_name}_{idx:02d}_{safe_name}_{v['gender']}.wav"
                wav_path = os.path.join(cohort_dir, filename)

                ok = synthesize_to_wav(v["voice_id"], text_to_read, wav_path, lang_code=lang_code)
                if ok:
                    manifest.append({
                        "sentence_key": sentence_key,
                        "cohort": cohort_name,
                        "voice_id": v["voice_id"],
                        "name": v["name"],
                        "gender": v["gender"],
                        "wav_path": wav_path,
                        "reference_en": script_dict["en"]
                    })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nGeneration complete. Total audio files: {len(manifest)}. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()