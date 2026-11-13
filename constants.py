"""
constants.py: NinjaPhonetics の共通定数・辞書定義
"""
import numpy as np

# A1 プリセット一覧
PRESET_DIR = "presets"
PRESETS = [
    {
        "label": "Right (US)",
        "file": "right_us.wav",
        "title": "Right /raɪt/ (US)",
        "phonetics": "/r/ のくびれ (谷)",
        "desc": "語頭でF3が急降下してF2に接近し、差分ライン(Δ)が1000Hz破線より深く潜り込みます。舌の両側面を上顎の奥歯に密着させることでこの深い谷が生まれます。"
    },
    {
        "label": "Right (UK)",
        "file": "right_uk.wav",
        "title": "Right /raɪt/ (UK)",
        "phonetics": "/r/ の狭窄",
        "desc": "語頭でF3が急降下して1000Hz基準線を下回ります。その後の二重母音 /aɪ/ への移行部でF2が右肩上がりに上昇します。"
    },
    {
        "label": "Light (US)",
        "file": "light_us.wav",
        "title": "Light /laɪt/ (US)",
        "phonetics": "Dark L（軟口蓋化）による低F2・高Δ",
        "desc": "米語の語頭 /l/ は奥舌が引かれるため（Dark L傾向）、開始部で F2（緑）が 1200Hz 付近と低く、F3（青）との差分（Δ）が 2500Hz 以上に大きく開くのが特徴です。"
    },
    {
        "label": "Light (UK)",
        "file": "light_uk.wav",
        "title": "Light /laɪt/ (UK)",
        "phonetics": "F3高域維持 (Clear L)",
        "desc": "語頭の /l/ でF3は2700〜2800Hzの高域にとどまり、差分(Δ)は1000Hzを割らずに推移します。"
    },
    {
        "label": "White (UK)",
        "file": "white_uk.wav",
        "title": "White /waɪt/ (UK)",
        "phonetics": "円唇によるF2低下",
        "desc": "語頭で唇を丸めるためF2が一時的に下がりますが、F3は下がらないため、差分ラインはRのようなくびれになりません。"
    },
    {
        "label": "ライト (JP)",
        "file": "raito_jp.wav",
        "title": "日本語「ライト」/ɾaɪto/",
        "phonetics": "タップ破裂と語尾母音付加",
        "desc": "「ラ」で瞬間的な弾き音（破裂）が起き、語尾には英語の /t/ にはない母音「オ」のホルマントが明瞭に残ります。"
    },
    {
        "label": "Really (US)",
        "file": "really_us.wav",
        "title": "Really /ˈrɪli/ ~ /ˈriːli/ (US)",
        "phonetics": "/r/ の深谷から /l/ の天井へのジャンプ",
        "desc": "語頭の /r/ で青（F3）が緑（F2）へ沈み込み、黄色（Δ）が 500Hz 前後まで深く潜ります。そこから後半の /l/ に向かって F3 が 3000Hz 超へ跳ね上がるコントラストが特徴です。"
    },
    {
        "label": "Really (UK)",
        "file": "really_uk.wav",
        "title": "Really /ˈrɪəli/ (UK)",
        "phonetics": "センタリング二重母音を経由する滑らかな渡り",
        "desc": "前半の /r/ でしっかり 1000Hz 破線を潜る谷を作った後、イギリス英語特有の曖昧母音 /ə/ を挟みながら、後半の /l/ に向かって滑らかに F3 が立ち上がります。"
    },
    {
        "label": "Very (US)",
        "file": "very_us.wav",
        "title": "Very /ˈveri/ (US)",
        "phonetics": "有声摩擦 /v/ の振動持続 ＋ r-colored母音のくびれ",
        "desc": "語頭は有声摩擦 /v/ の規則的な声帯振動が続き、破裂音（b）のようなスパイクがありません。続く母音〜/r/ で F3 が引きずり下ろされ、Δ が 500Hz 未満の深い谷を作ります。"
    },
    {
        "label": "Very (UK)",
        "file": "very_uk.wav",
        "title": "Very /ˈveri/ (UK)",
        "phonetics": "/v/ から短母音を経て短く鋭い /r/ へ",
        "desc": "米語に比べて母音のR音性化が少なく、谷の幅（持続時間）がコンパクトで引き締まっています。"
    }
]

# E2 ベンチマーク定義
E2_PASSAGE_BENCHMARKS = {
    "S1_stress": {
        "category": "drill",
        "x_metric": "pct_v",
        "title": "短文ドリル 1: 語彙強弱拍 (Alternating Stress & Reduction)",
        "target_text": "A photograph of a famous photographer was taken in the station.",
        "phonetics_desc": "photograph と photographer の強弱母音コントラストを測定。弱母音(/ə/)の鋭い短縮が不可欠です。",
        "mu": np.array([47.01, 58.61]),
        "cov": np.array([[1.45, 3.12], [3.12, 45.0]]),
        "x_range": (41.0, 58.0),
        "y_range": (35.0, 110.0),
        "ref_audio": "benchmarks/S1_stress/UK-Native/UK-Native_01_ArcherConver_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "S2_clusters": {
        "category": "drill",
        "x_metric": "pct_v",
        "title": "短文ドリル 2: 子音連結保持 (Consonant Clusters)",
        "target_text": "Strict instructors asked complex questions about the project tests.",
        "phonetics_desc": "連続子音に不要な母音を挟まずに子音を保持できるかを診断します。",
        "mu": np.array([35.85, 59.12]),
        "cov": np.array([[1.80, 2.85], [2.85, 52.0]]),
        "x_range": (30.0, 46.0),
        "y_range": (35.0, 120.0),
        "ref_audio": "benchmarks/S2_clusters/US-Native/US-Native_01_RogerLaidBac_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "S3_connected": {
        "category": "drill",
        "x_metric": "pct_v",
        "title": "短文ドリル 3: 機能語弱化 (Connected Speech Weakening)",
        "target_text": "Can you give him a cup of water from the kitchen at lunch?",
        "phonetics_desc": "機能語を極小化し、息の流れに乗せるテストです。",
        "mu": np.array([45.42, 45.18]),
        "cov": np.array([[1.60, 1.95], [1.95, 42.0]]),
        "x_range": (38.0, 54.0),
        "y_range": (30.0, 140.0),
        "ref_audio": "benchmarks/S3_connected/UK-Native/UK-Native_01_ArcherConver_male.wav",
        "bm_csv": "benchmarks/benchmark_results_all.csv"
    },
    "P1_northwind": {
        "category": "passage",
        "x_metric": "rcf",
        "title": "総合アセスメント 1: 北風と太陽 (The North Wind and the Sun 全文113語)",
        "target_text": "The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should be considered stronger than the other. Then the North Wind blew with all his might, but the more he blew, the more closely did the traveler fold his cloak around him, and at last the North Wind gave up the attempt. Then the Sun shone out warmly, and immediately the traveler took off his cloak. And so the North Wind was obliged to confess that the Sun was the stronger of the two.",
        "phonetics_desc": "ポーズ非感受性の2D軸 (内容語/機能語母音比 R_CF × 弱母音持続時間 Schwa P75) により、客観的なストレス拍を評価します。",
        "mu": np.array([1.45, 52.26]),
        "cov": np.array([[0.0068, -0.165], [-0.165, 16.50]]),
        "x_range": (0.7, 1.85),
        "y_range": (35.0, 105.0),
        "ref_audio": "benchmarks/P1_northwind/UK-Native/UK-Native_01_Archer.wav",
        "bm_csv": "benchmarks/P1_northwind/northwind_benchmark_results.csv"
    }
}

DIAGNOSTIC_FUNCTION_WORDS = {
    "to", "for", "at", "of", "in", "from", "with", "into", "through", "about",
    "and", "that", "than", "but", "as", "if", "or", "so",
    "was", "were", "should", "could", "would", "can", "his", "him", "them", "us"
}

ALL_FUNCTION_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "from", "with", "by", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against", "during", "without",
    "before", "under", "around", "among", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that", "these",
    "those", "who", "whom", "whose", "which", "what", "where", "when", "why", "how", "am", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "and", "but", "or", "so", "than", "off", "along"
}

VOWEL_SET = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}