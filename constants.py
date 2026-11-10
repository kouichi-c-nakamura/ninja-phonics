"""
constants.py: NinjaPhonetics の共通定数・辞書定義
"""
import numpy as np

# A1 プリセット一覧
PRESET_DIR = "presets"
PRESETS = [
    {"label": "Right (US)", "file": "right_us.wav"},
    {"label": "Right (UK)", "file": "right_uk.wav"},
    {"label": "Light (US)", "file": "light_us.wav"},
    {"label": "Light (UK)", "file": "light_uk.wav"},
    {"label": "White (UK)", "file": "white_uk.wav"},
    {"label": "ライト (JP)", "file": "raito_jp.wav"},
    {"label": "Really (US)", "file": "really_us.wav"},
    {"label": "Really (UK)", "file": "really_uk.wav"},
    {"label": "Very (US)", "file": "very_us.wav"},
    {"label": "Very (UK)", "file": "very_uk.wav"}
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