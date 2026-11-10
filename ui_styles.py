"""
ui_styles.py: ブランドアイコン & 包括的ダークテーマCSS定義
"""
import os
import base64

def get_shuriken_icon_html(size_px: int = 28) -> str:
    for fname in ["pink shuriken 32.png", "pink shuriken 640.jpg", "assets/pink shuriken 32.png"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = "image/png" if fname.endswith(".png") else "image/jpeg"
            return (
                f'<img src="data:{mime};base64,{b64}" width="{size_px}" height="{size_px}" '
                f'style="filter: hue-rotate(190deg) saturate(0.28) brightness(1.65) contrast(1.15) '
                f'drop-shadow(0 0 6px rgba(186, 230, 253, 0.55)); vertical-align: middle; border-radius: 4px;" />'
            )
    return (
        f'<svg width="{size_px}" height="{size_px}" viewBox="0 0 64 64" style="filter: drop-shadow(0 0 5px rgba(186, 230, 253, 0.5)); vertical-align: middle;">'
        f'<defs><linearGradient id="iceBlade" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="#F8FAFC"/><stop offset="55%" stop-color="#BAE6FD"/><stop offset="100%" stop-color="#38BDF8"/>'
        f'</linearGradient></defs>'
        f'<path d="M32 2 C36 18 46 24 62 32 C46 36 40 46 32 62 C28 46 18 40 2 32 C18 28 24 18 32 2 Z" fill="url(#iceBlade)" stroke="#E0F2FE" stroke-width="1.5"/>'
        f'<circle cx="32" cy="32" r="7" fill="#090D16" stroke="#BAE6FD" stroke-width="2"/>'
        f'</svg>'
    )

SHURIKEN_ICON_MAIN = get_shuriken_icon_html(28)
SHURIKEN_ICON_SM = get_shuriken_icon_html(16)

APP_CUSTOM_CSS = """<style>
:root {
    --ice-white: #F0F9FF;
    --ice-cyan: #BAE6FD;
    --accent-blue: #38BDF8;
    --bg-dark: #090D16;
    --bg-card: #0F172A;
    --border-muted: #1E293B;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #090D16 !important;
    color: #F8FAFC !important;
}
header[data-testid="stHeader"] {
    height: 1.8rem !important;
    background: transparent !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 0.8rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}
.brand-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0 10px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid #1E293B;
}
.brand-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    margin: 0;
}
.brand-sub {
    font-size: 0.72rem;
    font-weight: 700;
    color: #BAE6FD;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(186, 230, 253, 0.35);
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.05em;
}
.stButton button {
    background-color: #0F172A !important;
    color: #94A3B8 !important;
    border: 1px solid #1E293B !important;
    border-bottom: 2px solid #1E293B !important;
    font-weight: 600 !important;
    padding: 6px 10px !important;
    border-radius: 6px 6px 2px 2px !important;
    font-size: 0.86rem !important;
    transition: all 0.15s ease !important;
}
.stButton button:hover {
    background-color: #1E293B !important;
    color: #F0F9FF !important;
    border-color: #334155 !important;
}
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {
    background-color: #172554 !important;
    color: #F0F9FF !important;
    border-color: #1E3A8A !important;
    border-bottom: 2.5px solid #BAE6FD !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(56, 189, 248, 0.15) !important;
}
div[data-testid="stAudioInput"] {
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 4px 12px;
    margin-bottom: 0.4rem;
}
div[data-testid="stFileUploader"] {
    background: #0F172A !important;
    border: 1px dashed #334155 !important;
    border-radius: 8px !important;
    padding: 6px !important;
    margin-bottom: 0.4rem !important;
}
div[data-testid="stTextArea"] textarea {
    background-color: #090D16 !important;
    color: #F8FAFC !important;
    border: 1px solid #1E293B !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    line-height: 1.5 !important;
}
.sec-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.92rem;
    font-weight: 700;
    color: #F0F9FF;
    margin-bottom: 8px;
}
</style>"""