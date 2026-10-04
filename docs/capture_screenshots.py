"""One-off script: drives the running Streamlit demo (localhost:8501) with Playwright
and saves real screenshots of both pages into docs/screenshots/. Not part of the test
suite or the shipped app; run manually after `streamlit run app/streamlit_app.py`.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_IMAGE = ROOT / "data" / "raw" / "mvsa_single" / "extracted" / "data" / "1.jpg"

SAMPLE_TEXT = "Feeling so alone lately, nothing seems to make sense anymore #tired"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1600})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
    page.wait_for_selector("text=Multimodal Sentiment Analysis", timeout=30000)

    # --- Predict page, empty state ---
    page.screenshot(path=str(OUT / "01_predict_empty.png"), full_page=True)

    # Fill text + upload image
    page.get_by_placeholder("Type or paste a social-media-style post...").fill(SAMPLE_TEXT)
    page.set_input_files("input[type='file']", str(SAMPLE_IMAGE))
    page.wait_for_timeout(1500)

    page.get_by_role("button", name="Analyze").click()
    # Model loading + classifier refit can take a while on first run (CPU).
    page.wait_for_selector("text=Modality comparison", timeout=240000)
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "02_predict_result.png"), full_page=True)

    # --- Dashboard page ---
    page.get_by_text("Research Dashboard", exact=True).click()
    page.wait_for_selector("text=Model comparison", timeout=30000)
    page.wait_for_timeout(1000)
    page.screenshot(path=str(OUT / "03_dashboard.png"), full_page=True)

    browser.close()

print("Saved screenshots to", OUT)
