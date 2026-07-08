"""Render the submission slide deck (PDF), cover image, and placeholder video
frames. Pure Pillow, no browser needed.

Usage: python media/make_slides.py
Outputs: media/slides.pdf, media/cover.png, media/slide_*.png
"""

import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (15, 18, 32)
FG = (240, 242, 248)
DIM = (150, 158, 178)
RED = (237, 28, 36)      # AMD red
ORANGE = (255, 122, 26)  # Fireworks orange

FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_PATH, size, index=1 if bold else 0)


def new_slide():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 14, W, H], fill=RED)
    return img, d


def title_block(d, kicker, title, y=120):
    if kicker:
        d.text((120, y), kicker.upper(), font=font(34, True), fill=ORANGE)
        y += 70
    d.text((120, y), title, font=font(72, True), fill=FG)
    return y + 130


def bullets(d, items, y, size=40, gap=36, x=120):
    for item in items:
        d.ellipse([x, y + size * 0.35, x + 14, y + size * 0.35 + 14], fill=RED)
        d.text((x + 44, y), item, font=font(size), fill=FG)
        y += size + gap
    return y


SLIDES = []


def slide(fn):
    SLIDES.append(fn)
    return fn


@slide
def s1():
    img, d = new_slide()
    d.text((120, 330), "TranscendiantRouter", font=font(140, True), fill=FG)
    d.text((124, 520), "Every token earned. Most answers free.", font=font(54), fill=DIM)
    d.text((124, 720), "AMD Developer Hackathon ACT II  |  Track 1: General-Purpose AI Agent",
           font=font(36), fill=FG)
    d.text((124, 780), "Team Transcendiant  |  solo build by David Castro", font=font(36), fill=DIM)
    return img


@slide
def s2():
    img, d = new_slide()
    y = title_block(d, "The game", "Scoring rewards the frugal, not the flashy")
    bullets(d, [
        "Every answer is judged for accuracy. Below a hidden threshold, you are out.",
        "Everyone who passes is ranked by ONE thing: fewest Fireworks tokens.",
        "Local model tokens count as ZERO.",
        "So the winner is whoever knows exactly when NOT to pay.",
    ], y)
    return img


@slide
def s3():
    img, d = new_slide()
    y = title_block(d, "The insight", "Free tokens can buy calibration")
    bullets(d, [
        "A local model's confidence is a bad lie detector.",
        "But its CONSISTENCY is a good one.",
        "Ask it the same question several times. Agreement predicts correctness.",
        "Local sampling is free under the rules, so the lie detector costs nothing.",
    ], y)
    return img


@slide
def s4():
    img, d = new_slide()
    y = title_block(d, "Architecture", "Classify, vote, then decide who answers")
    bullets(d, [
        "1. Rule-based classifier sorts each task into 8 categories. Zero tokens.",
        "2. Qwen3-1.7B (llama.cpp, weights baked in) votes on each task.",
        "3. Votes agree past a tuned threshold: ship the free answer.",
        "4. Votes scatter: escalate that one task via Fireworks. Gemma 4 by default,",
        "    a code specialist for code.",
    ], y)
    return img


@slide
def s5():
    img, d = new_slide()
    y = title_block(d, "Engineering for the grading box", "4GB RAM. 2 vCPUs. 10 minutes. 10GB image.")
    bullets(d, [
        "3.1GB image with model weights baked in. Boots in seconds.",
        "Per-category token caps and adaptive vote counts sized for 2 CPU cores.",
        "A ratcheting time guard sheds sampling before the deadline is ever at risk.",
        "Dead remote model? Local fallback. A task can never return blank.",
        "Off-list model calls are impossible by construction.",
    ], y)
    return img


@slide
def s6():
    img, d = new_slide()
    y = title_block(d, "Measurement", "A 227-task benchmark harness, built first")
    bullets(d, [
        "GSM8K math, HumanEval code, and authored tasks across all 8 categories.",
        "Held-out split never touched by tuning.",
        "Executable grading: generated code actually runs against test suites.",
        "Every routing threshold comes from recorded runs, not intuition.",
    ], y)
    return img


@slide
def s7():
    img, d = new_slide()
    y = title_block(d, "The ladder", "One recording, every strategy")
    bullets(d, [
        "Record the local model once, record each expert once.",
        "Then replay every threshold combination offline, instantly.",
        "Result: a ladder from 77% accuracy at 0 tokens to 95% at 47K.",
        "Submissions walk down the ladder until the accuracy gate bites.",
    ], y)
    return img


@slide
def s8():
    img, d = new_slide()
    y = title_block(d, "Results", "Dress rehearsal on a clone of the grading machine")
    bullets(d, [
        "92.5% accuracy on held-out tasks it had never seen.",
        "5 minutes 58 seconds end to end, inside 4GB and 2 vCPUs.",
        "Most answers free. Escalations cost ~6.6K tokens per 40 tasks.",
        "Send-everything-remote baseline: ~50K+ tokens for similar accuracy.",
    ], y)
    return img


@slide
def s9():
    img, d = new_slide()
    y = title_block(d, "Gemma via Fireworks", "The escalation brain, not a cameo")
    bullets(d, [
        "Six of eight categories escalate to Gemma 4 31B through Fireworks.",
        "Model IDs resolve from ALLOWED_MODELS at runtime. Nothing hardcoded.",
        "Local Gemma was tested and ruled out honestly: the 4GB grading box",
        "    cannot hold the smallest Gemma 4 plus an agent.",
        "Gemma answers directly, with no hidden reasoning tokens to pay for.",
    ], y)
    return img


@slide
def s10():
    img, d = new_slide()
    y = title_block(d, "", "Built solo, in four days, in public")
    bullets(d, [
        "AMD GPU pod + GCP for development. llama.cpp + Qwen3 + Gemma + Fireworks.",
        "Reproducible: harness, tuning tools, and ladder ship in the repo.",
        "github.com/DavidOrtsac/frugal-router  |  MIT license",
    ], y)
    d.text((120, 900), "TranscendiantRouter  |  Team Transcendiant", font=font(32), fill=DIM)
    return img


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    images = []
    for i, fn in enumerate(SLIDES, 1):
        img = fn()
        img.save(os.path.join(out_dir, f"slide_{i:02d}.png"))
        images.append(img)
    images[0].save(os.path.join(out_dir, "slides.pdf"), save_all=True,
                   append_images=images[1:], resolution=96)
    cover = images[0].resize((1200, 675))
    cover.save(os.path.join(out_dir, "cover.png"))
    print(f"wrote {len(images)} slides -> slides.pdf, cover.png")


if __name__ == "__main__":
    main()
