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
        "Every answer is judged for accuracy. Below the gate, you are out.",
        "Everyone who passes is ranked by ONE thing: fewest Fireworks tokens.",
        "Local model tokens count as ZERO.",
        "So the winner is whoever knows exactly when NOT to pay.",
    ], y)
    return img


@slide
def s3():
    img, d = new_slide()
    y = title_block(d, "The insight", "Route by measured weakness, not by vibes")
    bullets(d, [
        "A strong small model is FREE. Qwen3-4B answers most categories perfectly.",
        "Measure where it is weak, and pay the expert for exactly that.",
        "Confidence is checked per answer: an explicit final answer for math,",
        "    a compile check for code, self-consistency voting everywhere else.",
        "An answer that cannot prove itself escalates. Nothing else does.",
    ], y)
    return img


@slide
def s4():
    img, d = new_slide()
    y = title_block(d, "Architecture", "Classify, answer locally, escalate on doubt")
    bullets(d, [
        "1. Rule-based classifier sorts each task into 8 categories. Zero tokens.",
        "2. Qwen3-4B (llama.cpp, weights baked in) answers locally on 2 CPU cores.",
        "3. Answers that prove themselves ship free: math with a final-answer check,",
        "    code that parses, consistent votes elsewhere.",
        "4. Logical reasoning and unproven answers escalate to kimi-k2p7-code.",
    ], y)
    return img


@slide
def s5():
    img, d = new_slide()
    y = title_block(d, "Hostile environment", "The judge's proxy is discovered, never assumed")
    bullets(d, [
        "A boot-time probe sweeps base URLs, model ID forms, transports, and",
        "    auth shapes, then pins the first combination that answers.",
        "Model IDs resolve from ALLOWED_MODELS at runtime. Nothing hardcoded.",
        "Off-list calls are impossible by construction.",
        "If nothing answers, the router degrades to local-only and still finishes.",
    ], y)
    return img


@slide
def s6():
    img, d = new_slide()
    y = title_block(d, "Time as an adversary", "A timeout scores zero, so timeouts are impossible")
    bullets(d, [
        "Time-fit preemption: a local generation that cannot finish inside the",
        "    remaining budget never starts. It escalates instead.",
        "Results are checkpointed atomically after every task: even a hard kill",
        "    leaves a valid, scoreable results.json.",
        "Cost adapts to the host: fast box ~331 tokens, degraded box ~2,600.",
    ], y)
    return img


@slide
def s7():
    img, d = new_slide()
    y = title_block(d, "Measurement", "A 272-task harness, built before the router")
    bullets(d, [
        "GSM8K math, HumanEval code, and authored tasks across all 8 categories.",
        "Held-out split never touched by tuning. Generated code actually executes.",
        "Offline replay: record models once, re-score every routing policy instantly.",
        "Every threshold in the shipped image comes from a measured rehearsal.",
    ], y)
    return img


@slide
def s8():
    img, d = new_slide()
    y = title_block(d, "Results", "Rehearsed on a clone of the grading machine")
    bullets(d, [
        "5/5 on the organizers' official validation tasks, at zero Fireworks tokens.",
        "19/19 on judge-scale rehearsals across a 2x hardware speed envelope.",
        "97.5-100% strict accuracy on held-out tasks, run after run.",
        "Wall clock 231-380 seconds against a 600-second limit, at every speed.",
    ], y)
    return img


@slide
def s9():
    img, d = new_slide()
    y = title_block(d, "Resilience, proven the hard way", "Four days of infrastructure storms")
    bullets(d, [
        "Survived scoring outages, queue stalls, and slow grading hardware.",
        "Connection retries, a half-open circuit breaker, and voted local fallback",
        "    mean a failed remote call never returns a blank answer.",
        "Gemma-ready: the probe prefers Gemma when the environment serves it;",
        "    the scoring proxy never exposed a deployment, verified empirically.",
    ], y)
    return img


@slide
def s10():
    img, d = new_slide()
    y = title_block(d, "", "Built solo, in five days, in public")
    bullets(d, [
        "AMD GPU pod + GCP for development. llama.cpp + Qwen3 + Fireworks.",
        "Reproducible: harness, probe, tuning tools, and rehearsal protocol",
        "    all ship in the repo.",
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
