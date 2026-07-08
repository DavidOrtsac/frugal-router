"""Narrated demo video with ElevenLabs TTS (voice: Brian).

Usage: ELEVENLABS_API_KEY=... python media/make_video_eleven.py
Output: media/frugalrouter_demo.mp4
"""

import os
import subprocess
import urllib.request
import json

HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_ID = "nPczCjzI2devNBz1zQrb"  # Brian: deep, resonant

NARRATION = [
    "TranscendiantRouter. A token efficient routing agent for Track One of the AMD Developer Hackathon, Act Two. Built by David Castro, team Transcendiant.",
    "The scoring for this track rewards frugality. Every answer is judged for accuracy, and everyone who passes the accuracy gate is ranked by one thing: the fewest Fireworks tokens spent. Local tokens are free. So the winner is whoever knows exactly when not to pay.",
    "The core insight: a local model's confidence is a bad lie detector, but its consistency is a good one. Ask it the same question several times. If the answers agree, it knows. If they scatter, it is guessing. And because local tokens are free, this lie detector costs nothing.",
    "The architecture has four steps. A rule based classifier sorts each task into one of eight categories, at zero cost. A small local model votes on each task. When the votes agree past a tuned threshold, the free answer ships. When they scatter, that single task escalates through Fireworks, to Gemma four by default, or a code specialist for code.",
    "Everything is engineered for the real grading box: four gigabytes of memory, two virtual CPUs, ten minutes, and a ten gigabyte image cap. The image is three point one gigabytes with weights baked in. A ratcheting time guard sheds sampling before the deadline is ever at risk, and a failed remote call falls back to a local answer, so no task ever returns blank.",
    "Every routing decision comes from measurement. A benchmark harness of two hundred twenty seven tasks, drawn from GSM eight K, Human Eval, and hand written questions, with a held out split that tuning never touched, and graders that actually execute generated code.",
    "Tuning uses recorded runs. Record the local model once, record each expert once, then replay every threshold combination offline, instantly. The result is a ladder of configurations, from seventy seven percent accuracy at zero tokens, up to ninety five percent.",
    "In the dress rehearsal, on a clone of the grading machine, TranscendiantRouter scored ninety two point five percent on held out tasks, finished in under six minutes, and spent a small fraction of the tokens that a send everything remote baseline burns.",
    "Gemma via Fireworks is the escalation brain, not a cameo. Six of the eight categories escalate to Gemma four. Model IDs resolve from the allowed models list at runtime. Nothing is hardcoded.",
    "TranscendiantRouter. Built solo, in four days, fully reproducible. The harness, the tuning tools, and the ladder all ship in the public repository. Thank you.",
]


def tts(text: str, out_path: str, api_key: str):
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128",
        method="POST",
        data=json.dumps({
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }).encode(),
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(out_path, "wb") as f:
            f.write(resp.read())


def main():
    api_key = os.environ["ELEVENLABS_API_KEY"]
    segments = []
    for i, text in enumerate(NARRATION, 1):
        mp3 = os.path.join(HERE, f"narr_{i:02d}.mp3")
        tts(text, mp3, api_key)
        slide = os.path.join(HERE, f"slide_{i:02d}.png")
        seg = os.path.join(HERE, f"seg_{i:02d}.mp4")
        subprocess.run([
            "/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-i", slide, "-i", mp3,
            "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac",
            "-pix_fmt", "yuv420p", "-vf", "scale=1920:1080",
            "-af", "apad=pad_dur=0.7", "-shortest",
            seg,
        ], check=True)
        segments.append(seg)
        print(f"segment {i}/10 done")

    concat_list = os.path.join(HERE, "concat.txt")
    with open(concat_list, "w") as f:
        for seg in segments:
            f.write(f"file '{seg}'\n")
    out = os.path.join(HERE, "frugalrouter_demo.mp4")
    subprocess.run([
        "/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", out,
    ], check=True)
    dur = subprocess.run(["/opt/homebrew/bin/ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", out],
                         capture_output=True, text=True).stdout.strip()
    print(f"wrote {out} ({float(dur):.0f}s)")


if __name__ == "__main__":
    main()
