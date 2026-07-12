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
    "The core idea: route by measured weakness, not by guesswork. A strong four billion parameter local model answers for free, and it turns out to be perfect at most categories. The router pays a remote expert for exactly the categories where measurement says the local model is weak, and nothing else.",
    "Every local answer has to prove itself before it ships. Math must end in an explicit final answer. Code must actually parse. Everything else is checked by self consistency voting. An answer that cannot prove itself escalates to the remote expert, kimi, through Fireworks.",
    "The grading environment is hostile, so nothing is assumed. At boot, a self healing probe sweeps base URLs, model I D forms, transports, and authentication shapes, and pins the first combination that actually answers. Model I Ds resolve from the allowed models list at runtime. Calls outside that list are impossible by construction.",
    "Time is an adversary too. A timeout scores zero, so timeouts are engineered out. A local generation that cannot finish inside the remaining budget never starts; it escalates instead. Results are checkpointed atomically after every task, so even a hard kill leaves a valid, scoreable output file.",
    "The cost adapts to the machine it lands on. On a fast grading box, the router spends around three hundred thirty tokens. On a degraded box under heavy load, it spends around two and a half thousand. Accuracy stays the same either way.",
    "Everything is verified by measurement. A two hundred seventy two task harness with a held out split, graders that actually execute generated code, and offline replay tuning. On the organizers own validation tasks, the router scored five out of five at zero Fireworks tokens. On judge scale rehearsals across a two x hardware speed envelope, nineteen out of nineteen, run after run.",
    "The router survived four days of infrastructure storms: scoring outages, queue stalls, and grading hardware running at half speed. Connection retries, a circuit breaker, and voted local fallback mean a failed remote call never returns a blank answer.",
    "TranscendiantRouter. Built solo, in five days, fully reproducible. The harness, the probe, the tuning tools, and the rehearsal protocol all ship in the public repository. Thank you.",
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
