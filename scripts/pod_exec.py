"""Remote command execution on a Jupyter-based GPU pod via the kernel API.

Runs shell commands on the pod through an IPython kernel (subprocess), so no
SSH is needed — only the Jupyter base URL and token.

Usage:
  python scripts/pod_exec.py "rocm-smi"
  python scripts/pod_exec.py --timeout 600 "bash pod_setup.sh"

Env:
  POD_BASE   e.g. https://host/instances/abc   (no trailing slash)
  POD_TOKEN  Jupyter token
"""

import argparse
import json
import os
import sys
import uuid

import urllib.request

import websocket


def _api(base: str, token: str, path: str, method: str = "GET", body: dict = None):
    req = urllib.request.Request(
        f"{base}/api/{path}{'&' if '?' in path else '?'}token={token}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or "{}")


def get_or_create_kernel(base: str, token: str) -> str:
    kernels = _api(base, token, "kernels")
    for k in kernels:
        if k.get("execution_state") != "dead":
            return k["id"]
    return _api(base, token, "kernels", method="POST", body={"name": "python3"})["id"]


def run(base: str, token: str, command: str, timeout: float) -> int:
    kernel_id = get_or_create_kernel(base, token)
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    ws = websocket.create_connection(
        f"{ws_base}/api/kernels/{kernel_id}/channels?token={token}", timeout=timeout)

    msg_id = uuid.uuid4().hex
    session = uuid.uuid4().hex
    code = (
        "import subprocess\n"
        f"_p = subprocess.run({command!r}, shell=True, capture_output=True, text=True, timeout={timeout})\n"
        "print(_p.stdout)\n"
        "if _p.stderr: print('[stderr]', _p.stderr)\n"
        "print('[exit]', _p.returncode)\n"
    )
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "frugal", "session": session,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell",
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False},
    }))

    exit_code = 0
    while True:
        msg = json.loads(ws.recv())
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        msg_type = msg.get("msg_type")
        content = msg.get("content", {})
        if msg_type == "stream":
            text = content.get("text", "")
            sys.stdout.write(text)
            for line in text.splitlines():
                if line.startswith("[exit] "):
                    exit_code = int(line.split()[1])
        elif msg_type == "error":
            print("\n".join(content.get("traceback", [])), file=sys.stderr)
            exit_code = 1
        elif msg_type == "status" and content.get("execution_state") == "idle":
            break
    ws.close()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    base = os.environ["POD_BASE"].rstrip("/")
    token = os.environ["POD_TOKEN"]
    return run(base, token, args.command, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
