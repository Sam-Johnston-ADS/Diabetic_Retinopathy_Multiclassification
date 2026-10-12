"""
Start FastAPI Backend + Free Cloudflare HTTPS Tunnel
====================================================
Launches the local inference API and generates a public HTTPS URL with SSL
that can be directly called from your Vercel deployment without Mixed Content issues.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLOUDFLARED_BIN = PROJECT_ROOT / "backend_api" / "cloudflared"

if not CLOUDFLARED_BIN.exists():
    print(f"Error: {CLOUDFLARED_BIN} not found.")
    sys.exit(1)

print("=" * 70)
print("  Starting RetinaVision AI Inference Backend & HTTPS Tunnel")
print("=" * 70)

# 1. Start FastAPI / Uvicorn Server
print("[1/2] Launching FastAPI backend server on port 8000...")
uvicorn_cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "backend_api.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]
uvicorn_proc = subprocess.Popen(
    uvicorn_cmd,
    cwd=str(PROJECT_ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

# Wait a brief moment for FastAPI to initialize
time.sleep(2)
if uvicorn_proc.poll() is not None:
    print("Error: FastAPI failed to start:")
    print(uvicorn_proc.stdout.read())
    sys.exit(1)

# 2. Start Cloudflare Tunnel
print("[2/2] Opening secure Cloudflare HTTPS tunnel...")
tunnel_cmd = [str(CLOUDFLARED_BIN), "tunnel", "--url", "http://127.0.0.1:8000"]
tunnel_proc = subprocess.Popen(
    tunnel_cmd,
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
    bufsize=1,
)

tunnel_url = None
start_time = time.time()

while time.time() - start_time < 25:
    line = tunnel_proc.stderr.readline()
    if not line:
        time.sleep(0.1)
        continue
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        tunnel_url = match.group(0)
        break

if not tunnel_url:
    print("Warning: Could not automatically detect Cloudflare URL within 25s.")
    print("Tunnel stderr output:")
    print(tunnel_proc.stderr.read())
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 SUCCESS! YOUR LIVE SECURE BACKEND IS ONLINE")
print("=" * 70)
print(f"🔗 Public HTTPS Endpoint:  {tunnel_url}")
print(f"🩺 Health Check URL:        {tunnel_url}/health")
print("=" * 70)
print("👉 INSTRUCTIONS:")
print(f"1. Open your Vercel web app (https://diabetic-retinopathy-multiclassification.vercel.app)")
print(f"2. In the top-right corner, paste this URL into the 'API Endpoint' box:")
print(f"   {tunnel_url}")
print(f"3. Click the refresh button — the status dot will turn GREEN!")
print("4. Upload an image or select a sample case and click 'Run Diagnostic Analysis'.")
print("=" * 70)
print("Press Ctrl+C to stop the server and tunnel.\n")


def cleanup(sig, frame):
    print("\nShutting down backend and tunnel...")
    tunnel_proc.terminate()
    uvicorn_proc.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Keep running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    cleanup(None, None)
