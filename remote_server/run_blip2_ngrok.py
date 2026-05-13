from __future__ import annotations

import os
import subprocess
import sys
import time

from pyngrok import ngrok


PORT = int(os.getenv("BLIP2_PORT", "8001"))


def main() -> None:
    token = os.getenv("NGROK_AUTHTOKEN")
    if token:
        ngrok.set_auth_token(token)

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "remote_server.blip2_service:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    process = subprocess.Popen(command)
    time.sleep(8)

    tunnel = ngrok.connect(PORT, "http")
    print(f"BLIP-2 service public URL: {tunnel.public_url}", flush=True)
    print("Use this as BLIP2_SERVER_URL for the Streamlit app.", flush=True)

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    finally:
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
