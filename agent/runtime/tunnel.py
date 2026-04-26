"""ngrok tunnel boot/teardown helpers."""

from __future__ import annotations

import os

from pyngrok import conf, ngrok  # pyright: ignore[reportMissingTypeStubs]


def open_tunnel(port: int) -> str:
    """Open an ngrok HTTPS tunnel to localhost:port; return the public URL."""
    auth = os.environ.get("NGROK_AUTHTOKEN")
    if auth:
        conf.get_default().auth_token = auth
    tunnel = ngrok.connect(str(port), "http", bind_tls=True)
    return str(tunnel.public_url)


def close_all_tunnels() -> None:
    """Close every active ngrok tunnel."""
    ngrok.kill()
