"""CLI: load order, boot server, open ngrok, place call, exit when call ends."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from twilio.rest import Client  # pyright: ignore[reportMissingTypeStubs]

from agent.pipeline.builder import run_pipeline_for_call
from agent.runtime import close_all_tunnels, open_tunnel, place_call
from agent.safety import assert_destination_allowed
from agent.server import build_app
from agent.types import Order


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agent")
    p.add_argument("order_file", type=Path, help="Path to order JSON")
    p.add_argument("--to", required=True, help="Destination phone number, E.164 (+1...)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code."""
    load_dotenv()
    args = _parse_args(argv or sys.argv[1:])

    order = Order.model_validate(json.loads(args.order_file.read_text()))
    assert_destination_allowed(args.to)

    port = int(os.environ.get("PORT", 8000))
    public_url = open_tunnel(port)

    app = build_app(public_url=public_url)
    app.state.twilio_account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    app.state.twilio_auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    app.state.order = order
    app.state.pipeline_runner = run_pipeline_for_call

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    server_thread.start()

    try:
        client = Client(app.state.twilio_account_sid, app.state.twilio_auth_token)
        sid = place_call(
            client,
            to=args.to,
            from_=os.environ["TWILIO_FROM_NUMBER"],
            voice_url=f"{public_url}/voice",
        )
        print(json.dumps({"event": "call_placed", "call_sid": sid, "order": order.customer_name}))

        while server_thread.is_alive():
            server_thread.join(timeout=1)
    finally:
        close_all_tunnels()

    return 0
