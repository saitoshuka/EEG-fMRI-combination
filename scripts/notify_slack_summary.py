#!/usr/bin/env python3
"""Send a short project-summary notification to Slack.

Preferred use in Codex is the Slack plugin. This script is a fallback for shell
jobs, cron jobs, or manual runs where a Slack token/webhook is available in the
environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_USER_ID = "U09HJQSFEJH"
DEFAULT_LOG_PATH = Path("logs/slack_notifications.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a concise Slack DM/project notification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Environment variables:
              SLACK_BOT_TOKEN          Slack token for Web API chat.postMessage.
              SLACK_NOTIFY_USER_ID     Slack user ID to DM. Default: U09HJQSFEJH.
              SLACK_NOTIFY_CHANNEL_ID  Optional known DM/channel ID.
              SLACK_WEBHOOK_URL        Optional incoming webhook alternative.

            Examples:
              python3 scripts/notify_slack_summary.py --title "Run done" --message "状态：done\\n结论：ok"
              python3 scripts/notify_slack_summary.py --message-file results/run/report.md --dry-run
            """
        ),
    )
    parser.add_argument("--title", default="Codex project update", help="Short notification title.")
    parser.add_argument("--message", help="Notification body. Use \\n for line breaks in shells.")
    parser.add_argument("--message-file", type=Path, help="Read notification body from a text file.")
    parser.add_argument("--channel-id", help="Slack channel or DM ID. Overrides SLACK_NOTIFY_CHANNEL_ID.")
    parser.add_argument("--user-id", default=os.environ.get("SLACK_NOTIFY_USER_ID", DEFAULT_USER_ID))
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Print payload without sending.")
    parser.add_argument("--no-log", action="store_true", help="Do not append logs/slack_notifications.jsonl.")
    return parser.parse_args()


def read_message(args: argparse.Namespace) -> str:
    if args.message and args.message_file:
        raise SystemExit("Use either --message or --message-file, not both.")
    if args.message_file:
        return args.message_file.read_text(encoding="utf-8").strip()
    if args.message:
        return args.message.replace("\\n", "\n").strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide --message, --message-file, or stdin content.")


def api_post(method: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"https://slack.com/api/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"Slack API {method} failed: {data}")
    return data


def webhook_post(webhook_url: str, text: str) -> dict[str, Any]:
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode("utf-8")
    if body.strip().lower() != "ok":
        raise RuntimeError(f"Slack webhook failed: {body}")
    return {"ok": True, "via": "webhook"}


def resolve_channel_id(args: argparse.Namespace, token: str) -> str:
    channel_id = args.channel_id or os.environ.get("SLACK_NOTIFY_CHANNEL_ID")
    if channel_id:
        return channel_id

    opened = api_post("conversations.open", token, {"users": args.user_id})
    channel = opened.get("channel") or {}
    channel_id = channel.get("id")
    if not channel_id:
        raise RuntimeError(f"Could not open DM for user {args.user_id}: {opened}")
    return channel_id


def write_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    body = read_message(args)
    text = f"{args.title}\n\n{body}".strip()
    record: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "title": args.title,
        "user_id": args.user_id,
        "preview": text[:500],
        "dry_run": args.dry_run,
    }

    try:
        if args.dry_run:
            print(json.dumps({"text": text, "target_user": args.user_id}, ensure_ascii=False, indent=2))
            record["status"] = "dry_run"
        elif os.environ.get("SLACK_WEBHOOK_URL"):
            result = webhook_post(os.environ["SLACK_WEBHOOK_URL"], text)
            record.update({"status": "sent", "result": result})
            print("Sent Slack notification via webhook.")
        else:
            token = os.environ.get("SLACK_BOT_TOKEN")
            if not token:
                raise RuntimeError(
                    "Set SLACK_BOT_TOKEN or SLACK_WEBHOOK_URL, or run with --dry-run. "
                    "Codex turns should normally use the Slack plugin instead."
                )
            channel_id = resolve_channel_id(args, token)
            result = api_post("chat.postMessage", token, {"channel": channel_id, "text": text})
            record.update(
                {
                    "status": "sent",
                    "channel_id": channel_id,
                    "ts": result.get("ts"),
                    "message": result.get("message", {}),
                }
            )
            print(f"Sent Slack notification to {channel_id}.")
    except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        record.update({"status": "failed", "error": str(exc)})
        if not args.no_log:
            write_log(args.log_path, record)
        print(f"Slack notification failed: {exc}", file=sys.stderr)
        return 1

    if not args.no_log:
        write_log(args.log_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
