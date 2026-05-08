# Slack Completion Notifications

This project uses a lightweight convention for notifying the user in Slack when
Codex finishes a meaningful turn.

## Preferred Path: Slack Plugin

When Codex has access to the Slack plugin, it should send a short DM before the
final chat response.

Target:

- User: `辛驕陽`
- User ID: `U09HJQSFEJH`
- Verified DM channel from the test send: `D09HJQTH293`

Suggested message shape:

```text
Codex 完成：<short task name>

状态：done / failed / blocked
结论：<one to three lines>
产物：<paths, result dirs, commit ids>
下一步：<optional>
```

This is intentionally a convention rather than a true global hook. The current
Codex runtime does not expose a repository-local event hook that automatically
fires after every model response. The project-level instruction in `AGENTS.md`
is therefore the main mechanism for future Codex turns in this repository.

## Fallback Path: Local Script

Use `scripts/notify_slack_summary.py` when running a shell workflow outside the
Slack plugin.

The script never stores Slack credentials. Configure one of these options in the
shell environment:

```bash
# Option A: Slack Web API bot/user token.
export SLACK_BOT_TOKEN='xoxb-...'
export SLACK_NOTIFY_USER_ID='U09HJQSFEJH'

# Optional: skip conversations.open and post to a known DM/channel directly.
export SLACK_NOTIFY_CHANNEL_ID='D09HJQTH293'
```

or:

```bash
# Option B: Incoming webhook configured to the desired Slack destination.
export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'
```

Example:

```bash
python3 scripts/notify_slack_summary.py \
  --title "NatView block diagnostic" \
  --message "状态：done\n结论：block split signal collapsed; shifted-null near real.\n产物：results/example/report.md"
```

Dry run:

```bash
python3 scripts/notify_slack_summary.py \
  --title "TRIBE v2 reading" \
  --message "状态：done\n结论：TRIBE v2 suggests long-context surface-space distillation." \
  --dry-run
```

The script appends a local notification log to
`logs/slack_notifications.jsonl`. The log contains message metadata and preview
text only, not Slack credentials.
