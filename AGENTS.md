# Project Instructions

## Slack Completion Notification

For this project, send a short Slack DM to the user before the final response for
every turn that produces a substantive result, experiment update, file change,
commit, or decision.

Use the Slack plugin directly when it is available.

- User display name: `辛驕陽`
- Slack user ID: `U09HJQSFEJH`
- Previously verified self-DM channel: `D09HJQTH293`
- Test message link: `https://math-biol-bioeng.slack.com/archives/D09HJQTH293/p1778260865561659`

The Slack message should be short and should include:

1. Status: done / failed / blocked / still running.
2. Main conclusion or result.
3. Important artifact paths, result directories, or commit IDs.
4. Recommended next step when useful.

Keep the Slack DM private-summary style. Do not paste long logs, raw stack
traces, secrets, API keys, or large tables. If the Slack plugin is unavailable,
do not block the user's work; mention in the final response that the Slack
notification could not be sent.

For non-Codex or shell-only workflows, use `scripts/notify_slack_summary.py` as a
fallback notifier. The fallback script reads Slack credentials only from
environment variables.
