# kb-io v1 — frozen promise about CLI bytes

Version: **v1** (frozen at Phase 0). Output-shape changes ship as **v2**; agents
parse against the contract version, never the evolving CLI. Independent of the
DB `user_version` — a schema migration doesn't bump the contract.

## Hard rules
- With `--json`, **stdout is ONLY JSON** — no logs, banners, or trailing noise.
- **Errors go to stderr only.**
- **Exit codes:** `0` ok · `1` user error · `2` infra error.
- **Stable field names & types per command** (shapes below).
- Result JSON is never truncated mid-stream; retrieval is capped (`--limit`).

## Commands (status: implemented | planned)
Machine-readable specs: `kb help --json` · `kb schema <cmd> --json` (ROOTLESS).

| command | status | output shape (--json) |
|---|---|---|
| `init --dir` | implemented | `{ok, repo_path, db_path, schema_version, config, dirs_created, git}` |
| `doctor` | implemented | `{ok, checks:{config, repo_path_exists, db_path_exists, vec0_loaded, schema_version, schema_ok, fts5, vec0_table, dup_slugs, stale_embeds}}` |
| `help` | implemented (rootless) | `{contract, version, rootless, exit_codes, commands}` |
| `schema <cmd>` | implemented (rootless) | one command's spec |
| `config [key [value]]` | implemented (rootless) | `{key, value, source}` or full dump |
| `put` / `get` / `list` / `stats` / `delete` / `restore` | planned (Phase 1) | frozen at that phase |
| `backup` / `recover` / `import` / `export` | planned (Phase 1) | frozen at that phase |
| `embed` | planned (Phase 2) | frozen at that phase |
| `graph` / `prune` | planned (Phase 3) | frozen at that phase |
| `rag` | planned (post-v1 Phase 4) | `{answer, citations:[{slug, score}]}` |

## Rootless commands
`help`, `schema`, `config` — run with no KB configured, so a fresh agent can
discover the contract before any store exists.
