# irminsul-io v1 — frozen promise about CLI bytes

Version: **v1** (frozen at Phase 0). Output-shape changes ship as **v2**; agents
parse against the contract version, never the evolving CLI. Independent of the
DB `user_version` — a schema migration doesn't bump the contract.

## Hard rules
- With `--json`, **stdout is ONLY JSON** — no logs, banners, or trailing noise.
- **Errors go to stderr only.**
- **Exit codes:** `0` ok · `1` user error · `2` infra error.
- **Stable field names & types per command** (shapes below).
- Result JSON is never truncated mid-stream; retrieval is capped (`--limit`).
- `doctor --json` **`ok` = structural integrity only** (config, schema, vec0/fts5
  present, no dup slugs). `stale_embeds` / `stale_fts` are informational readiness
  counts — nonzero does NOT fail `ok`; agents must read them before trusting
  search/vector retrieval.

## Commands (status: implemented | planned)
Machine-readable specs: `irminsul help --json` · `irminsul schema <cmd> --json` (ROOTLESS).

| command | status | output shape (--json) |
|---|---|---|
| `init --dir` | implemented | `{ok, repo_path, db_path, schema_version, config, dirs_created, git}` |
| `doctor` | implemented | `{ok, checks:{config, repo_path_exists, db_path_exists, vec0_loaded, schema_version, schema_ok, fts5, vec0_table, dup_slugs, stale_embeds, stale_fts}}` |
| `help` | implemented (rootless) | `{contract, version, rootless, exit_codes, commands}` |
| `schema <cmd>` | implemented (rootless) | one command's spec |
| `config [key [value]]` | implemented (rootless) | `{key, value, source}` or full dump |
| `put <slug> [--file]` | implemented | `{ok, slug, page_id, changed: created\|updated, chunks, type, title, created, updated}` |
| `get <slug>` | implemented | `{slug, type, title, created, updated, deleted_at, content}` |
| `list [--limit N] [--include-deleted]` | implemented | `{items:[{slug, type, title, updated, deleted_at}], count}` |
| `stats` | implemented | `{ok, pages, active, deleted, chunks, tags}` |
| `delete / restore <slug>` | implemented | `{ok, slug, deleted_at}` |
| `prune --older-than HOURS` | implemented | `{ok, pruned, older_than_hours}` |
| `backup [--keep N]` | implemented | `{ok, snapshot, size, kept, removed:[str]}` |
| `recover <ts\|path> --yes` | implemented | `{ok, from, to}` |
| `import <dir> [--dry-run]` | implemented (idempotent) | `{ok, dry_run, imported, skipped, errors:[{file, error}]}` |
| `export --dir PATH` | implemented | `{ok, exported, dir}` |
| `embed` | planned (Phase 2) | frozen at that phase |
| `graph` / `prune` | planned (Phase 3) | frozen at that phase |
| `rag` | planned (post-v1 Phase 4) | `{answer, citations:[{slug, score}]}` |

## Rootless commands
`help`, `schema`, `config` — run with no Irminsul configured, so a fresh agent can
discover the contract before any store exists.
