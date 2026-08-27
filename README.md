# irminsul — agent knowledge base (irminsul-io v1)

Command-driven, local-first knowledge base: an agent-tool contract with a
single-file SQLite engine (FTS5 keyword + sqlite-vec KNN, RRF hybrid; exactly one
outbound call in v1 — the Voyage embed/rerank provider).

- Contract: [`CONTRACT.md`](CONTRACT.md)
- Reference plan: `~/projects/brain/handoffs/2026-08-17-irminsul-plan.md`

## Quick start
```
uv run irminsul init --dir <vault-root>      # create store + config + scaffold
uv run irminsul doctor --json                # health check (schema v2, fts5+vec0)
uv run irminsul doctor --fix --json          # self-heal a stale schema, then re-check
uv run irminsul migrate --json               # explicit schema upgrade (auto-runs on any command)
uv run irminsul help --json                  # rootless, self-describing contract
```

## Layout
```
src/irminsul/   cli, config (~/.irminsul/config.toml bootstrap), db (runtime+migrations),
          schema.sql, pages/chunk/embed/vectors/search/graph/io/backup/doctor
tests/    pytest (Phase 0: migrate idempotence, config roundtrip, rootless CLI)
CONTRACT.md  irminsul-io v1 — frozen output/exit-code spec
```

Application data lives **outside** the vault (gbrain-style): `~/.irminsul/config.toml`,
`~/.irminsul/irminsul.db`, `~/.irminsul/backups/`. `repo.path` is the human vault only.
