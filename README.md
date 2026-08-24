# kb — agent knowledge base (kb-io v1)

Command-driven, local-first knowledge base: an agent-tool contract with a
single-file SQLite engine (FTS5 keyword + sqlite-vec KNN, RRF hybrid, zero
network surface in v1 beyond the ZeroEntropy embedder).

- Contract: [`CONTRACT.md`](CONTRACT.md)
- Reference plan: `~/projects/brain/handoffs/2026-08-17-knowledge-base-plan.md`

## Quick start
```
uv run kb init --dir <vault-root>      # create store + config + scaffold
uv run kb doctor --json                # health check (schema v1, fts5+vec0)
uv run kb help --json                  # rootless, self-describing contract
```

## Layout
```
src/kb/   cli, config (~/.kb/config.toml bootstrap), db (runtime+migrations),
          schema.sql, pages/chunk/embed/vectors/search/graph/io/backup/doctor
tests/    pytest (Phase 0: migrate idempotence, config roundtrip, rootless CLI)
CONTRACT.md  kb-io v1 — frozen output/exit-code spec
```

Application data lives **outside** the vault (gbrain-style): `~/.kb/config.toml`,
`~/.kb/kb.db`, `~/.kb/backups/`. `repo.path` is the human vault only.
