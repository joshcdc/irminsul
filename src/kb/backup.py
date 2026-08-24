"""VACUUM INTO snapshots + rotation + recover — Phase 1.

`kb backup`: consistent single-file snapshot via VACUUM INTO (NOT raw `cp`,
which can miss WAL data). `kb recover <ts|path>`: whole-DB replace (destructive).
Rotation via `backup.keep`. Skeletons only in Phase 0.
"""

__all__ = ["backup", "recover"]
