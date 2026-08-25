PRAGMA user_version = 2;

-- pages: slug-keyed identity, mirrors gbrain page model
create table pages (
  id integer primary key,
  slug text unique not null,          -- identity; put upserts on this
  type text not null default 'note',
  title text,
  created text not null default (datetime('now')),
  updated text not null default (datetime('now')),
  deleted_at text,                    -- soft delete (recoverable window)
  content_md text not null
);

-- chunks: content split for embedding/search
create table chunks (
  id integer primary key,
  page_id integer not null references pages(id),
  seq integer not null,
  chunk_text text not null,
  embed_model text
);
create index idx_chunks_page on chunks(page_id);

-- keyword search (stdlib FTS5; verified present on 3.44.2).
-- EXTERNAL CONTENT: mirror of chunks. The three triggers keep it in sync;
-- no code path touches chunks_fts directly. Backfill existing rows with
-- `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')` (done in migrate v1->v2).
create virtual table chunks_fts using fts5(chunk_text, content='chunks', content_rowid='id');

-- vector search (sqlite-vec vec0; dim <= 4096 cap).
-- v2: float[1024] — provider switched ZeroEntropy zembed-1 (1280) -> Voyage voyage-4-large.
-- vec0 has NO UPDATE: stale recompute is delete+insert (see vectors.py).
create virtual table chunk_embeddings using vec0(
  chunk_id integer primary key,
  embedding float[1024]
);

-- wikilink graph
create table links (src_chunk_id integer, dst_slug text, kind text default 'wikilink');

-- tags
create table tags (page_id integer references pages(id), tag text);

-- meta: RUNTIME FACTS ONLY (schema_version, embed_model) — user knobs live in ~/.irminsul/config.toml
create table meta (k text primary key, v text);

-- FTS5 external-content sync triggers (added v2). INSERT/UPDATE/DELETE on
-- chunks push the same op to chunks_fts; 'delete' special row form is REQUIRED
-- for external-content tables (plain DELETE gives "disk image is malformed").
create trigger chunks_ai after insert on chunks begin
  insert into chunks_fts(rowid, chunk_text) values (new.id, new.chunk_text);
end;

create trigger chunks_ad after delete on chunks begin
  insert into chunks_fts(chunks_fts, rowid, chunk_text) values('delete', old.id, old.chunk_text);
end;

create trigger chunks_au after update on chunks begin
  insert into chunks_fts(chunks_fts, rowid, chunk_text) values('delete', old.id, old.chunk_text);
  insert into chunks_fts(rowid, chunk_text) values (new.id, new.chunk_text);
end;
