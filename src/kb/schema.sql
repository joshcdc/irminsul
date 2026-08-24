PRAGMA user_version = 1;

-- pages: slug-keyed identity, mirrors gbrain page model
create table pages (
  id integer primary key,
  slug text unique not null,          -- identity; put upserts on this
  type text not null default 'note',
  title text,
  created text not null default (datetime('now')),
  updated text not null default (datetime('now')),
  deleted_at text,                    -- soft delete (72h recoverable)
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

-- keyword search (stdlib FTS5; verified present on 3.44.2)
create virtual table chunks_fts using fts5(chunk_text, content='chunks', content_rowid='id');

-- vector search (sqlite-vec vec0; 1280-d <= 4096 cap)
create virtual table chunk_embeddings using vec0(
  chunk_id integer primary key,
  embedding float[1280]
);

-- wikilink graph
create table links (src_chunk_id integer, dst_slug text, kind text default 'wikilink');

-- tags
create table tags (page_id integer references pages(id), tag text);

-- meta: RUNTIME FACTS ONLY (schema_version, embed_model) — user knobs live in ~/.kb/config.toml
create table meta (k text primary key, v text);
