-- Enable pgvector
create extension if not exists vector;

-- Create documents table
create table if not exists doc (
    id bigserial primary KEY,
    content TEXT,
    embedding vector(1536)
);

-- Create match document function
create or replace function match_documents (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    doc.id,
    doc.content,
    1 - (doc.embedding <=> query_embedding) as similarity
  from doc
  where 1 - (doc.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;