CREATE TABLE sources (
    source_id SERIAL PRIMARY KEY,
    source JSON,
    notes JSON,
);

CREATE UNIQUE INDEX unique_source
ON sources (source::text);
