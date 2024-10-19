CREATE TABLE generations(
    generation_id SERIAL PRIMARY KEY,
    generator_id INT REFERENCES generators(generator_id),
    model_id INT REFERENCES models(model_id),
    source_id INT REFERENCES sources(source_id),
    generated_notes JSON,  -- generated notes
    prompt_notes JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);
