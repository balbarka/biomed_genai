CREATE TABLE IF NOT EXISTS {uc_name} (
    id        STRING,
    pmid      STRING,
    journal   STRING,
    title     STRING,
    year      STRING,
    citation  STRING,
    content   STRING)
-- BRAD TODO: add table properties for managed sync
USING DELTA;