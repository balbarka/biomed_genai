CREATE TABLE IF NOT EXISTS {uc_name} (
    id               STRING NOT NULL,
    pmid             STRING,
    journal          STRING,
    title            STRING,
    year             STRING,
    citation         STRING,
    content          STRING,
    question         STRING,
    answer           STRING,
    train_test_split STRING,
    --CONSTRAINT pk_id PRIMARY KEY (id)
    pk_id            STRING
)
USING DELTA
TBLPROPERTIES (
  delta.checkpointPolicy = 'v2',
  delta.enableDeletionVectors = true,
  delta.enableRowTracking = true,
  delta.feature.deletionVectors = 'supported',
  delta.feature.rowTracking = 'supported',
  delta.feature.v2Checkpoint = 'supported',
  delta.enableChangeDataFeed = true
);