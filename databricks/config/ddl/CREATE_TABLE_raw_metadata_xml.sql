CREATE TABLE IF NOT EXISTS {uc_name} (
  KEY STRING,  
  ETag STRING,
  ArticleCitation STRING,
  AccessionID STRING,
  LastUpdated TIMESTAMP,
  PMID STRING,
  License STRING,
  Retracted STRING,
  _file_path STRING,
  _file_modification_time TIMESTAMP,
  _file_size BIGINT,
  _ingestion_timestamp TIMESTAMP,
  Status STRING,
  volume_path STRING)
USING DELTA
CLUSTER BY (AccessionID)
TBLPROPERTIES (
  'delta.checkpointPolicy' = 'v2',
  'delta.enableDeletionVectors' = 'true',
  'delta.enableRowTracking' = 'true',
  'delta.feature.deletionVectors' = 'supported',
  'delta.feature.rowTracking' = 'supported',
  'delta.feature.v2Checkpoint' = 'supported',
  'delta.enableChangeDataFeed' = 'true')