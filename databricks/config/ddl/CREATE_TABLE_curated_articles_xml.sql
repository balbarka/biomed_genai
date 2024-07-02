CREATE TABLE IF NOT EXISTS {uc_name} (
  AccessionID STRING,
  ETag STRING,
  LastUpdated TIMESTAMP,
  PMID STRING,
  attrs MAP<STRING, STRING>,
  front STRING,
  body STRING,
  floats_group STRING,
  back STRING,
  processing_metadata STRING,
  _ingestion_timestamp TIMESTAMP,
  volume_path STRING
  )
USING DELTA
CLUSTER BY (AccessionID)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
)