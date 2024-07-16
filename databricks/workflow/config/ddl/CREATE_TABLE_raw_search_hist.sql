CREATE TABLE IF NOT EXISTS {uc_name} (
  keyword STRING,
  min_dte STRING,
  max_dte STRING)
USING delta
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2')