CREATE TABLE IF NOT EXISTS {uc_name} (
  request_id                  STRING NOT NULL,
  request                     STRING NOT NULL,
  expected_retrieved_context  STRING,
  expected_response           STRING NOT NULL,
  response                    STRING,
  retrieved_context           STRING,
  trace                       STRING,
  PRIMARY KEY (request_id)
)
USING delta
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2')