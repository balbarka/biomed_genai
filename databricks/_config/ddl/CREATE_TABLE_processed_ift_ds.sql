CREATE TABLE biomed_genai.processed.ift_ds (
  messages ARRAY<STRUCT<role: STRING, content: STRING>>)
USING delta
TBLPROPERTIES (
  'delta.minReaderVersion' = '1',
  'delta.minWriterVersion' = '2')