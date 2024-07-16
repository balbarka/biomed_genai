# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # PMC MetaData Sync
# MAGIC
# MAGIC **Objective**: This notebook will syncronize the metadata of articles in the PubMed Central &#174; ([PMC](https://www.ncbi.nlm.nih.gov/pmc/)) hosted in an S3 bucket with a delta table, **metadata_xml**, in Unity Catalog. This will provide a local reference for updates as well as provide a historical record of when publications were available for download.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Initialize biomed Configuration Class
# MAGIC %run ./config/setup_workflow $SHOW_TABLE=false $SHOW_GRAPHIC=true

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # `metadata_xml` Streaming Merge 
# MAGIC
# MAGIC For the ingest of PMC metadata data into `metadata_xml` we'll be using [Upsert from streaming queries using foreachBatch](https://docs.databricks.com/en/structured-streaming/delta-lake.html#upsert-from-streaming-queries-using-foreachbatch). However, there are quite a few configurations that go into this streaming process that we'll document below:
# MAGIC
# MAGIC  * **CloudFormat and Options** - We are going to [query a cloud storage object using autoloader](https://docs.databricks.com/en/query/streaming.html#query-data-in-cloud-object-storage-with-auto-loader). Thus, we will set the format to do this as `.format("cloudFiles")`. However, cloudFiles takes additional arguements, called *options*, to ensure that the csv source is read correctly. We'll set those configurations in the dict `readStream_options` and apply those configuration to the readStream using [`.options`](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrameReader.options.html).
# MAGIC  * **Load Source** - We'll use our notebook scope constant `PMC_SOURCE_METADATA_BUCKET` to [`.load`](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.streaming.DataStreamReader.load.html) from the PMC S3 bucket source.
# MAGIC  * **Select Columns** - While a common pattern is to ingest a data file raw and persist then run a second query that curates the source file, we are not going to do that for this metadata file. To avoid that unnecessary intermediate step, we are going to transform into our target table format using [`.select`](https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.DataFrame.select.html). This is the same select method that is availailable with regular pyspark dataframes. For readability, we are going to write our select columns as a list of pyspark columns in `readStream_columns` and pass as positional arguments into the `.select` method.
# MAGIC  * **WriteStream for each microbatch** - You are able to write a structured streaming dataframe by converting the streaming Dataframe to [`.writeStream`](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.writeStream.html#pyspark-sql-dataframe-writestream) and the use [`.foreachBatch`](https://spark.apache.org/docs/3.1.3/api/python/reference/api/pyspark.sql.streaming.DataStreamWriter.foreachBatch.html) to write each microbatch. `.foreachBatch` takes a function arguement, `upsert_metadata`. What's nice about this api design is that the function accepts microbatches as dataframes. Thus, the syntax that we use for streaming into delta tables is the same syntax that we use for batch merge jobs.
# MAGIC
# MAGIC **NOTE**: Gathering large arguments like this is not just helpful for readability, it also helps mitigate syntax errors by the developer. Since this streaming job was getting a little verbose, we applied the technique below.
# MAGIC
# MAGIC **NOTE**: The [trigger](https://docs.databricks.com/en/structured-streaming/triggers.html#configuring-incremental-batch-processing) setting of available sets the behavior to running as an incremental batch which makes sense because this file is updated only once a day.

# COMMAND ----------

# DBTITLE 0,Stream Configurations
from pyspark.sql import SparkSession, DataFrame, functions as F
from delta.tables import DeltaTable

# readStream Options:
readStream_options = {"cloudFiles.format": "csv",
                      "cloudFiles.allowOverwrites": "true",
                      "cloudFiles.schemaLocation": biomed.raw_metadata_xml.cp.path,
                      "header": "true"}

# pyspark dataframe columns to select from PubMed Metadata CloudFile
readStream_columns = [F.col("Key"),
                      F.col("ETag"),
                      F.col("Article Citation").alias("ArticleCitation"),
                      F.col("AccessionID"),
                      F.col("Last Updated UTC (YYYY-MM-DD HH:MM:SS)").cast("timestamp").alias("LastUpdated"),
                      F.col("PMID"),
                      F.col("License"),
                      F.col("Retracted"),
                      F.col("_metadata.file_path").alias("_file_path"),
                      F.col("_metadata.file_modification_time").alias("_file_modification_time"),
                      F.col("_metadata.file_size").alias("_file_size"),
                      F.current_timestamp().alias("_ingestion_timestamp"),
                      F.lit("PENDING").alias("status"),
                      F.lit(None).alias('volume_path')]

def upsert_metadata(microBatchOutputDF: DataFrame, batchId: int):
    tgt_df = biomed.raw_metadata_xml.dt.alias("tgt")
    tgt_df.merge(source = microBatchOutputDF.alias("src"),
                 condition = "src.AccessionID = tgt.AccessionID") \
        .whenMatchedUpdateAll(condition="src.LastUpdated > tgt.LastUpdated") \
        .whenNotMatchedInsertAll() \
        .execute()

# COMMAND ----------

# Necessary to allow Anonoymous Downloads from PMC
spark.conf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")

PMC_SOURCE_METADATA_BUCKET = f"s3://pmc-oa-opendata/oa_comm/xml/metadata/csv/"

spark.readStream.format("cloudFiles") \
                .options(**readStream_options) \
                .load(PMC_SOURCE_METADATA_BUCKET) \
                .select(*readStream_columns) \
    .writeStream.foreachBatch(upsert_metadata) \
                .trigger(availableNow=True) \
                .option("checkpointLocation", biomed.raw_metadata_xml.cp.path) \
                .queryName(f"query_{biomed.raw_metadata_xml.name}".replace('`','').replace('-','_')) \
                .start() \
                .awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # [OPTIONAL] Inspect `PUBMED_METADATA_TABLE`
# MAGIC
# MAGIC Let's check out the history of the most recent versions.

# COMMAND ----------

INSPECT_METADATA_HIST = False
if INSPECT_METADATA_HIST:
    hist = spark.sql(f"DESCRIBE HISTORY {biomed.raw_metadata_xml.uc_name}")
    display(hist)

# COMMAND ----------

INSPECT_METADATA = False
if INSPECT_METADATA:
    display(biomed.raw_metadata_xml.df)
