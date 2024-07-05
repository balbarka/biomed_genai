# Databricks notebook source
# MAGIC %run ./config/setup_workflow $SHOW_TABLE=false $SHOW_GRAPHIC=true

# COMMAND ----------

biomed.raw_articles_xml.path

# COMMAND ----------

# MAGIC %sh
# MAGIC head -n5 /Volumes/biomed_genai/raw/articles/all/xml/PMC11214836.xml

# COMMAND ----------

df = spark.read.option("rowTag", "article").format("xml").load("/Volumes/biomed_genai/raw/articles/all/xml/PMC11214836.xml")
# df = spark.read.option("rowTag", "article").format("xml").load(biomed.raw_articles_xml.path)

display(df.limit(1))

# COMMAND ----------

schema_copy = df.schema

# COMMAND ----------

print(str(df.schema))

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC ls -l /Volumes/biomed_genai/raw/articles/all/xml

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col

xml_path = "/Volumes/biomed_genai/raw/articles/all/xml/*.xml"

custom_schema = StructType([StructField('_article-type', StringType(), True), 
	 StructField('_dtd-version', DoubleType(), True),
	 StructField('_xml:lang', StringType(), True),
	 StructField('_xmlns:mml', StringType(), True),
	 StructField('_xmlns:xlink', StringType(), True),
	 StructField('back', StringType(), True),
	 StructField('body', StringType(), True),
	 StructField('front', StringType(), True),
	 StructField('processing-meta', StringType(), True),])

dat = spark.read.options(rowTag='article',
                         schema=custom_schema) \
                .xml(xml_path).limit(1)

dat.createGlobalTempView(dat)

#display(dat)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE biomed_genai.raw.xml_test (
# MAGIC     `_article-type` STRING,
# MAGIC     `_dtd-version` DOUBLE,
# MAGIC 	  `_xml:lang`  STRING,
# MAGIC     `_xmlns:mml`  STRING,
# MAGIC     `_xmlns:xlink` STRING,
# MAGIC     back STRING,
# MAGIC     body STRING,
# MAGIC     front STRING,
# MAGIC 	  `processing-meta` STRING)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC INSERT OVERWRITE biomed_genai.raw.xml_test SELECT * FROM dat;

# COMMAND ----------

display(dat.select(col("*"), col("body").alias('xxx')))

# COMMAND ----------

dat.columns

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

custom_schema = StructType([
    StructField("_id", StringType(), True),])

df = spark.read.options(rowTag='article').xml("/Volumes/biomed_genai/raw/articles/all/xml/PMC11214836.xml", schema = df.schema)

display(df.limit(5))

# COMMAND ----------


