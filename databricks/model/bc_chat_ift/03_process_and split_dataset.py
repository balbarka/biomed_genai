# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is third in a series that **processes synthetic data for subsequent chat completion Fine Tuning** (FT).
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. Merge seed data (from NB 1) with evolved data (from NB 2)
# MAGIC 2. Re-format the merged data into a messages json array required for chat models
# MAGIC 3. Split the data into train and test sets for finetuning and evaluation respectively

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import StringType
from pyspark.sql.functions import lit, udf, split, size, col, coalesce, pandas_udf
from typing import List, Dict
from _setup.params import *

# COMMAND ----------

seed_table_name = "yen.syn_data_gen.seed"
evolved_table_name = "yen.syn_data_gen.evolved"

data_table_name = "yen.syn_data_gen.data"
train_table_name = "yen.syn_data_gen.train"
test_table_name = "yen.syn_data_gen.test"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read seed data and evolved data

# COMMAND ----------

seed_df = spark.table(seed_table_name)
evolved_df = spark.table(evolved_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Merge seed data and evolved data

# COMMAND ----------

merged_df = seed_df \
    .withColumn('question_new', lit(None).cast(StringType())) \
    .withColumn('answer_new', lit(None).cast(StringType())) \
    .withColumn('prompt', lit(None).cast(StringType())) \
    .select(*evolved_df.columns) \
    .union(evolved_df) \
    .dropDuplicates()
display(merged_df)

# COMMAND ----------

merged_df.count(), seed_df.count(), evolved_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Re-format context, Q&A into a `messages` json required of chat models

# COMMAND ----------

data = merged_df \
    .withColumnRenamed('prompt', 'evolve') \
    .withColumn('q', coalesce(col('question_new'), col('question'))) \
    .withColumn('a', coalesce(col('answer_new'), col('answer')))
data = data \
    .withColumn("messages", make_chat_udf(data.context, data.q, data.a))
display(data.orderBy(["id","evolve"]).select(["id", "evolve","question_new", "question", "q", "messages"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Split into train and test sets
# MAGIC Test set should contain the original seed and all its evolved variants

# COMMAND ----------

data.select('question').distinct().count()

# COMMAND ----------

test_questions = [row['question'] for row in data.select('question').distinct().sample(0.2).collect()]
test_questions

# COMMAND ----------

test = data.filter(data.question.isin(test_questions)) \
    .select(["messages"]) \
    .dropDuplicates()
train = data.filter(~data.question.isin(test_questions)) \
    .select(["messages"]) \
    .dropDuplicates()
test.count(), train.count()

# COMMAND ----------

display(test)

# COMMAND ----------

train.write.option("overwriteSchema", "true").saveAsTable(train_table_name, mode="overwrite")
test.write.option("overwriteSchema", "true").saveAsTable(test_table_name, mode="overwrite")

# COMMAND ----------


