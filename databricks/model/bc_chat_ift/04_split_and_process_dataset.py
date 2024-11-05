# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is third in a series that prepares synthetic data for Instruction Fine Tuning (IFT).
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. Merge seed data (from NB 1) with evolved data (from NB 2)
# MAGIC 2. Split the data into train and test sets for finetuning and evaluation respectively
# MAGIC %md
# MAGIC ## Merge seed data and evolved data

# COMMAND ----------

from _setup.params import *

# COMMAND ----------

seed_table_name = "yen.syn_data_gen.seed"
evolved_table_name = "yen.syn_data_gen.evolved"
data_table_name = "yen.syn_data_gen.data"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read seed data and evolved data

# COMMAND ----------

seed_df = spark.table(seed_table_name)
evolved_df = spark.table(evolved_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge seed data and evolved data

# COMMAND ----------

#TODO: mismatched columns
merged_df = seed_df.unionAll(evolved_df).na.drop(how='any')
display(merged_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split into train and test sets
# MAGIC Test set should contain the original seed and its evolved variants

# COMMAND ----------

test_questions = [row['question'] for row in merged_df.select('question').distinct().sample(0.2).collect()]
test_questions


print(data_annotated.count(), data_aug.count(), data.count())
