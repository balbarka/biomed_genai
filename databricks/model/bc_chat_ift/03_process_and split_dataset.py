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

from pyspark.sql.types import StringType
from pyspark.sql.functions import lit, udf, split, size, col

# COMMAND ----------

from _setup.params import *

# COMMAND ----------

seed_table_name = "yen.syn_data_gen.seed"
evolved_table_name = "yen.syn_data_gen.evolved"

data_table_name = "yen.syn_data_gen.data"
train_table_name = "yen.syn_data_gen.train"
test_table_name = "yen.syn_data_gen.test"

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

merged_df = seed_df \
    .withColumn('question_new', lit(None).cast(StringType())) \
    .withColumn('answer_new', lit(None).cast(StringType())) \
    .withColumn('prompt', lit(None).cast(StringType())) \
    .select(*evolved_df.columns) \
    .union(evolved_df).na.drop(how='any')
display(merged_df)

# COMMAND ----------

merged_df.count(), seed_df.count(), evolved_df.count()

# COMMAND ----------

def make_completion_prompt(question, context):
    return f"""You are a medical librarian. Answer the question given the following context
### Question: {question}
### Context: {context}
### Answer:"""
udf_make_completion_prompt = udf(make_completion_prompt, StringType())

# COMMAND ----------

#TODO
[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": a}]

# COMMAND ----------

data = merged_df.dropDuplicates() \
    .withColumnRenamed('prompt', 'evolve') \
    .withColumnRenamed('answer_new', 'response') \
    .withColumn("prompt", udf_make_completion_prompt(merged_df.question_new,
                                                     merged_df.context)) \
    .withColumn('len', size(split(col('prompt'), '\W')))
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split into train and test sets
# MAGIC Test set should contain the original seed and its evolved variants

# COMMAND ----------

test_questions = [row['question'] for row in data.select('question').distinct().sample(0.3).collect()]
test_questions

# COMMAND ----------

test = data.filter(merged_df.question.isin(test_questions)) \
    .select(['prompt','response']) \
    .dropDuplicates()
train = data.filter(~merged_df.question.isin(test_questions)) \
    .select(['prompt','response']) \
    .dropDuplicates()
test.count(), train.count()

# COMMAND ----------

display(test)

# COMMAND ----------

train.write.option("overwriteSchema", "true").saveAsTable(train_table_name, mode="overwrite")
test.write.option("overwriteSchema", "true").saveAsTable(test_table_name, mode="overwrite")

# COMMAND ----------


