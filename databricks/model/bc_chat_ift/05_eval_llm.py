# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is fifth in a series that generates synthetic data for subsequent chat completion Fine Tuning (FT). This notebook **compares the finetune model with the base model as answering LLMs alone (without a retriever).**
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. Perform prediction on the test set from NB 3 using the base model and finetuned model from NB 4
# MAGIC 2. Compare the predictions using `mlflow.evaluate`

# COMMAND ----------

# MAGIC %pip install databricks-genai databricks-sdk mlflow textstat
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze

# COMMAND ----------

import os, json
import mlflow
from mlflow import deployments
from pyspark.sql.functions import expr
from databricks.sdk import WorkspaceClient
from _setup.params import *

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set parameters and names

# COMMAND ----------

catalog = "yen"
db = "syn_data_gen"

test_table_name = f"{catalog}.{db}.test"

base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ft_model_name = f"{catalog}.{db}.pubmed_rag_model"
llm_judge = 'databricks-meta-llama-3-1-405b-instruct'

base_endpoint_name = "databricks-meta-llama-3-1-70b-instruct"
model_endpoint_name = "pubmed_rag_model"
inference_table_name = model_endpoint_name

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inferencing: get predictions from base model and FT model
# MAGIC Evaluate with our `test` dataset

# COMMAND ----------

test = spark.table(test_table_name)
display(test)

# COMMAND ----------

# Responses from base model without finetuning
pred_base = test \
    .withColumn("question", test.messages.getItem(1)['content']) \
    .withColumn("answer", test.messages.getItem(2)['content']) \
    .withColumn("prediction", expr(f"""ai_query('{base_endpoint_name}',
                                   CONCAT('{{"messages": [{{"role": "user", "content": "', question, '}}]}}'))"""))
display(pred_base)

# COMMAND ----------

# Responses from finetuned model
pred_ft = test \
    .withColumn("question", test.messages.getItem(1)['content']) \
    .withColumn("answer", test.messages.getItem(2)['content']) \
    .withColumn("prediction", expr(f"""ai_query('{model_endpoint_name}', 
                                   CONCAT('{{"messages": [{{"role": "user", "content": "', question, '}}]}}'))"""))
display(pred_ft)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evaluate model answering quality (no retrieval) - `mlflow.evaluate`
# MAGIC Use LLM judges with ground truth answers to compare responses from a finetuned model vs those from a base model without finetuning.
# MAGIC
# MAGIC #### Generate prediction dataframes using the base model and the finetuned model

# COMMAND ----------

# Eval base model
with mlflow.start_run(run_name=f"eval_{base_endpoint_name}") as run:
    results = mlflow.evaluate(
        data=pred_base,
        targets="answer",
        predictions="prediction",
        model_type="question-answering",
        extra_metrics=[
            mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}"),
            mlflow.metrics.genai.answer_correctness(model=f"endpoints:/{llm_judge}")
        ],
        evaluators="default",
        evaluator_config={'col_mapping': {'inputs': 'question'}}
    )

# COMMAND ----------

# Eval FT model
with mlflow.start_run(run_name=f"eval_{model_endpoint_name}") as run:
    results = mlflow.evaluate(
        data=pred_ft,
        targets="answer",
        predictions="prediction",
        model_type="question-answering",
        extra_metrics=[
            mlflow.metrics.genai.answer_similarity(model=f"endpoints:/{llm_judge}"),
            mlflow.metrics.genai.answer_correctness(model=f"endpoints:/{llm_judge}")
        ],
        evaluators="default",
        evaluator_config={'col_mapping': {'inputs': 'question'}}
    )


# COMMAND ----------


