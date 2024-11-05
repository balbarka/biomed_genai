# Databricks notebook source
# MAGIC %pip install databricks-genai transformers databricks-sdk mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import re, os, json
from pyspark.sql.functions import udf, col, flatten
from pyspark.sql.types import StringType, ArrayType
from typing import Optional
import pandas as pd
import datasets
import mlflow

from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM

# COMMAND ----------

# MAGIC %md
# MAGIC ## Method 1: Databricks Foundation Model API with prehosted models
# MAGIC Need TAC, UC

# COMMAND ----------

# REQUIRES UNITY CATALOG!!!
from databricks.model_training import foundation_model as fm

base_model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"

run = fm.create(
  model=base_model_name,
  experiment_path="/Users/yenlow@atroposhealth.com/llama3_1_405b",
  #train_data_path="/Volumes/test_uc_yen/tqlgen/vol/train.jsonl",
  train_data_path="test_uc_yen.tqlgen.train",
  #eval_data_path="/Volumes/test_uc_yen/tqlgen/vol/test.jsonl",
  eval_data_path="test_uc_yen.tqlgen.test",
  # eval_prompts="Write code in a new clinical programming language called Temporal Query Language (TQL) that returns patients with CKD and no history of heart attack.\n### TQL:\n",
  data_prep_cluster_id = "0812-220537-8jkmnll4",
  register_to="test_uc_yen.tqlgen.llama3_1_405b",
  training_duration='10ep' # Duration of the finetuning run, 10 epochs only to make it fast for the demo. Check the training run metrics to know when to stop it (when it reaches a plateau)
)
print(run)
#config.model must be one of the following values: codellama/CodeLlama-13b-hf, codellama/CodeLlama-13b-Instruct-hf, codellama/CodeLlama-13b-Python-hf, codellama/CodeLlama-34b-hf, codellama/CodeLlama-34b-Instruct-hf, codellama/CodeLlama-34b-Python-hf, codellama/CodeLlama-7b-hf, codellama/CodeLlama-7b-Instruct-hf, codellama/CodeLlama-7b-Python-hf, databricks/dbrx-base, databricks/dbrx-instruct, meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-13b-chat-hf, meta-llama/Llama-2-70b-hf, meta-llama/Llama-2-70b-chat-hf, meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-7b-chat-hf, meta-llama/Meta-Llama-3-70B, meta-llama/Meta-Llama-3-70B-Instruct, meta-llama/Meta-Llama-3-8B, meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-v0.1, mistralai/Mistral-7B-Instruct-v0.2, mistralai/Mixtral-8x7B-v0.1, meta-llama/Meta-Llama-3.1-405B, meta-llama/Meta-Llama-3.1-405B-Instruct

# COMMAND ----------

run.get_events()
# 21 mins, $3? for 7 epoch on Mistral-7B-Instruct-v0.2