# Databricks notebook source
# MAGIC %pip install transformers

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# https://mlflow.org/docs/latest/python_api/mlflow.transformers.html
from transformers import AutoTokenizer, pipeline
import mlflow

# COMMAND ----------

# Get model from HF
architecture = "gpt2"
artifact_path = "model"

# Method 1: Using pipeline
gen_pipeline = pipeline(
    task="text-generation",
    tokenizer=AutoTokenizer.from_pretrained(architecture),
    model=architecture,
)

# Test pipeline
# https://huggingface.co/csarron/mobilebert-uncased-squad-v2
queries = ["Generative models are", 
           "I'd like a coconut so that I can"]
gen_pipeline(queries)

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model=gen_pipeline,
        artifact_path=artifact_path,
        # TODO: register model
    )

# Method 2: Using components dict
# with mlflow.start_run() as run:
#     components = {
#         "model": model,
#         "tokenizer": tokenizer,
#     }
#     mlflow.transformers.log_model(
#         transformers_model=components,
#         artifact_path=artifact_path,
#     )

# COMMAND ----------

run.info

# COMMAND ----------

model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
loaded_model = mlflow.transformers.load_model(model_uri)
loaded_model.predict(queries)

# COMMAND ----------

loaded_model.__class__

# COMMAND ----------


