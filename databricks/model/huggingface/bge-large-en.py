# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This HF model is an embedding model. While it can be used directly, we'll include it as a hf model download and in a later model fine tune this embedding model. Check out the [model card](https://huggingface.co/BAAI/bge-large-en).

# COMMAND ----------

# MAGIC %run ./_setup/setup_hf_cache

# COMMAND ----------

MODEL_NAME = "BAAI/bge-large-en"
MODEL_REVISION = "abe7d9d814b775ca171121fb03f394dc42974275"

hf_model_run = hf_cache.experiment.get_or_create_hf_model_run(hf_model_name = MODEL_NAME,
                                                              hf_model_revision = MODEL_REVISION)

# COMMAND ----------


