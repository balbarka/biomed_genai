# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # config
# MAGIC
# MAGIC Brad to create the config class for all entities
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_chat_ift

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Entities That will need to be created:
# MAGIC  - Catalog
# MAGIC  - Schema
# MAGIC    - ift_ds (synthetic ds, include both training and holdout)
# MAGIC    - eval_ds (reused from original model)
# MAGIC    - ft_model
# MAGIC
# MAGIC  - experiment
# MAGIC  - 
# MAGIC
# MAGIC - base_model (databricks_meta_llama_3_models.models.meta_llama_3_8b_instruct)
# MAGIC
# MAGIC Discussion - what is the best practice on evaluation metrics - do we want to keep the same eval process for IFT model as we have for agent
