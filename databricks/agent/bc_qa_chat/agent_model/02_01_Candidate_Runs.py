# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC **WIP**: This notebook is only to assess that individial models, **agent.bc_qa_chat.candidate_models**, listed in **config_biomed_genai.yaml** have been updated to the current **agent.bc_qa_chat.release_version**.
# MAGIC
# MAGIC The majority of updates will be in the agent models contained within the sub folder **models**.

# COMMAND ----------

# MAGIC %run ../_setup/setup_bc_qa_chat $SHOW_GOVERNANCE=false $SHOW_AGENT_DEPLOY=false
