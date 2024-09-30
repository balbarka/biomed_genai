# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Experiment Run Validation
# MAGIC
# MAGIC This task notebook is inlcuded to verify that we have at least one experiment created. This will appear trivial when there is only one agent being considered, but when there are more than one candidate agents, this is helpful to have as a place to compare model scores (llm-as-judge) and select best model.

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_guided_chat $SHOW_AGENT_MODEL=true $SHOW_NOTEBOOK_TASKS=false

# COMMAND ----------

EXPERIMENT_RUN_VALIDATED = False
if len(bc_guided_chat.experiment.models) > 0:
    EXPERIMENT_RUN_VALIDATED = True

EXPERIMENT_RUN_VALIDATED
