# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Agent Evaluation
# MAGIC
# MAGIC This is the notebook where we'll create an agent evaluation run. This is how we evaluate our agent answering questions that were created in the 02-Evaluation_Dataset notebook. LLM-as-Judge, is a built in Databricks feature that provides an out of box experience for scoring models. However, customer metrics are also possible, please see databricks [documentation](https://docs.databricks.com/en/generative-ai/agent-evaluation/llm-judge-metrics.html).

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_guided_chat $SHOW_AGENT_MODEL=true $SHOW_NOTEBOOK_TASKS=false

# COMMAND ----------

import mlflow
from mlflow.entities.run_info import RunInfo
from mlflow.models.evaluation.base import EvaluationResult

# Don't use the experiment methods from the config class - ask Brad when you get here 
# with bc_guided_chat.experiment.create_model_run(overwrite=False, nb_experiment=False) as run:
eval_rslt: EvaluationResult = bc_guided_chat.experiment.evaluate()
