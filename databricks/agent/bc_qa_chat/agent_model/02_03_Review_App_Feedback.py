# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Review App Feedback
# MAGIC
# MAGIC This is the task where we look at evaluating the feedback given to an app.
# MAGIC
# MAGIC It is both possible and likely that in this process there will be instances where there will be feedback that will change the critiera of best model selection. This is ok, but you don't want to jump out of cycle and update the eval dataset. Instead, we need to make sure that we manage as a critieria and recorded in a candidate model expereiment model run tag like `diqualifying_feedback`.
# MAGIC
# MAGIC Thus, some other good results from feedback are:
# MAGIC     - Potential changes to arguments for a candidate model.
# MAGIC     - How to score candidate models. 
# MAGIC     - Criteria for Awarding Champion.
# MAGIC
# MAGIC **NOTE**: It is not appropriate to introduce new dependencies during the inner-loop. It is also not appropriate to create new eval_ds entries. Instead, annotate these opportunities. If within the inner-loop it's appropriate to do an iteration and implement changed within the same release_version.
# MAGIC
# MAGIC **TODO**: Write some exploritory analyses from the feedback generated from the review app.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM biomed_genai.agents.bc_qa_chat_2_payload

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM biomed_genai.agents.bc_qa_chat_2_payload_assessment_logs;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM biomed_genai.agents.bc_qa_chat_2_payload_request_logs;
