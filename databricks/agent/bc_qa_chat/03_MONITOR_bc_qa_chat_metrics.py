# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Monitoring
# MAGIC
# MAGIC Since the monitoring types (TimeSeries, InferenceLog, or Snapshot) don't work with genai agent inference tables out of box, we'll want to dev our own using this notebook as source. 
# MAGIC
# MAGIC **TODO**: Write a sample monitoring page, include summarization of feedback and total use metrics at the agent level and organized by agent model release_versions. We should avoid getting into model class specific attributes since that will be hard to make consistant across many agent release versions if the model class changes.
# MAGIC
# MAGIC

# COMMAND ----------

payload_table = "biomed_genai.agents.bc_qa_chat_2_payload"
assessment_table = "biomed_genai.agents.bc_qa_chat_2_payload_assessment_logs"
requests_table = "biomed_genai.agents.bc_qa_chat_2_payload_request_logs"


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC https://docs.databricks.com/en/lakehouse-monitoring/create-monitor-api.html

# COMMAND ----------

# MAGIC %pip install "databricks-sdk>=0.28.0"

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorSnapshot

w = WorkspaceClient()
w.quality_monitors.create(
  table_name="biomed_genai.agents.bc_qa_chat_2_payload",
  assets_dir=f"/Workspace/genai_dashboard",
  output_schema_name=f"biomed_genai.agents",
  snapshot=MonitorSnapshot()
)

# COMMAND ----------


