# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook is only used for demonstrating the funcitonality of a single component which will be used in a bricktrhrough idea submission.

# COMMAND ----------

# DBTITLE 1,Run - Agent Assemblage Setup
# MAGIC %run ./_setup/setup_bc_qa_chat $SHOW_AGENT_DEPLOY=true

# COMMAND ----------

sql_file = "CREATE TABLE IF NOT EXISTS {uc_name} (col1 INT, col2 INT)"
file_path = "/tmp/CREATE_TABLE_demo_table.sql"

with open(file_path, "w") as file:
    file.write(sql_file)

# COMMAND ----------

# DBTITLE 1,UC Table Component - Instantiation
from gaia.component import UC_Table

catalog = "biomed_genai"
schema = "main"
sql_folder = "/tmp"

table_component = UC_Table(uc_name=f"{catalog}.{schema}.table",
                           sql_file="CREATE_TABLE_demo_table.sql",
                           sql_folder=sql_folder)

# COMMAND ----------

# DBTITLE 1,UC Table Component - pyspark DataFrame class
table_component.df.__class__

# COMMAND ----------

# DBTITLE 1,UC Table Component - delta.io DeltaTable class
table_component.dt.__class__

# COMMAND ----------

# DBTITLE 1,UC Table Component - Navigability
table_component.uc_relative_url

# COMMAND ----------

from biomed_genai.agent.bc_qa_chat.agent_bc_qa_chat import Agent_model_bc_qa_chat
import yaml

with open('project_config.yaml', 'r') as file:
    config_bc_qa_chat = yaml.safe_load(file).get("agent")["bc_qa_chat"]

bc_qa_chat = Agent_model_bc_qa_chat(**config_bc_qa_chat)

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la /Workspace/Repos/brad.barker@databricks.com/biomed_genai/databricks/_config/config_biomed_genai.yaml
# MAGIC

# COMMAND ----------

# DBTITLE 1,Agent Assemblage - Instantiation
from biomed_genai.agent.bc_qa_chat.agent_bc_qa_chat import Agent_model_bc_qa_chat
import yaml

with open('config_biomed_genai.yaml', 'r') as file:
    config_bc_qa_chat = yaml.safe_load(file).get("agent")["bc_qa_chat"]

bc_qa_chat = Agent_model_bc_qa_chat(**config_bc_qa_chat)

# COMMAND ----------

# DBTITLE 1,Agent Assemblage - Experiment Component
bc_qa_chat.experiment.__class__

# COMMAND ----------

# DBTITLE 1,Agent Assemblage - Registered Model Component
bc_qa_chat.registered_model.__class__

# COMMAND ----------

# DBTITLE 1,Agent Assemblage - Navigability
bc_qa_chat.experiment.ws_relative_url
