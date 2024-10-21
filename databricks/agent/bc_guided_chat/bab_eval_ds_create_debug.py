# Databricks notebook source
# MAGIC %run ./_setup/setup_bc_guided_chat $SHOW_AGENT_MODEL=true $SHOW_NOTEBOOK_TASKS=false

# COMMAND ----------

bc_guided_chat.experiment.eval_ds.name

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE biomed_genai.agents.bc_guided_eval_ds;

# COMMAND ----------

eval_ds = bc_guided_chat.experiment.eval_ds

# COMMAND ----------

print(eval_ds.create_sql)

# COMMAND ----------

eval_ds.__class__

# COMMAND ----------

from dataclasses import dataclass
from biomed_genai.config import UC_SQL_Entity, WS_GenAI_Agent_Experiment, UC_Dataset, UC_Registered_Model

# COMMAND ----------



# COMMAND ----------

self = bc_guided_chat
eval = UC_Dataset(uc_name=f"{self.schema.agents.name}.{self.eval_ds_name}",
                  sql_file="CREATE_TABLE_agents_bc_guided_eval_ds.sql",
                  sql_folder=self.config_ddl_folder,
                  release_version=self.release_version)
self = eval                  

# COMMAND ----------

import re

sql = self.create_sql
kwargs = {k: getattr(self, k) for k in
          set(self.__dir__()).intersection(set(re.findall(r"{(.*?)}", sql)))}

# COMMAND ----------

print(sql.format(**kwargs))

# COMMAND ----------

eval.name

# COMMAND ----------


