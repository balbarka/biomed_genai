# Databricks notebook source
# DBTITLE 1,Install pip_requirements
import subprocess
pip_requirements = ['langchain==0.2.11',
                    'langchain-community==0.2.7',
                    'databricks_vectorsearch==0.39']

for package in pip_requirements:
    subprocess.run(["pip", "install", package], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# COMMAND ----------

# DBTITLE 1,Retrieve Agent Configs
# MAGIC %run ../_setup/setup_bc_guided_chat $SHOW_AGENT_MODEL=false $SHOW_NOTEBOOK_TASKS=false

# COMMAND ----------

# DBTITLE 1,Create PythonModelContext
import yaml
from mlflow.pyfunc import PythonModelContext

model_dir = PROJECT_ROOT_PATH + '/databricks/agent/bc_guided_chat/bc_guided_chat_agent/'
model_path = model_dir + 'model.py'
config_path = model_dir + 'config.yaml'

with open(config_path, 'r') as file:
    app_config = yaml.safe_load(file)

context = PythonModelContext(artifacts={"model": model_path},
                             model_config=app_config)

# COMMAND ----------

# DBTITLE 1,Create input_example
question = "What are some preventative measures that can be take to reduce the risk of Breast Cancer?"
input_example = {"messages": [{"role": "user",
                               "content": question}]}

# COMMAND ----------

# DBTITLE 1,Test Model Locally
exec(open(model_path).read())
chain.invoke(input_example)

# COMMAND ----------

import mlflow
from mlflow.entities.run_info import RunInfo
from mlflow.models.evaluation.base import EvaluationResult

# Don't use the experiment methods from the config class - ask Brad when you get here 
with bc_guided_chat.experiment.create_model_run(overwrite=True,
                                            nb_experiment=False) as run:
        run_info: RunInfo = mlflow.langchain.log_model(lc_model=context.artifacts['model'],
                                                       model_config=context.model_config,
                                                       artifact_path="model",
                                                       pip_requirements=pip_requirements,
                                                       input_example=input_example,
                                                       example_no_conversion=True)
        mlflow.set_tag("release_version", bc_guided_chat.experiment.release_version)

        # eval_rslt: EvaluationResult = bc_guided_chat.experiment.evaluate()

# COMMAND ----------

# DBTITLE 1,Create Experiment Model Run & Evaluation Run
eval_rslt: EvaluationResult = bc_guided_chat.experiment.evaluate()
