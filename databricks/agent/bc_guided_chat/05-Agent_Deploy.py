# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Agent Deploy
# MAGIC
# MAGIC We are now going to deploy to our model serving endpoint. However, because we are deploying an agent, we are going to use the Databricsk [agent deploy](https://docs.databricks.com/en/generative-ai/deploy-agent.html) from the Databricks sdk. This has the additional benefit that when used, a review app will be created to capture Human Feedback automatically. The link to this review app will be available in the output of the last cell in this notebook.

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_guided_chat $SHOW_AGENT_MODEL=true $SHOW_NOTEBOOK_TASKS=false

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Register Model
# MAGIC

# COMMAND ----------

# Taking first, but this needs to be updated to behave like looks for best performing.
bc_guided_chat.experiment.models[0].model_run_id

# COMMAND ----------

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion

def register_model(bc_qa_chat, model) -> ModelVersion:
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.register_model(model_uri=f"runs:/{model.model_run_id}/model",
                                 name=f'{bc_qa_chat.schema.agents.uc_name}.bc_guided_chat')

best_model = bc_guided_chat.experiment.models[0]
mdl_version = register_model(bc_guided_chat, best_model)

# COMMAND ----------

from databricks import agents
from databricks.agents.sdk_utils.entities import Deployment
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
import time

reviewer_instruction = """## Instructions for Testing the Guided BioMed Articles Assistant chatbot\n\nYour inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement."""

def deploy_agent(mdl_version: ModelVersion,
                 instructions: str) -> Deployment:
    deployment_info = agents.deploy(model_name = mdl_version.name,
                                    model_version = int(mdl_version.version),
                                    scale_to_zero=True)
    agents.set_review_instructions(mdl_version.name, instructions)
    return deployment_info

def await_agent_endpoint(deployment_info: Deployment):
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(deployment_info.endpoint_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {deployment_info.endpoint_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
          print('endpoint ready.')
          return
        else:
          break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

def get_or_create_agent_review(mdl_version: ModelVersion,
                               instructions: str) -> Deployment:
    try:
        return agents.client.rest_client.get_chain_deployments(model_name=mdl_version.name,
                                                               model_version=mdl_version.version)
    except Exception as e:
        deployment_info = deploy_agent(**locals())
        return await_agent_endpoint(deployment_info)

# Brad to confirm that it's safe to remove the get_or_create
# agent_deployment_info = get_or_create_agent_review(mdl_version, reviewer_instruction)
agent_deployment_info = deploy_agent(mdl_version, reviewer_instruction)

