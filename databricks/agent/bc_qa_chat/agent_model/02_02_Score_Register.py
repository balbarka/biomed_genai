# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Score, Register, and Deploy Agent Evaluation
# MAGIC
# MAGIC We already have run evaluate and gotten metrics for each of our models. It's now time to look at those results, protentially add further metrics and determine the best. In this notebook, we'll write a function to choose the best model and deploy it for feedback.
# MAGIC
# MAGIC **NOTE**: It is also possible that only a single model is created for a release. This step should still be taken as this will not always be the case and results written back to the experiment model run.
# MAGIC
# MAGIC It's always possible to deploy multiple models for feedback, but this will complicate the task of getting feedback equally across models. Instead, our convention will be to select a single best model from the candidate models with opportunity to improve any of the following:
# MAGIC  - Model Architecture
# MAGIC  - Model Parameters
# MAGIC  - Model Training
# MAGIC  - Evaluation Dataset
# MAGIC  - Metrics
# MAGIC
# MAGIC **NOTE**: There is an investment bias towards when selecting a model that you have spent the most time on. Thus it is important that the selection criteria are understood ahead of actual scoring so that it doesn't effect the scoring definition.
# MAGIC
# MAGIC **NOTE**: All criteria must be in a model run for best model selection. Meaning that you should make use of the tags and metrics feature in MLFlow so that all attributes of a decision are captured within the candidate model.
# MAGIC
# MAGIC **NOTE**: The results of a model comparison is a metric and therefor should be written back to the experiment model run accordingly. 

# COMMAND ----------

# MAGIC %run ../_setup/setup_bc_qa_chat $SHOW_GOVERNANCE=false $SHOW_AGENT_DEPLOY=false

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Define scoring metric and record to experiment
def score_model(model):
    run_metrics = client.get_run(model.eval.eval_run_id).data.metrics
    score = (0.25 * run_metrics['response/llm_judged/correctness/rating/percentage'] +
             0.25 * run_metrics['response/llm_judged/groundedness/rating/percentage'] +
             0.25 * run_metrics['response/llm_judged/relevance_to_query/rating/percentage'] +
             0.25 * run_metrics['response/llm_judged/safety/rating/percentage'])
    with mlflow.start_run(run_id=model.model_run_id) as run:
        mlflow.log_metric('score', score)
    return {'model': model,
            'metrics': run_metrics,
            'score': score}

model_scores = sorted([score_model(m) for m in bc_qa_chat.experiment.models],
                      key=lambda x: x['score'], reverse=True)

best_model = model_scores[0]

# COMMAND ----------

model_scores

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Register our Model
# MAGIC
# MAGIC Since this is our first iteration, we'll need to go ahead and register our model.
# MAGIC
# MAGIC **TODO**: Need to add additional logic to keep the model version in sync with the release version.

# COMMAND ----------

from mlflow.entities.model_registry.model_version import ModelVersion

def register_model(bc_qa_chat, model) -> ModelVersion:
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.register_model(model_uri=f"runs:/{model.model_run_id}/model",
                                 name=f'{bc_qa_chat.schema.agents.uc_name}.bc_qa_chat')

mdl_version = register_model(bc_qa_chat, best_model['model'])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Deploy a Agent Evaluation for Registered Model `bc_qa_chat`
# MAGIC
# MAGIC [Deploying an Agent for Generative AI Applications](https://docs.databricks.com/en/generative-ai/deploy-agent.html#deploy-an-agent-for-generative-ai-application) is still private preview. This is a significant deviation from our typical MLOps superviser learning deployment because:
# MAGIC
# MAGIC  * We are using LLM as a judge that requires a deployment of a model to evaluate our model
# MAGIC  * Our metrics are not simple scalers, instead we have a lot of complex type data to inspect. Infact, there are 3 new [agent-enhanced-inference-tables](https://docs.databricks.com/en/generative-ai/deploy-agent.html#agent-enhanced-inference-tables) to capture all of the feedback.
# MAGIC
# MAGIC To deploy an agent, we'll create a method that will inspect if an agent has already been deployed and deploy one if not. We'll also need to include `reviewer_instruction` with this process.
# MAGIC
# MAGIC There are a couple interesting artifacts that come out of agent deployment. Specifically:
# MAGIC
# MAGIC  * **View Status URL** which is a hyperlink to a model serving endopint that has both model and a feedback model deployed.
# MAGIC  * **Review App**: which is a front-end accessible ouside of the ws where we can gather human feedback
# MAGIC
# MAGIC Take some time to explore both via the hyperlinks provided in the deploy output.

# COMMAND ----------

from databricks import agents
from databricks.agents.sdk_utils.entities import Deployment
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
import time

reviewer_instruction = """## Instructions for Testing the BioMed Articles Assistant chatbot\n\nYour inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement."""

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

# agent_deployment_info = get_or_create_agent_review(mdl_version, reviewer_instruction)

agent_deployment_info = deploy_agent(mdl_version, reviewer_instruction)
