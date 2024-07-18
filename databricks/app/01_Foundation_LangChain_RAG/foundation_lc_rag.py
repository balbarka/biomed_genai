# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Deployment Cycle for a Foundation Model LangChain RAG Application
# MAGIC
# MAGIC In this notebook we are going to go through a deployment cycle from source code to model serving for a Foundation Model LangChain RAG Application. We have a couple pre-requisites:
# MAGIC  - **Access to Foundation Models** - Databricks comes with ready access to [Foundation Models](https://docs.databricks.com/en/machine-learning/foundation-models/index.html#databricks-foundation-model-apis). This are available on all Worskpace Deployments as long as it hasn't been disab;ed by an administrator.
# MAGIC  - **LangChain Library** - [LangChain](https://python.langchain.com/v0.2/docs/concepts/) is a very popular LLM chain framework libarary that employs OOM and functional programming concepts to simplify GenAI application development.
# MAGIC  - **Vector Index** - We'll make use of the VectorIndex, `articles_content_vs_index`, that we developed during `genai_workflow` which will provide all of the context from PMC that isn't in our foundation model.
# MAGIC  - **Evaluation Dataset** - Also developed during `genai_workflow` we'll use these questions and judge results to create metrics on the performance of our genai models.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Foundation Model LangChain RAG Application Stages:
# MAGIC
# MAGIC Here are the stages we'll go through in this notebook from source code to model serving endpoint:
# MAGIC
# MAGIC  * **Gather model code & configurations**
# MAGIC  * **Initialize & Test app locally**
# MAGIC  * **Create Experiment Run**
# MAGIC  * **Register Experiment as a UC Model**
# MAGIC  * **Deploy a Agent Evaluation Model**
# MAGIC  * **Evaluate Agent + Human Feedback** - TODO, for now show [agent enhanced inference tables](https://docs.databricks.com/en/generative-ai/deploy-agent.html#agent-enhanced-inference-tables)
# MAGIC  * **Test Deployed Model Locally**
# MAGIC  * **Deploy Model & Test deployment** 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Gather model code & configurations
# MAGIC
# MAGIC To make sure that we are adopting the configurations from the work done in `biomed_workflow` we are going to import the provided configurations for this genai app, but overwrite using derived names from `biomed_workflow` in case the workflow was run with non-default configurations.
# MAGIC
# MAGIC  * `app_config`: **WIP** - We are going to use default app configs, app configs from file, and derived configs from `biomed_workflow` until the setup_app notebook is complete. 
# MAGIC  * `biomed_app.foundation_lc_rag`: Our langchain application source code, <a href="$../../../python/biomed_app/foundation_lc_rag.py" target="_blank">foundation_lc_rag.py</a> ,  for this model is provided to us as a single file module. That means that genai developers would likely iterate and commit this code in an IDE before being pulled into Databricks. Developers will typically make use of [databricks connect](https://docs.databricks.com/en/dev-tools/databricks-connect/index.html#what-is-databricks-connect) for that development cycle and won't be covered here.
# MAGIC  * `input_example`: commonly used in local testing and a common argument provided to `log_model`, we are going to create an `input_example` that will ultimately be uploaded into mlflow and eventually be used as the example in the model serving UI.
# MAGIC
# MAGIC  **TODO**: Update with setup_app, for now will use setup_workflow just to update python path which is needed for module import

# COMMAND ----------

import mlflow
import yaml

# This is our default configs
app_config = {"vector_search_endpoint_name": "biomed",
              "vector_search_index_name": "biomed_genai.processed.articles_content_vs_index",
              "llm_model_serving_endpoint_name": "databricks-dbrx-instruct",
              "llm_prompt_template": ""}

# Import the configs we are setting for this app
# Could use mlflow.models.ModelConfig, but it's more succint to read this config into dict with yaml
# appConfig = mlflow.models.ModelConfig(development_config="config.yaml")

with open('config.yaml', 'r') as file:
    app_config.update(yaml.safe_load(file))

OVERWRITE_DEFAULT_CONFIGS = True

if OVERWRITE_DEFAULT_CONFIGS:
    # This update is unnecessary if ModelConfig has been updated or using default biomed_workflow configs
    with open("../../workflow/config/config_workflow.yaml", 'r') as file:
        workflow_config = yaml.safe_load(file)
    app_config.update({'vector_search_index_name':
                       f"{workflow_config['APP_CATALOG']}.{workflow_config['APP_PROCESSED_SCHEMA']}.articles_content_vs_index"})

app_config

# COMMAND ----------

# Unlike other mlflow supervised learning processes where mlflow.xxx.log_model accepts a model class instance that is pickled,
# in the langchain built-in model type, the expectation is that we provide a source python script that will instantiate our model
# NOTE: done to mitigate complexity that arises from un-pickleable items that can arise from local dependencies when instantiating lc classes
# Below is a way to inspect the code we will be uploading to mlflow

import os, sys
import importlib.util
import inspect 

def get_model():
    """This script assumes you have a file in this notebook directory called model.py"""
    module_dir = '.'
    module_name = 'model'
    module_file = 'model.py'

    # Add the directory to sys.path
    sys.path.append(os.path.abspath(module_dir))

    # Import the module
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_dir, module_file))
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model

model = get_model()

print(inspect.getsource(model))

# COMMAND ----------

question = "What is Breast Cancer?"
input_example = {"messages": [{"role": "user",
                               "content": question}]}

# COMMAND ----------

# Create Model Context
import os

from mlflow.pyfunc import PythonModelContext

context = PythonModelContext(artifacts={"model": os.path.abspath("./model.py")},
                             model_config=app_config)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Initialize & Test app locally
# MAGIC
# MAGIC Before we get into creating an experiment in mlflow, we'll want to make sure our base application code works first. 
# MAGIC
# MAGIC We are going to run the test in two ways:
# MAGIC  - from import
# MAGIC  - from exec
# MAGIC
# MAGIC Working with product team to clarify what an appropriate local test should be for this approach.

# COMMAND ----------

# Test our chain instantiated from our configs, with our input_example

chain = model.get_lc_chain(endpoint_name=app_config['vector_search_endpoint_name'],
                           index_name=app_config['vector_search_index_name'],
                           prompt_template=app_config['llm_prompt_template'],
                           model_name=app_config['llm_model_serving_endpoint_name'])

chain.invoke(input_example)

# COMMAND ----------

exec(open('model.py').read())
chain.invoke(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create Experiment Run
# MAGIC
# MAGIC Unlike most other supervised learning built-in model frameworks, [mlflow.langchain.log_model](https://mlflow.org/docs/latest/python_api/mlflow.langchain.html#mlflow.langchain.log_model) accepts the arguement **lc_model** as a *path* to our model file. So that mlflow knows what class to use as the instantiated model, you must include **mlflow.models.set_model()**.
# MAGIC
# MAGIC **NOTE**: This approach is still experimental so another approach just passing our chain if provided. Also not clear yet if the existing documented pattern in mlflow, [Logging the Chain in MLflow](https://mlflow.org/docs/latest/llms/langchain/notebooks/langchain-quickstart.html#Logging-the-Chain-in-MLflow), is still viable. I was not able to use it due to my decortors not being serializable.

# COMMAND ----------

import mlflow
import os
from mlflow.entities.run_info import RunInfo

mlflow.set_experiment(experiment_name='/experiments/biomed_app/foundation_lc_rag')

def create_mlflow_run(lc_model: str,
                      context: PythonModelContext,
                      input_example: dict) -> RunInfo:
    with mlflow.start_run(run_name="foundation_lc_rag") as run:
        return mlflow.langchain.log_model(
            lc_model=context.artifacts['model'],
            model_config=context.model_config,
            artifact_path="foundation_lc_rag",
            pip_requirements=["langchain_community", "databricks_vectorsearch"],
            input_example=input_example,
            example_no_conversion=True)
        run_info = run.info

run_info = create_mlflow_run(lc_model="./model.py",
                             context=context,
                             input_example=input_example)

# In case of running in a new session, this'll get the last experiement_info
# runs = mlflow.get_experiment_by_name("/experiments/biomed_app/foundation_lc_rag")
# run_info = mlflow.get_run(list(runs.run_id)[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Register Experiment as a UC Model
# MAGIC
# MAGIC This is comparable to registering any other model. Notice here, that we create our deploy function so that it uses the output of our mlflow_run. 
# MAGIC
# MAGIC We are imposing a convention here that the vector search index catalog is used and we'll use a fixed schema `model`.

# COMMAND ----------

# DBTITLE 1,UC Model Deployment
from mlflow.entities.model_registry.model_version import ModelVersion

def register_model(catalog: str,
                   schema: str,
                   run_info: RunInfo) -> ModelVersion:
    mlflow.set_registry_uri("databricks-uc")
    model_name = "foundation_lc_rag"
    return mlflow.register_model(model_uri=f'runs:/{run_info.run_id}/{model_name}',
                                 name=f'{catalog}.{schema}.{model_name}')

mdl_version = register_model(catalog=app_config["vector_search_index_name"].split('.')[0],
                             schema="models",
                             run_info=run_info)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Deploy a Agent Evaluation Model
# MAGIC
# MAGIC **NEW!**
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
                                    model_version = mdl_version.version,
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

# COMMAND ----------

# TODO: Run a judge evaluation
# Eval DS not ready yet

# ref code:
#with mlflow.start_run(run_id=logged_chain_info.run_id):
#    # Evaluate the logged model
#    eval_results = mlflow.evaluate(
#        data=eval_dataset,
#        model=logged_chain_info.model_uri,
#        model_type="databricks-agent",
#    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluate Agent + Human Feedback
# MAGIC
# MAGIC **TODO**: Go through all the api calls for evaluating feedback as well as cover all the agent-enhanced inference tables that come with using agent evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Test Deployed Model Locally
# MAGIC
# MAGIC **TODO**: Pull a copy of the model and test it locally. This test is helpful in case developers want to use the model for a specific transform, but not employ a serving end-point for the model.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Deploy Model & Test deployment
# MAGIC
# MAGIC Not clear if this is still needed. We have deployed the model using agents, but it confuses the fact that we've deployed something to get metrics (llm as judge), but typical deployment validations would require such metrics. Raising convention discussion with prod team.
