# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create `dbrx_pyfunc` agent application model
# MAGIC
# MAGIC We are going to create locally, test, and create an experiemnt run for a simple pyfunc model. We won't be using any mlflow built-in classes, but rather the generic pyfunc model class
# MAGIC
# MAGIC Similar to all notebooks in the agent application `bc_qa_chat`, we will first retrieve our `bc_qa_chat` configs, `config_bc_qa_chat` and then instantiate our config class, `bc_qa_chat`.

# COMMAND ----------

# MAGIC %run ../setup_bc_qa_chat

# COMMAND ----------

from biomed_genai.agent.bc_qa_chat.agent_bc_qa_chat import Agent_model_bc_qa_chat

# The configurations for all of our bc_qa_chat agent application
bc_qa_chat = Agent_model_bc_qa_chat(**config_bc_qa_chat)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create `dbrx_pyfunc` parameters
# MAGIC
# MAGIC We are going to write all that parameters that we want tracked in mlflow model run.
# MAGIC
# MAGIC **NOTE**: Assuming that the model class code is unchanged, we expect that any change to these parameters would justify the creation of a new model_run.

# COMMAND ----------

#import mlflow
from mlflow.pyfunc import PythonModelContext
#import yaml
#import os
from mlflow.types.llm import ChatResponse
#from mlflow.types.llm import ChatRequest, ChatMessage, ChatParams, ChatResponse
#import json
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
#from mlflow.models.signature import ModelSignature
#from mlflow.types.schema import Schema, ColSpec, DataType, Array, Object, Property

# This is our agent application configs which will need to be populated as model_configs in mlflow
model_config = {"llm_model_serving_endpoint_name": "databricks-dbrx-instruct",
                "llm_system_instruction": (
                    "You are a research aid for biomedical research specific to breast cancer. "
                    "Provide a detailed and research-oriented response to user questions. "
                    "Ensure that the tone is formal and the information is accurate. "
                    "If the context is unclear, assume the topic is related to breast cancer. "
                    "Try to answer user questions in three sentences or less. ")}

question="What are the most common indicators of breast cancer?"
messages_example=[ChatMessage(role=ChatMessageRole.USER, content=question),]
input_example = {"messages": [{"role": "user",
                               "content": question}]}

#agent_signature = ModelSignature(inputs=Schema([ColSpec(Array(Object(
#                                 properties=[Property(name="content", dtype=DataType.string),
#                                             Property(name="role", dtype=DataType.string),])), "messages")]),
#                                 outputs=Schema([ColSpec(DataType.string)]))

context = PythonModelContext(artifacts={},
                             model_config=model_config)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Define `dbrx_pyfunc` model class
# MAGIC
# MAGIC We'll now define our model useing the enhereted class `mlflow.pyfunc.ChatModel`.
# MAGIC
# MAGIC **NOTE**: While it is advised to write more complicated models as modular python code, it is still common and accepted to write small, succinnt models inline in a notebook. This does complicate the class code portability and re-use.

# COMMAND ----------

import mlflow
from overrides import overrides
from mlflow.types.llm import ChatParams, ChatResponse
from typing import List
from mlflow.pyfunc import PythonModelContext

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from mlflow.types.llm import ChatResponse, ChatChoice, ChatMessage, ChatChoiceLogProbs, TokenUsageStats

import requests




# COMMAND ----------

import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.deployments.databricks import DatabricksDeploymentClient
from mlflow.deployments import get_deploy_client
from typing import List

class Dbrx_pyfunc(ChatModel):

    _deploy_client:DatabricksDeploymentClient = None

    def __init__(self, llm_model_serving_endpoint_name:str,
                       llm_system_instruction:str):
        self.llm_model_serving_endpoint_name = llm_model_serving_endpoint_name
        self.llm_system_instruction = llm_system_instruction

    @property
    def deploy_client(self):
        if not self._deploy_client:
            self._deploy_client = get_deploy_client("databricks")
        return self._deploy_client

    def chat_predict(self, messages):
        return self.deploy_client.predict(endpoint=self.llm_model_serving_endpoint_name,
                                          inputs={"messages": [{"role": "system" , "content": self.llm_system_instruction},
                                                               {"role": "user", "content": messages[0].content}],
                                                  "temperature": 0.1,
                                                  "max_tokens": 250})
    
    @mlflow.trace()
    def predict(self, context, messages: List[ChatMessage], params) -> ChatResponse:
        return ChatResponse(**self.chat_predict(messages))

# COMMAND ----------

dbrx_pyfunc = Dbrx_pyfunc(**model_config)
dbrx_pyfunc.load_context(context)
dbrx_pyfunc.predict(context=context,
                    messages=messages_example,
                    params=None)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create a model_run
# MAGIC
# MAGIC **NOTE**: As convention, we always save the model to `artifact_path='model'`. This ensures ease of discoverability when retrieving artifacts.

# COMMAND ----------

with bc_qa_chat.experiment.create_model_run(overwrite=True) as model_run:
    run_info = mlflow.pyfunc.log_model(artifact_path='model',
                                       python_model=Dbrx_pyfunc(**model_config),
                                       pip_requirements=["openai==1.37.1", "mlflow-skinny==2.15.1", "openai==1.37.1", "requests==2.32.3"],
                                       input_example=input_example,
                                       example_no_conversion=True)
    mlflow.log_params(model_config)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Run Artifact Validation
# MAGIC
# MAGIC    There is a validation that runs when you execute an mlflow experiment run and log_model. However, a common pattern is to also validate the accessibility in another environement. Mlflow provides a convenience meethod `validate_serving_input` which can be helpful when troubleshooting message payloads or if you want to inspect a trace output, but are not ready to deploy the model to a serving endpoint.

# COMMAND ----------

from mlflow.models import validate_serving_input

model_uri = 'runs:/1a90c36a62414e3c8e888c08ff8d2881/model'

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = """{
  "messages": [
    {
      "role": "user",
      "content": "What are the most common indicators of breast cancer?"
    }
  ],
  "temperature": 1.0,
  "n": 1,
  "stream": false
}"""

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Run Evaluate
# MAGIC
# MAGIC    With traditional supervised learning we had access to well studied, deterministic metrics like; accuracy, precision, recall, F1, AUC, MSE, RMSE, R2, etc. A new study of metrics is happening to evaluate the performance of LLMs. When we run evaluate on an llm model, we are running these metrics which are detailed in databricks [llm-as-judge](https://docs.databricks.com/en/generative-ai/agent-evaluation/llm-judge-metrics.html). While some of these LLM metrics are determinist, those designated as judge are not.
# MAGIC
# MAGIC    Similar to having a test dataset for supervised learning models, we'll need to have a test set for LLMs. This is the eval_ds that we create in notebook TODO which is now being used in the evaluate method below.

# COMMAND ----------

bc_qa_chat.experiment.evaluate()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Register Chat Model
# MAGIC
# MAGIC When we register a chat model, we are doing more than just making a UC address for an experiment run model. It is also an indicator that the following business criteria have been met:
# MAGIC  - The model has been validate that the model is able to run without throwing a runtime exception
# MAGIC  - The model has passed business quality requirements
# MAGIC  - The model is viable as a challenger agent application model for the agent application or the model has value as an alternate model for the application - think like having a reserve model that may not perform as well but offers lower cost or better response time performance.
# MAGIC
# MAGIC  The method below uses the `run_info` from above to register the model. Alternately you can retrieve the run_info from the `bc_qa_chat` config class.

# COMMAND ----------

from dataclasses import dataclass
from mlflow.tracking import MlflowClient
from functools import cached_property

@dataclass
class UC_Registered_Model():
    # Registered model class
    uc_name: str
    _exists: bool = None

    @cached_property
    def name(self):
        return self.uc_name


    @property
    def exists(self) -> bool:
        #TODO: use try and get_model

# COMMAND ----------

# KEEP AS REFERENCE TO BUILD FUNCTION FOR REGISTERING

from mlflow.entities.model_registry.model_version import ModelVersion

mlflow.set_registry_uri("databricks-uc")
mdl_version: ModelVersion = mlflow.register_model(model_uri=f'runs:/{run_info.run_id}/model',
                                                      name=f'biomed_genai.agents.bc_qa_chat')


# COMMAND ----------

# KEEP AS REFERENCE FOR dataclass development
from mlflow.tracking import MlflowClient
client = MlflowClient()

registered_model = client.get_registered_model("biomed_genai.agents.bc_qa_chat")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Deploy w/ Agent Evaluation Review App
# MAGIC
# MAGIC From the agent evaluation above, you'll notice that while we curate the evaluation dataset to get the right questions for evaluation, we are still releying on the judgement of an LLM which will not have the context or situational awareness that a user may have. To be able to get the model aligned more closely to human expectations, we will want human evaluation. This human evaluation is captured via a review app as below.
# MAGIC
# MAGIC In the context of Large Language Models (LLMs), such as GPT or other AI models, a "human feedback loop" refers to the process where human evaluators provide feedback on the model's outputs to help improve its performance. This feedback can be used to fine-tune the model, guide its learning process, or refine its responses based on human judgment and preferences.
# MAGIC
# MAGIC Key Aspects of Human Feedback Loop in LLM Model Review:
# MAGIC Evaluation of Model Outputs:
# MAGIC
# MAGIC Human reviewers assess the quality, accuracy, relevance, and appropriateness of the model's responses. This evaluation can include checking for factual correctness, coherence, ethical considerations, and whether the response aligns with user intent.
# MAGIC Reinforcement Learning from Human Feedback (RLHF):
# MAGIC
# MAGIC In RLHF, human feedback is used as part of the training process. Human reviewers rank or score the outputs of the model, and this feedback is used to adjust the model's parameters. The goal is to make the model generate responses that are more aligned with human preferences and expectations.
# MAGIC Continuous Improvement:
# MAGIC
# MAGIC The human feedback loop is often an ongoing process, where models are continually evaluated and refined based on new feedback. This iterative process helps the model adapt to new information, changing standards, or more nuanced human preferences.
# MAGIC Use in Safety and Ethics:
# MAGIC
# MAGIC Human feedback is particularly important in ensuring that the model's outputs are safe, ethical, and free from bias. Humans can provide context-sensitive judgments that help prevent harmful or inappropriate content from being generated by the model.
# MAGIC Customization and Personalization:
# MAGIC
# MAGIC Human feedback can also be used to tailor models to specific domains, industries, or user groups. By incorporating feedback from experts or targeted users, the model can be fine-tuned to meet particular needs.
# MAGIC Example:
# MAGIC In a typical workflow, an LLM might generate several possible responses to a user query. Human evaluators would then rank these responses based on criteria such as accuracy, helpfulness, and tone. The model then uses this ranking to adjust its future outputs, aiming to produce responses that more closely match the top-ranked examples.
# MAGIC
# MAGIC Importance:
# MAGIC The human feedback loop is crucial for aligning LLMs with human values and ensuring they operate effectively in real-world applications. It allows for the incorporation of human judgment into AI systems, making them more reliable, ethical, and user-friendly.

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


# COMMAND ----------

# agent_deployment_info = get_or_create_agent_review(mdl_version, reviewer_instruction)

agent_deployment_info = deploy_agent(mdl_version, reviewer_instruction)

#TODO: these methods need to be updated such that future iterations identify that a review app already exists
