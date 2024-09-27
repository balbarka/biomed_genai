# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # llama3-8B HF Model Sync
# MAGIC
# MAGIC [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) is a popular model from Meta. This notebook will save a current snapshot to databricks volumes and create a registered model making hugging face model sources easier to evaluate within `biomed_genai`.

# COMMAND ----------

# MAGIC %run ./_setup/setup_hf_cache

# COMMAND ----------

?mlflow.create_experiment

# COMMAND ----------

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "e1945c40cd546c78e41f1151f4db032b271faeaa"
    

# COMMAND ----------

from functools import cached_property
from dataclasses import dataclass

class WS_GenAI_HF_Experiment_Run():
    """This class will be created from a WS_GenAI_HF_Experiment ensuring that all paths have already been instantiated"""
    experiment_id: str
    model_run_id: str
    model_name: str
    model_revision: str
    experiment_ws_path: str
    volume_hf_hub_artifact_path: str
    hf_login_token_secret: str
    local_hf_hub_cache: str = "/root/.cache/huggingface/hub"

    @property
    def _hf_login_token(self):
        secret_scope, secret_key = [s for s in self.hf_login_token_secret.translate(str.maketrans("", "", "{}")).split("/")][-2:]
        return dbutils.secrets.get(secret_scope, secret_key)

    @cached_property
    def rev_folder(self) -> str:    
        return '/'.join(['', '--'.join(['models',] +  self.model_name.split('/')), 'snapshots', self.model_revision])
    
    @cached_property
    def cache_src(self) -> str:
        return self.volume_hf_hub_artifact_path + self.rev_folder

    @cached_property
    def cache_tgt(self) -> str:
        return self.local_hf_hub_cache + self.rev_folder
    
    @cached_property
    def hf_url_commit(self) -> str:
        return f'{self.hf_url_model_card}/commit/{self.model_revision}'
    
    @cached_property
    def hf_url_model_card(self) -> str:
        return f'https://huggingface.co/{self.model_name}'

    def hf_cache_snapshot_download(self):
        """This will sync the volume snapshot for the given model_name, model_revision"""
        try: 
            snapshot_location = snapshot_download(repo_id=self.model_name,
                                                  revision=self.model_revision,
                                                  local_dir_use_symlinks=False,
                                                  cache_dir=self.volume_hf_hub_cache,
                                                  token=self._hf_login_token,
                                                  resume_download=True,)
            print(f"Successfully completed up-to-date download of model {self.model_name}, revision {self.model_revision} " +
                  f"in {self.volume_hf_hub_cache}")
        except Exception as e: 
            print(f"Error: {e}")

    def hf_cache_artifact_download(self):
        """This will overwrite local cache and copy experiment artifacts into local cache"""
        pass

    def hf_cache_volume_sync(self, model_name: str = None, model_revision: str = None, sync_volume = False):
        if sync_volume:
            self.sync_volume_snapshot(model_name, model_revision)
        self._sync_local_snapshot(model_name, model_revision)

    def start_log_model_run(self):
        """Hidden method to sync the local snapshot for the given model_name, model_revision"""
        return mlflow.start_run(run_id=model.model_run_id)

# COMMAND ----------



# COMMAND ----------

config_hf_cache

# COMMAND ----------

# DBTITLE 1,Notebook Scoped Configurations
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "e1945c40cd546c78e41f1151f4db032b271faeaa"

    experiment_id: str
    model_run_id: str
    model_name: str
    model_revision: str
    experiment_ws_path: str
    volume_hf_hub_artifact_path: str
    hf_login_token_secret: str
    local_hf_hub_cache: str = "/root/.cache/huggingface/hub"



    "volume_hf_hub_cache_catalog": "biomed_genai",
    "volume_hf_hub_cache_schema": "models",
    "volume_hf_hub_cache_volume": "huggingface",
    "volume_hf_hub_cache_dir": "hub",
    "local_hf_hub_cache": "/root/.cache/huggingface/hub",
    "hf_login_token_secret": "{{secrets/biomed_genai/hf_login_token}}"

    """This experiment has a convention where every huggingface model name, model revision is an experiment run."""

    experiment_ws_path: str
    volume_hf_hub_root = f'/Volumes/{}.
    hf_login_token_secret: str
    local_hf_hub_cache: str = "/root/.cache/huggingface/hub"



# hf_cache.model_name=MODEL_NAME
# hf_cache.model_revision=MODEL_REVISION

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # MLFlow PythonModelContext
# MAGIC
# MAGIC When working with MLFlow pyfunc models, there is a complexity of needing to test code local artifacts, but that path needs to be relative to the deployed serving container once in model serving. To hnadle this change in path due to execution environment, MLFlow uses

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModelContext
from mlflow.types.llm import ChatRequest, ChatMessage, ChatParams, ChatResponse
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType, Array, Object, Property

# To define the path that we want to use, we have to first create the experiment name
# Thus the init of the class will need to initiate a run to get the run_id

run_id = '1b9586c37f3d4dac954a4ca0c15df524'
HF_CACHE_HUB = f'{hf_cache.volume_hf_hub_cache}/experiments/{run_id}/artifacts/'
"/Volumes/biomed_genai/models/huggingface/experiments/1b9586c37f3d4dac954a4ca0c15df524/artifacts/"

context = PythonModelContext(artifacts = {"HF_CACHE_HUB": HF_CACHE_HUB},
                             model_config = {"hf_model_params":{"repo_id": hf_cache.model_name,
                                                                "revision": hf_cache.model_revision},
                                             "from_pretrained_params": {"torch_dtype": "torch.bfloat16",
                                                                        "device_map": "auto",
                                                                        "trust_remote_code": True}})


# COMMAND ----------

HF_CACHE_HUB

# COMMAND ----------

dbutils.fs.ls('dbfs:/Volumes//biomed_genai/models/huggingface/experiments/1b9586c37f3d4dac954a4ca0c15df524/artifacts')

# COMMAND ----------

hf_cache.local

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC ls -la /Volumes/biomed_genai/models/huggingface/hub/experiments/1b9586c37f3d4dac954a4ca0c15df524/artifacts/

# COMMAND ----------

# We need to make the hub_cache map to an artifact path for the experiment we just created
snapshot_location = snapshot_download(repo_id=hf_cache.model_name,
                                      revision=hf_cache.model_revision,
                                      local_dir_use_symlinks=False,
                                      cache_dir=HF_CACHE_HUB,
                                      token=hf_cache._hf_login_token,
                                      resume_download=True,)

# COMMAND ----------

from mlflow.artifacts import download_artifacts

    run_id: ID of the MLflow Run containing the artifacts. Exactly one of ``run_id`` or
        ``artifact_uri`` must be specified.
    artifact_path: (For use with ``run_id``) If specified, a path relative to the MLflow
        Run's root directory containing the artifacts to download.

xxx = download_artifacts(run_id=run_id,
                         artifact_path="")

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModelContext
from mlflow.types.llm import ChatRequest, ChatMessage, ChatParams, ChatResponse
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType, Array, Object, Property

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

agent_signature = ModelSignature(inputs=Schema([ColSpec(Array(Object(
                                 properties=[Property(name="content", dtype=DataType.string),
                                             Property(name="role", dtype=DataType.string),])), "messages")]),
                                 outputs=Schema([ColSpec(DataType.string)]))

context = PythonModelContext(artifacts={},
                             model_config=model_config)

# COMMAND ----------

# MAGIC %md
# MAGIC # Tested with Git Repos
# MAGIC Include project root determination with workspace files and git folders

# COMMAND ----------

from huggingface_hub import login, snapshot_download
import shutil

login(
    token=dbutils.secrets.get(
        scope=f"{CONFIG.secret_scope}", key=f"{CONFIG.secret_key}"
    ),
    add_to_git_credential=True,
)

# snapshot_download will download any remaining files for a given repo snapshot
# resume_download = True due to timeouts for additional files after weights are downloaded. Typically just needs to rerun and will resume where it failed to download.
try: 
    snapshot_location = snapshot_download(
      repo_id=MODEL_NAME,
      revision=MODEL_REVISION,
      local_dir_use_symlinks=False,
      cache_dir=VOLUME_HUGGINGFACE_HUB_CACHE,
      token=dbutils.secrets.get(
          scope=f"{CONFIG.secret_scope}", key=f"{CONFIG.secret_key}"
      ),
      resume_download=True,
    )
    print(f"Successfully saved the model {MODEL_NAME} in {VOLUME_HUGGINGFACE_HUB_CACHE}") 
    
# Corrected the catch syntax to except and fixed the exception handling syntax
except Exception as e: 
    print(f"Error: {e}")

# COMMAND ----------

# DBTITLE 1,Copy to local huggingface cache if needed
import shutil

# Copy volume cache to local cache if not already there
if not os.path.exists(cache_tgt):
    try: 
        print(f"Loading model from {cache_src} to {cache_tgt}.")
        snapshots_dir = '/'.join(cache_tgt.split('/')[:-1])
        if not os.path.exists(snapshots_dir):
            os.makedirs(snapshots_dir)
            shutil.copytree(cache_src, cache_tgt) 
        print(f"Successfully loaded model from {cache_src} to {cache_tgt}!")
    except Exception as e: 
        print(f"Error: {e}")
        

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la /Volumes/biomed_genai/models/huggingface/hub

# COMMAND ----------

import os

os.environ["TRANSFORMERS_CACHE"] = "/Volumes/biomed_genai/models/huggingface/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/Volumes/biomed_genai/models/huggingface/hub"
os.environ["HF_HOME"] = "/Volumes/biomed_genai/models/huggingface/hub"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"

# COMMAND ----------

cache_src =  "/Volumes/biomed_genai/models/huggingface/hub"

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -la /Volumes/biomed_genai/models/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa

# COMMAND ----------

cache_src = "/Volumes/biomed_genai/models/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(cache_src, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(cache_src)

# COMMAND ----------

# DBTITLE 1,Load model from volumes - MD
# MAGIC %md
# MAGIC # Loading from Volumes will take between 3-8 minutes
# MAGIC ```
# MAGIC import torch
# MAGIC from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# MAGIC
# MAGIC torch.random.manual_seed(0)
# MAGIC
# MAGIC model = AutoModelForCausalLM.from_pretrained(
# MAGIC     cache_src, 
# MAGIC     torch_dtype=torch.bfloat16,
# MAGIC     device_map="auto",
# MAGIC     trust_remote_code=True, 
# MAGIC )
# MAGIC tokenizer = AutoTokenizer.from_pretrained(cache_src)
# MAGIC ```

# COMMAND ----------

hf_cache.sync_local_snapshot()

# COMMAND ----------

cache_tgt = hf_cache.cache_tgt

# COMMAND ----------

# DBTITLE 1,Load model from HF cache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    cache_tgt, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(cache_tgt)


# COMMAND ----------

# DBTITLE 1,Input for llama 3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

messages = [
    {"role": "system", "content": "You are a specialist in healthcare plans"},
    {"role": "user", "content": "What is an HMO and how is it different from a PPO?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# COMMAND ----------

# DBTITLE 1,Results for llama 3

outputs = model.generate(
    input_ids,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC Register model to UC, using MLflow flavor. Load using that flavor

# COMMAND ----------

# MAGIC %md
# MAGIC TODO : SHOW BULK / BATCH PROCESSING LOCALLY

# COMMAND ----------


