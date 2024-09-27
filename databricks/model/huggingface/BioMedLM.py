# Databricks notebook source
# MAGIC %md
# MAGIC This notebook will make a HF served model using the hf_cache class

# COMMAND ----------

# MAGIC %run ./_setup/setup_hf_cache

# COMMAND ----------


# This section of code would be run in individual notebooks.
# We want to create a hf_model_run that has all configurations such that it is a single stand alone class that no longer needs the 
# Parent class that instantiated it.
MODEL_NAME = "stanford-crfm/BioMedLM"
MODEL_REVISION = "3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f"

hf_model_run = hf_cache.experiment.get_or_create_hf_model_run(hf_model_name = MODEL_NAME,
                                                              hf_model_revision = MODEL_REVISION)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Download a snapshot
# MAGIC
# MAGIC Our first task will be to download our model so we can work off a local cache.

# COMMAND ----------

hf_model_run.hf_cache_snapshot_download()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Create model example_input and parameters
# MAGIC
# MAGIC

# COMMAND ----------

hf_model_run.local_hf_hub_cache

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

snapshot_path = '/root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f'

model = AutoModelForCausalLM.from_pretrained(snapshot_path, 
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot_path, padding_side='left')

# COMMAND ----------

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

# MAGIC %sh
# MAGIC ls -la /root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f

# COMMAND ----------

from sentence_transformers import SentenceTransformer

import mlflow
import mlflow.sentence_transformers

model = SentenceTransformer("all-MiniLM-L6-v2")

example_sentences = ["This is a sentence.", "This is another sentence."]

# Define the signature
signature = mlflow.models.infer_signature(
    model_input=example_sentences,
    model_output=model.encode(example_sentences),
)

# Log the model using mlflow
with mlflow.start_run():
    logged_model = mlflow.sentence_transformers.log_model(
        model=model,
        artifact_path="sbert_model",
        signature=signature,
        input_example=example_sentences,
    )

# COMMAND ----------



# COMMAND ----------

# Now that we have our experiment model run class created with all our requisite configs, we can get into the actual testing and logging

import mlflow
from mlflow.pyfunc import PythonModelContext
from mlflow.types.llm import ChatRequest, ChatMessage, ChatParams, ChatResponse
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, DataType, Array, Object, Property

# To define the path that we want to use, we have to first create the experiment name
# Thus the init of the class will need to initiate a run to get the run_id

context = PythonModelContext(artifacts = {"hub": '/root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM'},
                             model_config = {"hf_model_params":{"repo_id": hf_model_run.model_name,
                                                                "revision": hf_model_run.model_revision},
                                             "from_pretrained_params": {"torch_dtype": "torch.bfloat16",
                                                                        "device_map": "auto",
                                                                        "trust_remote_code": True}})

# COMMAND ----------

# New we need to create a pyfunc model 

# COMMAND ----------

# DO NOT DELETE - This is the log artifact that places drops the snapshot into the right path for later sync.

import mlflow

with mlflow.start_run(experiment_id=hf_model_run.experiment_run,
                      run_name=hf_model_run.model_name.split('/')[-1]) as run:
    mlflow.set_tag("hf_model_name", hf_model_run.model_name)
    mlflow.set_tag("hf_model_revision", hf_model_run.model_revision)
    # TODO: write the log_artifact to find the models directory and log only that
    mlflow.log_artifact('/root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM',
                        artifact_path='hub')
    run_info = run.info

# COMMAND ----------

# MAGIC %sh 
# MAGIC ls -la  /root/.cache/huggingface/hub

# COMMAND ----------

# MAGIC %sh
# MAGIC # New we need to check that the log_artifact method worked. We can inspect volumes directly:
# MAGIC ls -la  /Volumes/biomed_genai/models/huggingface/hub/b15376f5dec84c98a56ee09aa9c3b364/artifacts/hub/models--stanford-crfm--BioMedLM

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la /root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f

# COMMAND ----------

f'/Volumes/biomed_genai/models/huggingface/hub/{run_id}/artifacts/{rel_path}'

# COMMAND ----------

run_info.run_id

# COMMAND ----------

client.log_text(run_info.run_id, "text1", "file1.txt")

# COMMAND ----------

?mlflow.start_run

# COMMAND ----------

# First thing that we'll want to do is download a local copy of the model.
hf_model_run.hf_cache_snapshot_download()

# COMMAND ----------

hf_model_run.volume_hf_hub_model_snapshots
hf_model_run.volume_hf_hub_root + '/models--stanford-crfm--BioMedLM/snapshots/'

# COMMAND ----------

# TODO: create mkdirs process to create the volume snapshots root directory for the given model:
#/root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/

# COMMAND ----------

hf_model_run.model_run_id

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la /root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f
# MAGIC

# COMMAND ----------

?mlflow.log_artifact

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la ../../../../../../../root

# COMMAND ----------

?mlflow.log_artifact

# COMMAND ----------

#dbutils.fs.mkdirs('dbfs:/Volumes/biomed_genai/models/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots')
dbutils.fs.ls('dbfs:/Volumes/biomed_genai/models/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots')

# COMMAND ----------

import mlflow

with mlflow.start_run(experiment_id=hf_model_run.experiment_id,
                      run_id=hf_model_run.model_run_id) as run:
    # Log the artifact
    mlflow.log_artifact('/root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f',
                        artifact_path='dbfs:/Volumes/biomed_genai/models/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots')

# COMMAND ----------

# MAGIC %sh
# MAGIC mkdir -p /root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -la /root/.cache/huggingface/hub/models--stanford-crfm--BioMedLM/snapshots/3e1a0abb814b8398bc34b4b6680ecf2c26d6a66f

# COMMAND ----------

# Now that we have local copy, we want to just inspect that it exists:
