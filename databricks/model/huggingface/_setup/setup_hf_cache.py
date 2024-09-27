# Databricks notebook source
# DBTITLE 1,Setup Overview
#  This setup file is used as setup across all huggingface model sync and model registration 

#  As convention this setup notebook will complete the following tasks:
# ...

# COMMAND ----------

# DBTITLE 1,Install Agent Library Dependencies
# TODO: want to get databricks_vectorsearch==0.39 dependent libraries out of share configs to avoid pip install

import subprocess
commands = [["pip", "install", "huggingface_hub==0.19.4"],
            ["pip", "install", "transformers==4.36.1"],
            ["pip", "install", "databricks_vectorsearch==0.39"]]
for command in commands:
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        pass

# COMMAND ----------

# DBTITLE 1,Define & Set Paths
import sys
from os import path

# Define absolute paths as python CONSTANTS
_nb_path_lst = ("/Workspace" + dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)).split('/')
PROJECT_ROOT_PATH = path.abspath('/'.join(_nb_path_lst[0:_nb_path_lst.index('databricks')]))
PYTHON_ROOT_PATH = path.join(PROJECT_ROOT_PATH, "python")
del _nb_path_lst

# Add PYTHON_ROOT_PATH to python path
if PYTHON_ROOT_PATH not in sys.path:
    sys.path.insert(0, PYTHON_ROOT_PATH)

# COMMAND ----------

# DBTITLE 1,Set bc_qa_chat_config & bc_qa_chat
import yaml
import json

with open(path.join(PROJECT_ROOT_PATH,'databricks/_config/config_biomed_genai.yaml'), 'r') as file:
    config_hf_cache = yaml.safe_load(file).get("model")["hf_cache"]
    
config_hf_cache['config_ddl_folder'] = f"{PROJECT_ROOT_PATH}/{config_hf_cache['config_ddl_folder']}"
# Using our configs, we'll instantiate our HF_Cache class, hf_cache:

# WIP, once we have the dataclass complete, we'll go ahead and call from library instead of hardcoding in the setup notebook
from biomed_genai.model.huggingface.model_hf_config import Model_hf_cache
hf_cache = Model_hf_cache(**config_hf_cache)

html_configs = ('The config class, <i>HF_Cachet</i>, has been instantiated as <b>hf_cache</b>.<br>' +
                'The instantiation arguments were retrieved from  ' +                                
                f'<a href=#w{PROJECT_ROOT_PATH[2:]}/databricks/_config/config_biomed_genai.yaml>config_biomed_genai.yaml</a> ' +
                'and are reviewable in the dict <b>config_hf_cache</b>:')
displayHTML(html_configs)
print(json.dumps(config_hf_cache, indent=4))

# COMMAND ----------

# DBTITLE 1,Conditional Display of Visualizations
# TODO: it will be helpful to show the curation for this model with hyperlinks even if it only includes the following:
# 1 - huggingface repository
# 2 - MLFLow experiment run
# 3 - the huggingface cache target directory
# 4 - the registered model in UC

# NOTE: It will have to be a separate visualization for the curataition that happens doing hf model fine-tuning and we will want to establish conventions for how that will be organized within a experiemnt run (meaning we may want to consider nesting for intermediate training artifacts)
