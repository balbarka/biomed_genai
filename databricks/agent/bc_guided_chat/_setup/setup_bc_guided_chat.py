# Databricks notebook source
# DBTITLE 1,Setup Overview
#  This setup file is used as setup across all models in the agent application, setup_bc_qa_chat

#  As convention this setup notebook will complete the following tasks:
#    - Install Agent Library Dependencies: Installs required libraries used across all of bc_qa_chat models
#    - Define & Set Paths: Declare constants PROJECT_ROOT_PATH, PYTHON_ROOT_PATH & set PYTHON_ROOT_PATH in sys.path
#    - Set config_bc_qa_chat: Retrieves bc_qa_chat config from project configs consolidated at databricks/_config/config_biomed_genai.yaml
#    - Conditionally display a graphic of the deployment architecture

# COMMAND ----------

# DBTITLE 1,Install Agent Library Dependencies
import subprocess
commands = [["pip", "install", "openpyxl==3.1.5"],
            ["pip", "install", "openai==1.37.1"],
            ["pip", "install", "databricks-agents==0.3.0"],
            ["pip", "install", "pip install mlflow==2.17.0rc0"]]
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
    config_bc_guided_chat = yaml.safe_load(file).get("agent")["bc_guided_chat"]

# Update config paths from relative to absolute:
config_bc_guided_chat['config_ddl_folder'] = path.join(PROJECT_ROOT_PATH, config_bc_guided_chat['config_ddl_folder'])
# NOTE: experiments_workspace_folder is expecetd to be a root path within /Workspace and is therefor not included in abs path conversions

# Using our configs, we'll instantiate our agent config class, bc_qa_chat:
from biomed_genai.agent.bc_guided_chat.agent_bc_guided_chat import Agent_model_bc_guided_chat
bc_guided_chat = Agent_model_bc_guided_chat(**config_bc_guided_chat)

html_configs = ('The config class, <i>Agent_model_bc_guided_chat</i>, has been instantiated as <b>bc_guided_chat</b>.<br>' +
                'The instantiation arguments were retrieved from  ' +                                
                f'<a href=#w{PROJECT_ROOT_PATH[2:]}/databricks/_config/config_biomed_genai.yaml>config_biomed_genai.yaml</a> ' +
                'and are reviewable in the dict <b>config_bc_guided_chat</b>:')
displayHTML(html_configs)
print(json.dumps(config_bc_guided_chat, indent=4))

# COMMAND ----------

# DBTITLE 1,Conditional Display of Visualizations
# Setup Notebook Widgets
dbutils.widgets.dropdown(name="SHOW_NOTEBOOK_TASKS",
                         defaultValue="false",
                         choices=["false", "true"])
dbutils.widgets.dropdown(name="SHOW_AGENT_MODEL",
                         defaultValue="false",
                         choices=["false", "true"])

if (dbutils.widgets.getArgument("SHOW_NOTEBOOK_TASKS") == 'true') or (dbutils.widgets.getArgument("SHOW_AGENT_MODEL") == 'true'):
    import subprocess
    commands = [["apt-get", "update"],
                ["apt-get", "install", "-y", "graphviz"],
                ["pip", "install", "graphviz"]]
    for command in commands:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            pass
    if dbutils.widgets.getArgument("SHOW_NOTEBOOK_TASKS") == 'true':
        try:
            from biomed_genai.agent.bc_guided_chat.viz_agent_model import agent_model_graphic
            displayHTML(agent_model_graphic())
            del agent_model_graphic
        except:
            pass
    if dbutils.widgets.getArgument("SHOW_AGENT_MODEL") == 'true':
        try:
            from biomed_genai.agent.bc_guided_chat.viz_agent_model import agent_model_graphic
            displayHTML(agent_model_graphic())
            del agent_model_graphic
        except:
            pass

# COMMAND ----------


