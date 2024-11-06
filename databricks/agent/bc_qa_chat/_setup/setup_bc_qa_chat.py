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
            ["pip", "install", "databricks-agents==0.3.0"]]
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
    config_bc_qa_chat = yaml.safe_load(file).get("agent")["bc_qa_chat"]

# Update config paths from relative to absolute:
config_bc_qa_chat['config_ddl_folder'] = path.join(PROJECT_ROOT_PATH, config_bc_qa_chat['config_ddl_folder'])
# NOTE: experiments_workspace_folder is expecetd to be a root path within /Workspace and is therefor not included in abs path conversions

# To support a convention where the default model name is based upon the producing notebook, we will pass the name of the calling
# notebook name as a configuration parameter
config_bc_qa_chat['default_model_name'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split('/')[-1]

# Using our configs, we'll instantiate our agent config class, bc_qa_chat:
from biomed_genai.agent.bc_qa_chat.agent_bc_qa_chat import Agent_model_bc_qa_chat
bc_qa_chat = Agent_model_bc_qa_chat(**config_bc_qa_chat)

html_configs = ('The config class, <i>Agent_model_bc_qa_chat</i>, has been instantiated as <b>bc_qa_chat</b>.<br>' +
                'The instantiation arguments were retrieved from  ' +                                
                f'<a href=#w{PROJECT_ROOT_PATH[2:]}/databricks/_config/config_biomed_genai.yaml>config_biomed_genai.yaml</a> ' +
                'and are reviewable in the dict <b>config_bc_qa_chat</b>:')
displayHTML(html_configs)
#print(json.dumps(config_bc_qa_chat, indent=4))

# COMMAND ----------

bc_qa_chat


# COMMAND ----------

# DBTITLE 1,Conditional Display of Visualizations
# Setup Notebook Widgets
dbutils.widgets.dropdown(name="SHOW_GOVERNANCE",
                         defaultValue="true",
                         choices=["false", "true"])
dbutils.widgets.dropdown(name="SHOW_AGENT_DEPLOY",
                         defaultValue="true",
                         choices=["false", "true"])

if (dbutils.widgets.getArgument("SHOW_AGENT_DEPLOY") == 'true') or (dbutils.widgets.getArgument("SHOW_GOVERNANCE") == 'true'):
    import subprocess
    commands = [["apt-get", "update"],
                ["apt-get", "install", "-y", "graphviz"],
                ["pip", "install", "graphviz"]]
    for command in commands:
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            pass
    if dbutils.widgets.getArgument("SHOW_GOVERNANCE") == 'true':
        try:
            from biomed_genai.agent.viz_governance import agent_governance_graphic
            curr_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
            displayHTML(agent_governance_graphic(PROJECT_ROOT_PATH, curr_nb_path))
            #del agent_governance_graphic
        except:
            pass
    if dbutils.widgets.getArgument("SHOW_AGENT_DEPLOY") == 'true':
        try:
            from biomed_genai.agent.viz_agent_deploy import agent_deploy_graphic
            displayHTML(agent_deploy_graphic(bc_qa_chat))
            #del agent_deploy_graphic
        except:
            pass

# COMMAND ----------


