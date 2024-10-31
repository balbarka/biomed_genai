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
# Brad, TODO: will want to have current DBR version checked for ML and min version - 15.4 LTS ML

import subprocess
commands = [["pip", "install", "mlflow==2.17"],
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

# TODO: If we are going to have dependencies between assemblages, we should do the following to simplify assemblage class instantiations
#          - Create a config class classmethod to instantiate from a config file absolute path, PROJECT_ROOT_PATH, calling_nb(?) 
#          - Within class with assemblage dependencies, instantiate all dependent assemblages
#          - Pull only the required dependent assemblage components that are needed

# For now, since there are assemblage dependencies, we'll be instantiating multiple assemblages
# This convention ensure that dependent objects are only instantiated by the assemblage they are assigned
with open(path.join(PROJECT_ROOT_PATH,'databricks/_config/config_biomed_genai.yaml'), 'r') as file:
    config_yaml = yaml.safe_load(file)
    config_pubmed_wf = config_yaml.get("workflow")["pubmed_wf"]
    config_bc_qa_chat = config_yaml.get("agent")["bc_qa_chat"]
    config_bc_chat_ift = config_yaml.get("model")["bc_chat_ift"]

# NOTE: experiments_workspace_folder is expecetd to be a root path within /Workspace and is therefor not included in abs path conversions
# Update ALL config paths from relative to absolute:
config_pubmed_wf['config_ddl_folder'] = path.join(PROJECT_ROOT_PATH, config_pubmed_wf['config_ddl_folder'])
config_pubmed_wf['config_vs_folder'] = path.join(PROJECT_ROOT_PATH, config_pubmed_wf['config_vs_folder'])
config_bc_qa_chat['config_ddl_folder'] = path.join(PROJECT_ROOT_PATH, config_bc_qa_chat['config_ddl_folder'])
config_bc_chat_ift['config_ddl_folder'] = path.join(PROJECT_ROOT_PATH, config_bc_chat_ift['config_ddl_folder'])

# Using our configs, we'll instantiate our workflow config class pubmed_wf:
from biomed_genai.workflow.pubmed_wf.workflow_pubmed_wf import Workflow_pubmed_wf
from biomed_genai.agent.bc_qa_chat.agent_bc_qa_chat import Agent_model_bc_qa_chat
from biomed_genai.model.bc_chat_ift.model_bc_chat_ift import Model_bc_chat_ift

config_bc_chat_ift['source_table'] = Agent_model_bc_qa_chat(**config_bc_qa_chat).experiment.eval_ds
config_bc_chat_ift['eval_ds'] = Workflow_pubmed_wf(**config_pubmed_wf).processed_articles_content

bc_chat_ift = Model_bc_chat_ift(**config_bc_chat_ift)

# Display message about configs:
html_configs = ('The config class, <i>Model_bc_chat_ift</i>, has been instantiated as <b>bc_chat_ift</b>.<br>' +
                'The instantiation arguments were retrieved from  ' +                                
                f'<a href=#w{PROJECT_ROOT_PATH[2:]}/databricks/_config/config_biomed_genai.yaml>config_biomed_genai.yaml</a> ' +
                'and are reviewable in the dict <b>config_bc_chat_ift</b>.')
displayHTML(html_configs)
#print(json.dumps(config_bc_chat_ift, indent=4))

# COMMAND ----------

# DBTITLE 1,Conditional Display of Visualizations
# Setup Notebook Widgets
# dbutils.widgets.dropdown(name="SHOW_GOVERNANCE",
#                          defaultValue="false",
#                          choices=["false", "true"])
# dbutils.widgets.dropdown(name="SHOW_AGENT_DEPLOY",
#                          defaultValue="false",
#                          choices=["false", "true"])
# if (dbutils.widgets.getArgument("SHOW_AGENT_DEPLOY") == 'true') or (dbutils.widgets.getArgument("SHOW_GOVERNANCE") == 'true'):
#     import subprocess
#     commands = [["apt-get", "update"],
#                 ["apt-get", "install", "-y", "graphviz"],
#                 ["pip", "install", "graphviz"]]
#     for command in commands:
#         try:
#             subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         except subprocess.CalledProcessError as e:
#             pass
#     if dbutils.widgets.getArgument("SHOW_GOVERNANCE") == 'true':
#         try:
#             from biomed_genai.agent.viz_governance import agent_governance_graphic
#             curr_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
#             displayHTML(agent_governance_graphic(PROJECT_ROOT_PATH, curr_nb_path))
#             #del agent_governance_graphic
#         except:
#             pass
#     if dbutils.widgets.getArgument("SHOW_AGENT_DEPLOY") == 'true':
#         try:
#             from biomed_genai.agent.viz_agent_deploy import agent_deploy_graphic
#             displayHTML(agent_deploy_graphic(bc_qa_chat))
#             #del agent_deploy_graphic
#         except:
#             pass

