# Databricks notebook source
# DBTITLE 1,Setup Overview
#  This setup file is used as setup across all workflow tasks in pubmed_wf

#  As convention this setup notebook will complete the following tasks:
#    - Install Agent Library Dependencies: Installs required libraries used across all of pubmed_wf tasks
#    - Define & Set Paths: Declare constants PROJECT_ROOT_PATH, PYTHON_ROOT_PATH & set PYTHON_ROOT_PATH in sys.path
#    - Set config_pubmed_wf: Retrieves pubmed_wf config from project configs consolidated at databricks/_config/config_biomed_genai.yaml
#    - Conditionally display a graphic of the deployment architecture

# COMMAND ----------

# DBTITLE 1,Install Agent Library Dependencies
#import subprocess
#commands = [["pip", "install", "openpyxl==3.1.5"],
#            ["pip", "install", "openai==1.37.1"],
#            ["pip", "install", "databricks-agents==0.3.0"]]
#for command in commands:
#    try:
#        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#    except subprocess.CalledProcessError as e:
#        pass

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
    config_pubmed_wf = yaml.safe_load(file).get("workflow")["pubmed_wf"]

# Update config paths from relative to absolute:
config_pubmed_wf['config_ddl_folder'] = path.join(PROJECT_ROOT_PATH, config_pubmed_wf['config_ddl_folder'])
config_pubmed_wf['config_vs_folder'] = path.join(PROJECT_ROOT_PATH, config_pubmed_wf['config_vs_folder'])

# Using our configs, we'll instantiate our workflow config class pubmed_wf:
from biomed_genai.workflow.pubmed_wf.workflow_pubmed_wf import Workflow_pubmed_wf

pubmed_wf = Workflow_pubmed_wf(**config_pubmed_wf)

# COMMAND ----------

html_configs = ('The config class, <i>Workflow_pubmed_wf</i>, has been instantiated as <b>pubmed_wf</b>.<br>' +
                'The instantiation arguments were retrieved from  ' +                                
                f'<a href=#w{PROJECT_ROOT_PATH[2:]}/databricks/_config/config_biomed_genai.yaml>config_biomed_genai.yaml</a> ' +
                'and are reviewable in the dict <b>config_pubmed_wf</b>:')
displayHTML(html_configs)
print(json.dumps(config_pubmed_wf, indent=4))

# COMMAND ----------

dbutils.widgets.dropdown(name="SHOW_TABLE",
                         defaultValue="false",
                         choices=["false", "true"])
dbutils.widgets.dropdown(name="SHOW_WORKFLOW",
                         defaultValue="false",
                         choices=["false", "true"])

# COMMAND ----------

# DBTITLE 1,Conditional Display of Visualizations
# Setup Notebook Widgets
dbutils.widgets.dropdown(name="SHOW_TABLE",
                         defaultValue="false",
                         choices=["false", "true"])
dbutils.widgets.dropdown(name="SHOW_WORKFLOW",
                         defaultValue="false",
                         choices=["false", "true"])

if (dbutils.widgets.getArgument("SHOW_TABLE") == 'true') or (dbutils.widgets.getArgument("SHOW_WORKFLOW") == 'true'):
    if dbutils.widgets.getArgument("SHOW_TABLE") == 'true':
        from biomed_genai.workflow.pubmed_wf.viz_table import workflow_table
        displayHTML(workflow_table(config=pubmed_wf))
        del workflow_table
    if dbutils.widgets.getArgument("SHOW_WORKFLOW") == 'true':
        import subprocess
        commands = [["apt-get", "update"],
                    ["apt-get", "install", "-y", "graphviz"],
                    ["pip", "install", "graphviz"]]
        for command in commands:
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                pass
        from biomed_genai.workflow.pubmed_wf.viz_workflow import workflow_graphic
        displayHTML(workflow_graphic(config=pubmed_wf))
        #del workflow_graphic


# COMMAND ----------

workflow_graphic(config=pubmed_wf)


# COMMAND ----------


