# Databricks notebook source
# DBTITLE 1,Set Workflow Arguments as Constants
import yaml
from os import path

# To allow providing configurations using relative path from root, we include a convenience variable, APP_ROOT_PATH
# Also for ease of debudding, we reassign any relative paths to absolute paths

_nb_path_lst = ("/Workspace" + dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)).split('/')

APP_ROOT_PATH = path.abspath('/'.join(_nb_path_lst[0:_nb_path_lst.index('databricks')]))
PYTHON_ROOT_PATH = path.join(APP_ROOT_PATH, "python")

# The following are the available workflow configurations that are set in config_workflow.yaml
# The workflow code is writen so that the following are argumented:
#        * Catalog
#        * Schemas (there will be 3 using a medallion architecture renamed raw, curated, processed)
#        * Workflow DDL folder (can be an absolute path or relational path from project root)
#        * Workflow Vector Search folder (can be an absolute path or relational path from project root)
# The workflow code is written so that the following are static:
#        * Table Names (with UC namespace derived from catalog and schema arguments)
#        * Volume Paths (with UC namespace derived from catalog and schema arguments)
#        * Workflow DDL file names (with paths derived from application DDL folder arguments)
#        * Workflow Vector Search file names (with paths derived from application DDL folder arguments)

# The priority of argument assignment is:
#  - kwarg (Used when we write our workflow as a job)
#  - yaml (Used when run interactively)
#  - local default (Used when interactive and yaml is missing)

with open(path.join(APP_ROOT_PATH,'databricks/workflow/config/config_workflow.yaml'), 'r') as file:
    _setup_workflow_yaml = yaml.safe_load(file)

_setup_workflow_kwargs = dbutils.widgets.getAll()

def get_config(key:str, default:str) -> str:
    return _setup_workflow_kwargs.get(key, _setup_workflow_yaml.get(key, default))

APP_CATALOG          = get_config("APP_CATALOG", "biomed_genai")
APP_RAW_SCHEMA       = get_config("APP_RAW_SCHEMA", "raw")
APP_CURATED_SCHEMA   = get_config("APP_CURATED_SCHEMA", "curated")
APP_PROCESSED_SCHEMA = get_config("APP_PROCESSED_SCHEMA", "processed")

# Paths from application root for workflow component configuration files
ddl_folder = get_config("APP_CONFIG_DDL_FOLDER", "databricks/workflow/config/ddl")
APP_CONFIG_DDL_FOLDER = path.abspath(ddl_folder) if path.exists(ddl_folder) else path.join(APP_ROOT_PATH, ddl_folder)

vs_folder = get_config("APP_CONFIG_VS_FOLDER", "databricks/workflow/config/vector_search")
APP_CONFIG_VS_FOLDER = path.abspath(vs_folder) if path.exists(vs_folder) else path.join(APP_ROOT_PATH, vs_folder)

del _nb_path_lst, path, _setup_workflow_yaml, _setup_workflow_kwargs, get_config, yaml

# COMMAND ----------

# DBTITLE 1,Instantiate BioMedWorkflowConfig with Constants
import sys

# Add modular code assets to python path so it can be imported
if PYTHON_ROOT_PATH not in sys.path:
    sys.path.insert(0, PYTHON_ROOT_PATH)

from biomed_workflow.config import BioMedWorkflowConfig

biomed = BioMedWorkflowConfig(_catalog_name = APP_CATALOG,
                              _schema_raw_name = APP_RAW_SCHEMA,
                              _schema_curated_name = APP_CURATED_SCHEMA,
                              _schema_processed_name = APP_PROCESSED_SCHEMA,
                              _config_sql_folder = APP_CONFIG_DDL_FOLDER,
                              _config_json_folder = APP_CONFIG_VS_FOLDER)

del BioMedWorkflowConfig

# COMMAND ----------

# DBTITLE 1,Widget Definitions
# Setting the variable biomed above is all that is required for running the workflow notebooks
# However, developers can optionally include visuals in notebooks which we will argument with widgets

dbutils.widgets.dropdown(name="SHOW_TABLE",
                         defaultValue="false",
                         choices=["false", "true"])
dbutils.widgets.dropdown(name="SHOW_GRAPHIC",
                         defaultValue="false",
                         choices=["false", "true"])

# COMMAND ----------

# DBTITLE 1,Conditional Display of Visualizations
if (dbutils.widgets.getArgument("SHOW_TABLE") == 'true') or (dbutils.widgets.getArgument("SHOW_GRAPHIC") == 'true'):
    if dbutils.widgets.getArgument("SHOW_TABLE") == 'true':
        from biomed_workflow.visualizations import workflow_table
        displayHTML(workflow_table(config=biomed))
        del workflow_table
    if dbutils.widgets.getArgument("SHOW_GRAPHIC") == 'true':
        import subprocess
        commands = [["apt-get", "update"],
                    ["apt-get", "install", "-y", "graphviz"],
                    ["pip", "install", "graphviz"]]
        for command in commands:
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                pass
        from biomed_workflow.visualizations import workflow_graphic
        displayHTML(workflow_graphic(config=biomed))
        del workflow_graphic
