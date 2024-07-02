# Databricks notebook source
# DBTITLE 1,Application Configurations
# The following are the available application configurations that can be set for biomed_genai
# The application code is writen so that the following are argumented:
#        * Catalog
#        * Schemas (there will be 3 using a medallion architecture renamed raw, curated, processed)
#        * Application DDL folder (can be an absolute path or relational path from project root)
# The application code is written so that the following are static:
#        * Table Names (with UC namespace derived from catalog and schema arguments)
#        * Volume Paths (with UC namespace derived from catalog and schema arguments)
#        * Application DDL file names (with paths derived from application DDL folder arguments)

APP_CATALOG = "biomed_genai"
APP_RAW_SCHEMA = "raw"
APP_CURATED_SCHEMA = "curated"
APP_PROCESSED_SCHEMA = "processed"

# Path from application root
APP_CONFIG_SQL_FOLDER = 'databricks/config/ddl'
APP_CONFIG_JSON_FOLDER = 'databricks/config/ddl'

# COMMAND ----------

# DBTITLE 1,Assign Absolute Paths
# To allow providing configurations using relative path from root, we include a convenience variable, APP_ROOT_PATH
# Also for ease of debudding, we reassign any relative paths to absolute paths, specifically APP_CONFIG_*_FOLDER

import os

_nb_path_lst = ("/Workspace" + dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None)).split('/')
APP_ROOT_PATH = os.path.abspath('/'.join(_nb_path_lst[0:_nb_path_lst.index('databricks')]))
del _nb_path_lst

# Update config paths so that they are absolute
import os

if os.path.exists(APP_CONFIG_SQL_FOLDER):
    APP_CONFIG_SQL_FOLDER = os.path.abspath(APP_CONFIG_SQL_FOLDER)
else:    
    APP_CONFIG_SQL_FOLDER = os.path.join(APP_ROOT_PATH, APP_CONFIG_SQL_FOLDER)

if os.path.exists(APP_CONFIG_JSON_FOLDER):
    APP_CONFIG_JSON_FOLDER = os.path.abspath(APP_CONFIG_JSON_FOLDER)
else:    
    APP_CONFIG_JSON_FOLDER = os.path.join(APP_ROOT_PATH, APP_CONFIG_JSON_FOLDER)

# COMMAND ----------

# DBTITLE 1,Module Import and BioMedConflig class instantiation
import importlib.util
import sys
import os

# To provide a single source of code that can optionally be packaged, we will use import lib
# The following methos will be used to import the following modules:
#    * config: the required configuration class that will be used throughout the workflow
#    * viz: optional graphics helpful for developer understanding

def import_module_from_root_relative_path(root_path, relative_path):
    absolute_path = os.path.join(root_path, relative_path)
    module_name = os.path.basename(absolute_path).split('.')[0]
    module_dir = os.path.dirname(absolute_path)
    spec = importlib.util.spec_from_file_location(module_name, absolute_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

config = import_module_from_root_relative_path(APP_ROOT_PATH, 'python/biomed_genai/config.py')

# Create a biomed configuration class with arguments set above
biomed = config.BioMedConfig(_catalog_name = APP_CATALOG,
                             _schema_raw_name = APP_RAW_SCHEMA,
                             _schema_curated_name = APP_CURATED_SCHEMA,
                             _schema_processed_name = APP_PROCESSED_SCHEMA,
                             _config_sql_folder = APP_CONFIG_SQL_FOLDER,
                             _config_json_folder = APP_CONFIG_JSON_FOLDER)

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
    viz = import_module_from_root_relative_path(APP_ROOT_PATH, 'python/biomed_genai/visualizations.py')
    if dbutils.widgets.getArgument("SHOW_TABLE") == 'true':
        displayHTML(viz.workflow_table(config=biomed))
    if dbutils.widgets.getArgument("SHOW_GRAPHIC") == 'true':
        import subprocess
        commands = [["apt-get", "update"],
                    ["apt-get", "install", "-y", "graphviz"],
                    ["pip", "install", "graphviz"]]
        for command in commands:
            try:
                #print(f"Running command: {' '.join(command)}")
                #result = subprocess.run(command, check=True, capture_output=True, text=True)
                #print(f"Command succeeded with output:\n{result.stdout}")
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                #print(f"Error occurred while running command: {' '.join(command)}")
                #print(f"Error output:\n{e.stderr}")
                pass
        displayHTML(viz.workflow_graphic(config=biomed))

# COMMAND ----------

# Visualization Scratch Code

#config.raw_metadata_xml.sql_relative_url
#config.raw_metadata_xml.sql_file
#config.raw_metadata_xml.uc_relative_url
#config.raw_metadata_xml.name
#config.raw_metadata_xml.cp.uc_relative_url
#config.raw_metadata_xml.cp.name
#config.raw_search_hist.sql_relative_url
#config.raw_search_hist.sql_file
#config.raw_search_hist.uc_relative_url
#config.raw_search_hist.name
#config.raw_articles_xml.sql_relative_url
#config.raw_articles_xml.sql_file
#config.raw_articles_xml.uc_relative_url
#config.raw_articles_xml.name
#config.curated_articles_xml.sql_relative_url
#config.curated_articles_xml.sql_file
#config.curated_articles_xml.uc_relative_url
#config.curated_articles_xml.name
#config.curated_articles_xml.cp.uc_relative_url
#config.curated_articles_xml.cp.name
#config.processed_articles_content.sql_relative_url
#config.processed_articles_content.sql_file
#config.processed_articles_content.uc_relative_url
#config.processed_articles_content.name
#config.processed_articles_content.cp.uc_relative_url
#config.processed_articles_content.cp.name
