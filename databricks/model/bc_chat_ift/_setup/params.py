# Databricks notebook source
#from databricks.connect import DatabricksSession
import os
import json
import pandas as pd
from databricks.sdk.runtime import dbutils
#from utils import get_current_cluster_id

DATABRICKS_TOKEN: str = os.environ.get('DATABRICKS_TOKEN', dbutils.secrets.get("yen_hls_azure", "token"))
DATABRICKS_HOST: str = os.environ.get('DATABRICKS_HOST', f"https://{json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())['tags']['browserHostName']}")
BASE_URL: str = f"{DATABRICKS_HOST}serving-endpoints"


# if not in Databricks notebook but in IDE
if not os.environ.get('DATABRICKS_RUNTIME_VERSION'):
   spark = DatabricksSession.builder.remote(
      host       = DATABRICKS_HOST,
      token      = DATABRICKS_TOKEN,
      cluster_id = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']
   ).getOrCreate()