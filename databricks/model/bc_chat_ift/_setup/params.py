from databricks.connect import DatabricksSession
import os
import pandas as pd

# Databricks notebook source
DATABRICKS_TOKEN: str = os.getenv('DATABRICKS_TOKEN')
OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
BASE_URL: str = f"{os.getenv('DATABRICKS_HOST')}serving-endpoints"
model = 'databricks-meta-llama-3.1-405b-instruct'
TEMPERATURE: float = 0.7

spark = DatabricksSession.builder.remote(
   host       = os.getenv('DATABRICKS_HOST'),
   token      = os.getenv('DATABRICKS_PAT'),
   cluster_id = os.getenv('CLUSTER_ID')
).getOrCreate()
