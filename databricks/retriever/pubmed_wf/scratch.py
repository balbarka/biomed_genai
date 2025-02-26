# Databricks notebook source
# MAGIC %run ./_setup/setup_pubmed_wf $SHOW_TABLE=false $SHOW_WORKFLOW=true

# COMMAND ----------

display(pubmed_wf.curated_articles_xml.df)

# COMMAND ----------


