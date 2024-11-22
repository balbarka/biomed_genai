# Databricks notebook source
# MAGIC %pip install databricks-sdk[dashboards] databricks-agents
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk.service.dashboards import GenieAPI
from databricks.sdk import WorkspaceClient

ws = WorkspaceClient()
gc = GenieAPI(ws)
gc_cookbook = ws.genie

# COMMAND ----------

xxx = ws.genie

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # `Workflow_pubmed_wf` Config Class Instantiation
# MAGIC
# MAGIC To make the configurations for our BioMed GenAI workflow more straight forward to comprehend we have wrapped a lot of convenience functions into a python module configuration dataclass, **Workflow_pubmed_wf**. This notebook will be deticated to only going through how the use of an application dataclass which is a non-traditional approach of providing a dataclass instantiated with application configurations in a notebook instead of using the configurations directly in a notebook. This appoach reduces the amount of boilerplate code in a notebook task, improves readibility, and simplifies creation of application entities like tables.
# MAGIC
# MAGIC This notebook doesn't perform a workflow task and thus can be skipped if no overview is necessary. However, there is a benefit to run this if the `pubmed_wf` entities haven't been initialized yet. This will create all workflow entities which can be time consuming for vector search endpoint and vector search index.
# MAGIC
# MAGIC Setup is intended to be executed using [%run](https://docs.databricks.com/en/notebooks/notebook-workflows.html). Which is used to run our setup notebook, <a href="$./_setup/setup_pubmed_wf" target="_blank">./_setup/setup_pubmed_wf</a>. That setup notebook uses the following python modules:
# MAGIC  * <a href="$../../../python/biomed_genai/config.py" target="_blank">/biomed_genai/python/biomed_genai/config.py</a>
# MAGIC  * <a href="$../../../python/biomed_genai/retriever/pubmed_wf/workflow_pubmed_wf.py" target="_blank">/biomed_genai/retriever/pubmed_wf/workflow_pubmed_wf.py</a>
# MAGIC  * <a href="$../../../python/biomed_genai/retriever/pubmed_wf/viz_workflow.py" target="_blank">/biomed_genai/retriever/pubmed_wf/viz_workflow.py</a>
# MAGIC
# MAGIC
# MAGIC ## BioMedConfig Workflow Configurations
# MAGIC
# MAGIC This workflow has many fixed names for entities created to facilitate ease of initial deployment. However, there are a few arguments that have been left configurable that can be modified directly in <a href="$../../_config/config_biomed_genai.yaml" target="_blank">databricks/_config/config_biomed_genai.yaml</a>. If you look in the config file you'll see that it is reused across other genai components in this preoject. This is a simple convention to have shared configs across multiple components.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### `SHOW_TABLE`
# MAGIC
# MAGIC The best way to look at the entities in the class is to run the setup with the SHOW_TABLE argument set to true:

# COMMAND ----------

# MAGIC %run ./_setup/setup_pubmed_wf $SHOW_TABLE=true $SHOW_WORKFLOW=false

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The table format gives a listing of all the major workflow entities this application will maintain. Those entities will be tables or volumes (in case the the data is itself a collection of files). Let's first take a look at the table class. 

# COMMAND ----------

# As convenience feature, Delta Table (dt) and Spark DataFrame (df) instances of the UC asset are included as cached_properies
print(pubmed_wf.raw_metadata_xml.df.__class__)
pubmed_wf.raw_metadata_xml.df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now let's take a look at the volume class:

# COMMAND ----------

pubmed_wf.raw_articles_xml.path

# COMMAND ----------

# MAGIC %md
# MAGIC ### `SHOW_WORKFLOW`
# MAGIC
# MAGIC The best way to look at how the entities are interrelated in the workflow we can run the setup with the `SHOW_GRAPHIC` argument set to true:
# MAGIC

# COMMAND ----------

# MAGIC %run ./_setup/setup_pubmed_wf $SHOW_TABLE=false $SHOW_WORKFLOW=true

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC So the setup of BioMedConfig will only define each of these entities. What it will not do is actually populate those with jobs. Each of the following notebooks we populate a single entity and it is composed this way to more easily step through this data curation.
# MAGIC
# MAGIC  * <a href="$./00_pubmed_wf_config" target="_blank">00_pubmed_wf_config</a> (This notebook)
# MAGIC  * <a href="$./01_Metadata_Sync" target="_blank">01_Metadata_Sync</a> Syncs the UC metadata table with PMC. This will ensure that we have a relatively local, fast, current listing of all PMC articles.
# MAGIC  * <a href="$./02_Articles_Ingest" target="_blank">02_Articles_Ingest</a> This notebook runs a job that ingests PMC raw files into a UC volume dependent upon search criteria.
# MAGIC  * <a href="$./03_Curate_Articles" target="_blank">03_Curate_Articles</a> This notebook creates an intermediate delta table with all xml articles. This provides a performant source, much faster than reading individual raw xml files to apply our chunking strategy.
# MAGIC  * <a href="$./04_Chunk_Articles_Content" target="_blank">04_Chunk_Articles_Content</a> This notebook applies a chunking strategy on articles to populate the source delta table for our vector serving index.
# MAGIC  * <a href="$./05_Sync_VectorSearch_Index" target="_blank">05_Sync_VectorSearch_Index</a> This notebook will sync our delta source table to our vector index.
# MAGIC  * <a href="$./06_Create_Sync_Job" target="_blank">06_Create_Sync_Job</a> Being able to run each of our notebooks from a single config is nice, but to capture the dependencies in these workbooks and simplify recurring job execution, we'll create a databricks job.
