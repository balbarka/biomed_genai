# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # BioMed Workflow Config Instantiation
# MAGIC
# MAGIC To make the configurations for our BioMed GenAI workflow more straight forward to comprehend we have wrapped a lot of convenience functions into a python module configuration dataclass, **BioMedWorkflowConfig**. This notebook will be deticated to only going through how the use of an application dataclass which is a non-traditional approach of providing a dataclass instantiated with application configurations in a notebook instead of using the configurations directly in a notebook. This appoach reduces the amount of boilerplate code in a notebook task, improves readibility, and simplifies creation of application entities like tables.
# MAGIC
# MAGIC This notebook doesn't perform a workflow task and thus can be skipped if no overview is necessary. However, there is a benefit to run this if the `biomed_workflow` entities haven't been initialized yet. This will create all workflow entities which can be time consuming for vector search endpoint and vector search index.
# MAGIC
# MAGIC Setup is intended to be executed using [%run](https://docs.databricks.com/en/notebooks/notebook-workflows.html). Which is used to run our setup notebook, <a href="$./config/setup_workflow" target="_blank">./config/setup_workflow</a>. That setup notebook will include two python modules (<a href="$../../python/biomed_workflow/config.py" target="_blank">config.py</a> & <a href="$../../python/biomed_workflow/visualizations.py" target="_blank">visualizations.py</a>) which it will import after adding the python folder they are in to path. Thus every notebook in `genai_workflow` will have the same configurations due to a single run call and the following dependencies:
# MAGIC
# MAGIC  * <a href="$./config/setup_workflow" target="_blank">databricks/config/setup_workflow</a> - Notebook to initialize config class `biomed` with application configs and the following dependencies:
# MAGIC    * <a href="$../../python/biomed_workflow/config.py" target="_blank">python/biomed_workflow/config.py</a>
# MAGIC    * <a href="$../../python/biomed_workflow/visualizations.py" target="_blank">python/biomed_workflow/visualizations.py</a>
# MAGIC
# MAGIC ## BioMedConfig Workflow Arguments
# MAGIC
# MAGIC This workflow has many fixed names to facilitate ease of initial deployment. However, there are a few arguments that have been left configurable that can be modified directly in <a href="$./config/config_workflow.yaml" target="_blank">databricks/config/config_workflow.yaml</a>:
# MAGIC
# MAGIC | Config Constant          | Description                      |
# MAGIC | ------------------------ | -------------------------------- |
# MAGIC | `APP_CATALOG`            | Defaults to **biomed_genai**. This is the name of the catalog to be used for the workflow. This will be created if it does not already exist. |
# MAGIC | `APP_RAW_SCHEMA`         | Defaults to **raw**. This is the name of the schema within `APP_CATALOG` where the following entities will be created: </br>&bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**metadata_xml** (Table) - All available xml articles from PMC and ingest status.</br>&bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**search_hist** (Table) - History of all searches for download. </br> &bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**articles** (Volume) - The blob storage of all raw \*.xml article files. </br>&bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**_checkpoints** (Volume) - The checkpoints folder of for **metadata_xml**.
# MAGIC | `APP_CURATED_SCHEMA`     | Defaults to **curated**. This is the name of the schema within `APP_CATALOG` where the following entities will be created: </br>&bull; {`APP_CATALOG`}.{`APP_CURATED_SCHEMA`}**.articles_xml** (Table) - Downloaded article sections stored a delta table records.</br>&bull; {`APP_CATALOG`}.{`APP_CURATED_SCHEMA`}.**_checkpoints** (Volume) - The checkpoints folder of for **articles_xml**.
# MAGIC | `APP_PROCESSED_SCHEMA`   | Defaults to **processed**. This is the name of the schema within `APP_CATALOG` where the following entities will be created: </br>&bull; {`APP_CATALOG`}.{`APP_PROCESSED_SCHEMA`}**.articles_content** (Table) - Parsed and chunked articles content.</br>&bull; {`APP_CATALOG`}.{`APP_PROCESSED_SCHEMA`}.**_checkpoints** (Volume) - The checkpoints folder of for **articles_content**.
# MAGIC | `APP_CONFIG_SQL_FOLDER`  | Defaults to **databricks/workflow/config/ddl** will be the directory containing the following \*.sql ddl files: </br>&bull; **CREATE_CATALOG_biomed_pipeline.sql** </br>&bull; **CREATE_SCHEMA_raw.sql** </br>&bull; **CREATE_SCHEMA_curated.sql**</br>&bull; **CREATE_SCHEMA_processed.sql** </br>&bull; **CREATE_TABLE_raw_metadata_xml.sql** </br>&bull; **CREATE_VOLUME_raw_checkpoints.sql** </br>&bull; **CREATE_TABLE_raw_search_hist.sql** </br>&bull; **CREATE_VOLUME_raw_articles_xml.sql** </br>&bull; **CREATE_TABLE_curated_articles_xml.sql** </br>&bull; **CREATE_VOLUME_curated_checkpoints.sql** </br>&bull; **CREATE_TABLE_processed_articles_content.sql** </br>&bull; **CREATE_VOLUME_processed_checkpoints.sql** |
# MAGIC | `APP_CONFIG_VS_FOLDER` | Also defaults to **databricks/workflow/config/vector_search** will be the directory containing the following \*.json config files: </br>&bull; **ENDPOINT_biomed.json** </br>&bull; **INDEX_processed_articles_content_vs_index.json** |

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
# MAGIC  * <a href="$./00_BioMedWorkflowConfig" target="_blank">00_BioMedWorkflowConfig</a> (This notebook)
# MAGIC  * <a href="$./01_Metadata_Sync" target="_blank">01_Metadata_Sync</a> Syncs the UC metadata table with PMC. This will ensure that we have a relatively local, fast, current listing of all PMC articles.
# MAGIC  * <a href="$./02_Articles_Ingest" target="_blank">02_Articles_Ingest</a> This notebook runs a job that ingests PMC raw files into a UC volume dependent upon search criteria.
# MAGIC  * <a href="$./03_Curate_Articles" target="_blank">03_Curate_Articles</a> This notebook creates an intermediate delta table with all xml articles. This provides a performant source, much faster than reading individual raw xml files to apply our chunking strategy.
# MAGIC  * <a href="$./04_Chunk_Articles_Content" target="_blank">04_Chunk_Articles_Content</a> This notebook applies a chunking strategy on articles to populate the source delta table for our vector serving index.
# MAGIC  * <a href="$./05_Sync_VectorSearch_Index" target="_blank">05_Sync_VectorSearch_Index</a> This notebook will sync our delta source table to our vector index.
# MAGIC  * <a href="$./06_Create_Sync_Job" target="_blank">06_Create_Sync_Job</a> Being able to run each of our notebooks from a single config is nice, but to capture the dependencies in these workbooks and simplify recurring job execution, we'll create a databricks job.
