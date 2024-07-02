# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # BioMedConfig Instantiation
# MAGIC
# MAGIC To make the configurations for our BioMed GenAI workflow more straight forward to comprehend we have wrapped a lot of conventience functions into a python module configuration dataclass, **BioMedConfig**. This notebook will be deticated to only going through how configurations for this workflow are managed since they are a bit different than typical configuration libarary approaches.
# MAGIC
# MAGIC This notebook isn't critical to demo and can be skipped if no overview is necessary.
# MAGIC
# MAGIC Setup is intended to be executed using [%run](https://docs.databricks.com/en/notebooks/notebook-workflows.html). Which is used to run our setup notebook, <a href="$./config/setup" target="_blank">./config/setup</a>. That setup notebook will include two python modules (<a href="$../python/biomed_genai/config.py" target="_blank">config.py</a> & <a href="$../python/biomed_genai/visualizations.py" target="_blank">visualizations.py</a>) which it will import using [importlib](https://docs.python.org/3/library/importlib.html). Thus every notebook will have the same configurations due to a single run call and the following dependencies:
# MAGIC
# MAGIC  * <a href="$./config/setup" target="_blank">databricks/config/setup</a> - Notebook to initialize config class `biomed` with application configs and the following dependencies:
# MAGIC    * <a href="$../python/biomed_genai/config.py" target="_blank">python/biomed_genai/config.py</a>
# MAGIC    * <a href="$../python/biomed_genai/visualizations.py" target="_blank">python/biomed_genai/visualizations.py</a>
# MAGIC
# MAGIC ## BioMedConfig Workflow Arguments
# MAGIC
# MAGIC This workflow has many fixed names to facilitate ease of development. However, there are a few arguments that have been left configurable that can be modified directly in cell 1 of <a href="$./config/setup" target="_blank">databricks/config/setup</a>:
# MAGIC
# MAGIC | Config Constant          | Description                      |
# MAGIC | ------------------------ | -------------------------------- |
# MAGIC | `APP_CATALOG`            | Defaults to **biomed_genai**. This is the name of the catalog to be used for the workflow. This will be created if it does not already exist. |
# MAGIC | `APP_RAW_SCHEMA`         | Defaults to **raw**. This is the name of the schema within `APP_CATALOG` where the following entities will be created: </br>&bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**metadata_xml** (Table) - All available xml articles from PMC and ingest status.</br>&bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**search_hist** (Table) - History of all searches for download. </br> &bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**articles** (Volume) - The blob storage of all raw \*.xml article files. </br>&bull; {`APP_CATALOG`}.{`APP_RAW_SCHEMA`}.**_checkpoints** (Volume) - The checkpoints folder of for **metadata_xml**.
# MAGIC | `APP_CURATED_SCHEMA`     | Defaults to **curated**. This is the name of the schema within `APP_CATALOG` where the following entities will be created: </br>&bull; {`APP_CATALOG`}.{`APP_CURATED_SCHEMA`}**.articles_xml** (Table) - Downloaded article sections stored a delta table records.</br>&bull; {`APP_CATALOG`}.{`APP_CURATED_SCHEMA`}.**_checkpoints** (Volume) - The checkpoints folder of for **articles_xml**.
# MAGIC | `APP_PROCESSED_SCHEMA`   | Defaults to **processed**. This is the name of the schema within `APP_CATALOG` where the following entities will be created: </br>&bull; {`APP_CATALOG`}.{`APP_PROCESSED_SCHEMA`}**.articles_content** (Table) - Parsed and chunked articles content.</br>&bull; {`APP_CATALOG`}.{`APP_PROCESSED_SCHEMA`}.**_checkpoints** (Volume) - The checkpoints folder of for **articles_content**.
# MAGIC | `APP_CONFIG_SQL_FOLDER`  | Defaults to **databricks/config/ddl** will be the directory containing the following \*.sql ddl files: </br>&bull; **CREATE_CATALOG_biomed_pipeline.sql** </br>&bull; **CREATE_SCHEMA_raw.sql** </br>&bull; **CREATE_SCHEMA_curated.sql**</br>&bull; **CREATE_SCHEMA_processed.sql** </br>&bull; **CREATE_TABLE_raw_metadata_xml.sql** </br>&bull; **CREATE_VOLUME_raw_checkpoints.sql** </br>&bull; **CREATE_TABLE_raw_search_hist.sql** </br>&bull; **CREATE_VOLUME_raw_articles_xml.sql** </br>&bull; **CREATE_TABLE_curated_articles_xml.sql** </br>&bull; **CREATE_VOLUME_curated_checkpoints.sql** </br>&bull; **CREATE_TABLE_processed_articles_content.sql** </br>&bull; **CREATE_VOLUME_processed_checkpoints.sql** |
# MAGIC | `APP_CONFIG_JSON_FOLDER` | TODO: List dataset and VectorStore entities |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### `SHOW_TABLE`
# MAGIC
# MAGIC The best way to look at the entities in the class is to run the setup with the SHOW_TABLE argument set to true:

# COMMAND ----------

# MAGIC %run ./config/setup $SHOW_TABLE=true $SHOW_GRAPH=false

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The table format gives a listing of all the major workflow entities this application will maintain. Those entities will be tables or volumes (in case the the data is itself a collection of files). Let's first take a look at the table class. 

# COMMAND ----------

# As convenience feature, Delta Table (dt) and Spark DataFrame (df) instances of the UC asset are included as cached_properies
print(biomed.raw_metadata_xml.df.__class__)
biomed.raw_metadata_xml.df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now let's take a look at the volume class

# COMMAND ----------

biomed.raw_articles_xml.path

# COMMAND ----------

# MAGIC %md
# MAGIC ### `SHOW_TABLE`
# MAGIC
# MAGIC The best way to look at how the entities are interrelated in the workflow we can run the setup with the `SHOW_GRAPHIC` argument set to true:
# MAGIC

# COMMAND ----------

# MAGIC %run ./config/setup $SHOW_TABLE=false $SHOW_GRAPHIC=true

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC So the setup of BioMedConfig will only define each of these entities. What it will not do is actually populate those with jobs. Each of the following notebooks we populate a single entity and it is composed this way to more easily step through this data curation.
# MAGIC
# MAGIC  * <a href="$./00_BioMedConfig" target="_blank">00_BioMedConfig</a> (This notebook)
# MAGIC  * <a href="$./0x_xxx" target="_blank">01_xxx</a>
# MAGIC  * <a href="$./0x_xxx" target="_blank">02_xxx</a>
# MAGIC  * <a href="$./0x_xxx" target="_blank">03_xxx</a>
# MAGIC  * <a href="$./0x_xxx" target="_blank">04_xxx</a>
# MAGIC  * <a href="$./0x_xxx" target="_blank">05_xxx</a>
# MAGIC  * <a href="$./0x_xxx" target="_blank">06_xxx</a>
