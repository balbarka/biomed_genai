# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Articles Ingest
# MAGIC
# MAGIC **Objective**: This notebook will use arguments `biomed.raw_metadata`, and `biomed.raw_search_hist` to query PMC for new articles related to our key word search topics of interest. Upon a successful run, `biomed.raw_metadata` and `biomed.raw_search_hist` tables will be updated with search and download metadata and articles will be downloaded to `biomed.raw_articles` folder location.
# MAGIC
# MAGIC This notebook uses convenience functions imported from a <a href="$../../python/biomed_workflow/pmc.py" target="_blank">pmc</a> module. They include:
# MAGIC
# MAGIC | Method | Arguments | Description |
# MAGIC | ------ | --------- | ----------- |
# MAGIC | `search_articles` | [REQUIRED] `keyword`: *str*</br>[OPTIONAL] `min_dte`: *str*</br>[OPTIONAL] `max_dte`: *str* | Uses the PMC search to return list of PMIDs.</br>`min_dte` defaults to `01/01/2022`.</br>`max_dte` defaults to yesterday. |
# MAGIC | `download_pending_articles` | [REQUIRED] `search_hist`: *UC_Table*</br>[REQUIRED] `metadata`: *UC_Table*</br>[REQUIRED] `articles`: *UC_Volume*</br>[REQUIRED] `keywords`: *Union[str, List[str]]*</br>[OPTIONAL] `min_dte`: *str*</br>[OPTIONAL] `max_dte`: *str*  | Downloads to `articles` all pending articles in `metadata` that have</br>not already been downloaded and match `keywords`. Upon completion</br>of successful run, both `metadata` and `search_hist` will be updated.</br>`min_dte` and `max_dte` have the same defaults as used in `search_articles` |
# MAGIC
# MAGIC **NOTE**: While not explicit in the method names, the search and download functions are limited to xml documents only.
# MAGIC
# MAGIC **NOTE**: We are later able to import the `pmc` convenience functions because we have added the path of the modular code to the python path when we run `%run ./config/setup_workflow`.

# COMMAND ----------

# DBTITLE 1,Initialize pubmed_wf Application Class
# MAGIC %run ./_setup/setup_pubmed_wf $SHOW_TABLE=false $SHOW_WORKFLOW=true

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## [OPTIONAL] Inspect `search_articles` helper function
# MAGIC
# MAGIC `search_articles` is example is provided below as it will be a useful function to identify which articles are included in a search. Be aware that the same article can be returned based on different keywords.

# COMMAND ----------

from biomed_genai.workflow.pmc import search_articles

INSPECT_SEARCH_ARTICLES = False

if INSPECT_SEARCH_ARTICLES:
    pmids = search_articles(keyword='breast cancer',
                            min_dte='2024/07/01',
                            max_dte='2024/07/03')
    print("Search PMIDs found: " + str(len(pmids)))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Download Articles using `download_pending_pmids`
# MAGIC
# MAGIC There are two examples provided one where you download with a keyword and one where you download and updates from all previous searches. The later is provided to simplify the eventual creation of an update workflow.
# MAGIC
# MAGIC **NOTE**: When running without keywords **ALL** keywords ever downloaded, will be downloaded if pending though yesterday. This could be a larger job if there are many keyword entries in `search_hist`.

# COMMAND ----------

from biomed_genai.workflow.pmc import download_pending_articles

DOWNLOAD_WITH_KEYWORD = True
DOWNLOAD_WITHOUT_KEYWORD = False

if DOWNLOAD_WITH_KEYWORD:
    articles_status = download_pending_articles(search_hist=pubmed_wf.raw_search_hist, 
                                                metadata=pubmed_wf.raw_metadata_xml, 
                                                articles=pubmed_wf.raw_articles_xml, 
                                                keywords="breast cancer", 
                                                min_dte='2024/06/01',
                                                max_dte='2024/06/03')

if DOWNLOAD_WITHOUT_KEYWORD:
    articles_status = download_pending_articles(search_hist=pubmed_wf.raw_search_hist, 
                                                metadata=pubmed_wf.raw_metadata_xml, 
                                                articles=pubmed_wf.raw_articles_xml,
                                                min_dte='2024/06/01')

display(articles_status.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Inspect `metadata_update` (OPTIONAL)
# MAGIC
# MAGIC Sometimes it's easier to inspect updates using the change data capture functionality. The following section will allow us to inspect all of the records updated between our previous version and our current verion in our `metadata_xml` table.
# MAGIC
# MAGIC The impact of `download_pending_articles` on `metadata_xml` is seen only by the last two fields which are not provided as part of the metadata sync from PMC:
# MAGIC
# MAGIC | Field Name | Description |
# MAGIC | ---------- | ----------- |
# MAGIC | `Status`   | This is the status of the given article which can be assigned one of the following states: </br>- `PENDING` : The article is known to exist, but not downloaded </br>- `DOWNLOADED`: The article is downloaded with an assigned Volume path </br>- `ERROR`: There was an Error in the download process and the download should be corrected manually </br> - `RETRACTED` : The article has been retracted by PMC. |
# MAGIC | `volume_path` | This is the path that the raw file has been extracted to | 

# COMMAND ----------

INSPECT_METADATA_XML_CHANGES = False

if INSPECT_METADATA_XML_CHANGES:
    curr_version = spark.sql(f'DESCRIBE HISTORY {pubmed_wf.raw_metadata_xml.name} LIMIT 1').collect()[0][0]
    prev_version = curr_version - 1
    changes_df = spark.read.format("delta") \
                           .option("readChangeFeed", "true") \
                           .option("startingVersion", prev_version) \
                           .option("endingVersion", curr_version) \
                           .table(pubmed_wf.raw_metadata_xml.name)
    display(changes_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # [OPTIONAL] Inspect `search_hist`
# MAGIC
# MAGIC You will notice if you look at `search_hist` that the entire range of search is captured, not just for the most current update, but for all updates. 

# COMMAND ----------

INSPECT_SEARCH_HIST = False

if INSPECT_SEARCH_HIST:
    display(pubmed_wf.raw_search_hist.df)
