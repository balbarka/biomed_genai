# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Articles Ingest
# MAGIC
# MAGIC **Objective**: This notebook will use arguments `biomed.raw_metadata`, and `biomed.raw_search_hist` to query PMC for new articles related to our key word search topics of interest. Upon a successful run, `biomed.raw_metadata` and `biomed.raw_search_hist` tables will be updated with search and download metadata and articles will be downloaded to `biomed.raw_articles` folder location.
# MAGIC
# MAGIC This notebook uses convenience functions imported from a <a href="$../../../python/biomed_genai/workflow/pubmed_wf/pmc.py" target="_blank">pmc</a> module. They include:
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC | Method | Arguments | Description |
# MAGIC | ------ | --------- | ----------- |
# MAGIC | `search_articles` | [REQUIRED] `keyword`: *str*</br>[OPTIONAL] `min_dte`: *str*</br>[OPTIONAL] `max_dte`: *str* | Uses the PMC search to return list of PMIDs.</br>`min_dte` defaults to `01/01/2022`.</br>`max_dte` defaults to yesterday. |
# MAGIC | `download_pending_articles` | [REQUIRED] `search_hist`: *UC_Table*</br>[REQUIRED] `metadata`: *UC_Table*</br>[REQUIRED] `articles`: *UC_Volume*</br>[REQUIRED] `keywords`: *Union[str, List[str]]*</br>[OPTIONAL] `min_dte`: *str*</br>[OPTIONAL] `max_dte`: *str*  | Downloads to `articles` all pending articles in `metadata` that have</br>not already been downloaded and match `keywords`. Upon completion</br>of successful run, both `metadata` and `search_hist` will be updated.</br>`min_dte` and `max_dte` have the same defaults as used in `search_articles` |
# MAGIC
# MAGIC **NOTE**: While not explicit in the method names, the search and download functions are limited to xml documents only.
# MAGIC
# MAGIC **NOTE**: We are later able to import the `pmc` convenience functions because we have added the path of the modular code to the python path when we run `%run ./_setup/setup_pubmed_wf`.

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

from biomed_genai.workflow.pubmed_wf.pmc import search_articles

INSPECT_SEARCH_ARTICLES = True

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

"""Consolidation of all convenience functions when working with PMC"""

import defusedxml.ElementTree as ET
from typing import Optional, Union, List
from time import sleep
from dataclasses import dataclass
from datetime import datetime
import requests
from requests.models import Response
from pyspark.sql.types import StructType, MapType, StringType, StructField
from bs4 import BeautifulSoup
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
import pandas as pd

from biomed_genai.config import UC_Table, UC_Volume


@dataclass
class PaperSearchResponse:
    response: Response

    def __post_init__(self):
        # Coerce response text into Element Tree
        self.response_tree = ET.fromstring(self.response.text)
        self.count = int(self.response_tree.findtext("Count"))
        self.web_env = self.response_tree.findtext("WebEnv")
        self.query_key = self.response_tree.findtext("QueryKey")
        self.pmids = set(["PMC"+i.text for i in self.response_tree.find("IdList")])


def search_articles(keyword: str,
                    min_dte: str,
                    max_dte: Optional[str] = None,
                    ret_max: int = 5000) -> {str}:
    """Search to retrieve docs"""
    max_dte = max_dte or (datetime.today() - datetime.timedelta(days=1)).strftime('%Y/%m/%d')
    date_range = f"mindate={min_dte}&maxdate={max_dte}"
    search_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&usehistory=y"
    search_page_url = f'{search_base_url}&term="{keyword}"&{date_range}&retmax={ret_max}' + '&retstart={i_pmid}'

    response = PaperSearchResponse(requests.get(search_page_url.format(i_pmid=0)))
    pmids = response.pmids
    
    for i_pmid in range(ret_max, response.count, ret_max):
        sleep(0.5)  # Limit 2 api calls/sec
        response = PaperSearchResponse(requests.get(search_page_url.format(i_pmid=i_pmid)))
        pmids |= response.pmids

    return pmids


def get_search_hist_args(keywords: Union[str, List[str]],
                         search_hist: UC_Table,
                         min_dte: str = "2022/01/01",
                         max_dte: Optional[str] = None) -> List[dict]:
    # Get a date range that will avoid creating gaps in search range from previous runs
    keywords: [str] = keywords if isinstance(keywords, list) else [str(keywords),]
    max_dte = max_dte or (datetime.today() - datetime.timedelta(days=1)).strftime('%Y/%m/%d')
    search_args_df = search_hist.spark.createDataFrame(data=[(kw.lower(), min_dte, max_dte) for kw in keywords],
                                                       schema="keyword STRING, min_dte STRING, max_dte STRING")
    args = search_args_df.alias('a').join(search_hist.df.alias('h'), "keyword", "outer") \
                         .withColumn('h_min', F.coalesce(F.col('h.min_dte'), F.col('a.min_dte'))) \
                         .withColumn('h_max', F.coalesce(F.col('h.max_dte'), F.col('a.max_dte'))) \
                         .select(F.col('keyword').alias('keyword'),
                                 F.when(F.col('a.min_dte') < F.col('h_min'),
                                        F.col('a.min_dte')).otherwise(F.col('h_min')).alias('min_dte'),
                                 F.when(F.col('a.max_dte') > F.col('h_max'),
                                        F.col('a.max_dte')).otherwise(F.col('h_max')).alias('max_dte')) \
                         .filter(F.col('min_dte') <= F.col('max_dte')).collect()
    return [r.asDict() for r in args]


@udf
def download_articles(accession_id: str,
                      articles_path: str,
                      file_type: str = "xml"):
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    s3_conn = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    s3_conn.download_file("pmc-oa-opendata",
                          f'oa_comm/{file_type}/all/{accession_id}.{file_type}',
                          f'{articles_path}/{accession_id}.{file_type}')
    return "DOWNLOADED"


@udf(returnType=StructType([
    StructField("attrs", MapType(StringType(), StringType())), 
    StructField("front", StringType()),
    StructField("body", StringType()),
    StructField("floats_group", StringType()),
    StructField("back", StringType()),
    StructField("processing_metadata", StringType())]))
def curate_xml_dict(xmlPath: str):
    try:
        with open(xmlPath, 'r') as file:
            file_content = file.read()
        soup = BeautifulSoup(file_content, 'xml')
        article_detail_map = {'dtd-version': 'dtd_version',
                              'xml:lang': 'xml_lang',
                              'article-type': 'article_type',
                              'xmlns:mml': 'xmlns_mml',
                              'xmlns:xlink': 'xmlns_xlink'}
        article_dict = {'attrs': {article_detail_map[k]: str(v) for k, v in soup.find('article').attrs.items()
                                  if k in article_detail_map.keys()},
                        'front':               str(soup.find('front')),
                        'body':                str(soup.find('body')),
                        'floats_group':        str(soup.find('floats-group')),
                        'back':                str(soup.find('back')),
                        'processing_metadata': str(soup.find('processing-meta'))}
        return article_dict
    except:
        # TODO: Populate with Exception Handling
        return None


def download_pending_articles(search_hist: UC_Table,
                              metadata: UC_Table,
                              articles: UC_Volume,
                              keywords: Union[str, List[str]] = None,
                              min_dte: str = "2022/01/01",
                              max_dte: Optional[str] = None):
    # This will return a list of needed pmids based upon what has already been searched and pulled as well as what
    # is still pending download. If keywords is left blank, range will be applied to all keywords already in search hist
    if keywords:
        keywords: [str] = keywords if isinstance(keywords, list) else [str(keywords),]
    else:
        keywords: [str] = [r.keyword for r in search_hist.df.select(F.col('keyword')).distinct().collect()]
        if len(keywords) == 0:
            raise ValueError("When raw_search_hist is empty, you must provide a value for keywords which will " +
                             "instantiate the raw_search_hist table.")

    max_dte = max_dte or datetime.today().strftime('%Y/%m/%d')
    kwargs_list = get_search_hist_args(keywords, search_hist, min_dte, max_dte)
    pmids = search_articles(**kwargs_list[0])
    for kwargs in kwargs_list[1:]:
        sleep(0.5)  # limit 2 api calls/sec
        pmids |= search_articles(**kwargs)

    print(f"Search identified {len(pmids)} articles.")

    pmids_df = metadata.df.sparkSession.createDataFrame([(i,) for i in pmids], "AccessionId STRING")
    file_type = metadata.name.split('_')[-1]
    articles_path = articles.path

    # NOTE: download_articles udf actually downloads the articles in addition to returning DOWNLOAD or ERROR status
    metadata_src = metadata.df.filter(F.col('Status') == F.lit('PENDING')) \
                           .join(pmids_df, "AccessionId", "leftsemi") \
                           .repartition(64) \
                           .withColumn("Status",
                                       F.when(F.col('Retracted') == F.lit('yes'), F.lit("RETRACTED"))
                                       .otherwise(download_articles(F.col("AccessionId"),
                                                                    F.lit(articles_path),
                                                                    F.lit(file_type)))) \
                           .withColumn("volume_path", F.when(F.col('Status') == F.lit("DOWNLOADED"),
                                                             F.concat(F.lit(articles_path), F.lit('/'),
                                                                      F.col("AccessionId"), F.lit('.'),
                                                                      F.lit(file_type)))).cache()

    # Update metadata table
    #metadata.dt.alias("tgt").merge(source=metadata_src.alias("src"),
    #                               condition="src.AccessionID = tgt.AccessionID") \
    #    .whenMatchedUpdateAll() \
    #    .execute()

    # Update search_hist
    #kwargs_df = pd.DataFrame(kwargs_list)
    #search_hist.dt.alias("tgt").merge(source=search_hist.spark.createDataFrame(kwargs_df).alias("src"),
    #                                  condition="src.keyword = tgt.keyword") \
    #    .whenMatchedUpdateAll() \
    #    .whenNotMatchedInsertAll() \
    #    .execute()

    # TODO: Removal of RETRACTED Files
    # metadata_src will likely be inspected after return, with unpersist an undesired re-download would happen
    # Therefor we will redefine with a definition that does not call download_articles
    #metadata_src.unpersist()
    #return metadata.df.join(pmids_df, "AccessionId", "leftsemi")
    return pmids_df, metadata_src


# COMMAND ----------

display(pubmed_wf.raw_metadata_xml.df)

# COMMAND ----------

from datetime import datetime, timedelta

pmids_df, metadata_src = download_pending_articles(search_hist=pubmed_wf.raw_search_hist, 
                                                metadata=pubmed_wf.raw_metadata_xml, 
                                                articles=pubmed_wf.raw_articles_xml, 
                                                keywords="breast cancer", 
                                                min_dte=(datetime.now() - timedelta(days=5)).strftime('%Y/%m/%d'),
                                                max_dte=(datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d'))

# COMMAND ----------

display(pubmed_wf.raw_metadata_xml.df.filter("volume_path IS NOT NULL"))

# COMMAND ----------

article_files[0].path[5:]

# COMMAND ----------

# We need to add this to synchronize metadata table with actual downloads

SYNC_METADATA_DOWNLOADS = True

if SYNC_METADATA_DOWNLOADS:
    from pyspark.sql.functions import col, lit
    article_files = dbutils.fs.ls(pubmed_wf.raw_articles_xml.path)
    pmid_df = spark.createDataFrame([(a.name.split('.')[0],
                                      a.path[5:]) for a in article_files],
                                    ["AccessionID", "volume_path"])
    pubmed_wf.raw_metadata_xml.dt.alias("tgt").merge(source=metadata_src.alias("src"),
                                                     condition="src.AccessionID = tgt.AccessionID") \
                                              .whenMatchedUpdate(
                                                  set={"Status": lit("DOWNLOADED"),
                                                       "volume_path": col("src.volume_path")})

# COMMAND ----------

display(pubmed_wf.raw_metadata_xml.df)

# COMMAND ----------

display(pmid_df)

# COMMAND ----------

# from biomed_genai.workflow.pubmed_wf.pmc import download_pending_articles
from datetime import datetime, timedelta

DOWNLOAD_WITH_KEYWORD = True
DOWNLOAD_WITHOUT_KEYWORD = False

if DOWNLOAD_WITH_KEYWORD:
    articles_status = download_pending_articles(search_hist=pubmed_wf.raw_search_hist, 
                                                metadata=pubmed_wf.raw_metadata_xml, 
                                                articles=pubmed_wf.raw_articles_xml, 
                                                keywords="breast cancer", 
                                                min_dte=(datetime.now() - timedelta(days=5)).strftime('%Y/%m/%d'),
                                                max_dte=(datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d'))

if DOWNLOAD_WITHOUT_KEYWORD:
    articles_status = download_pending_articles(search_hist=pubmed_wf.raw_search_hist, 
                                                metadata=pubmed_wf.raw_metadata_xml, 
                                                articles=pubmed_wf.raw_articles_xml,
                                                min_dte=(datetime.now() - timedelta(days=5)).strftime('%Y/%m/%d'))

# display(articles_status.limit(5))

# COMMAND ----------

display(articles_status)

# COMMAND ----------

pmids_df = articles_status
metadata = pubmed_wf.raw_metadata_xml

metadata_src = metadata.df.filter(F.col('Status') == F.lit('PENDING')) \
                       .join(pmids_df, "AccessionId", "leftsemi")

# COMMAND ----------

display(metadata_src)

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

INSPECT_METADATA_XML_CHANGES = True

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

INSPECT_SEARCH_HIST = True

if INSPECT_SEARCH_HIST:
    display(pubmed_wf.raw_search_hist.df)

# COMMAND ----------


from databricks.model_training import foundation_model

foundation_model.get_models()

# COMMAND ----------

# Optional, this forces a sync 

SYNC_METADATA_DOWNLOADS = False

if SYNC_METADATA_DOWNLOADS:
    from pyspark.sql.functions import col, lit
    article_files = dbutils.fs.ls(pubmed_wf.raw_articles_xml.path)
    pmid_df = spark.createDataFrame([(a.name.split('.')[0],
                                      a.path[5:]) for a in article_files],
                                    ["AccessionID", "volume_path"])
    pubmed_wf.raw_metadata_xml.dt.alias("tgt").merge(source=metadata_src.alias("src"),
                                                     condition="src.AccessionID = tgt.AccessionID") \
                                              .whenMatchedUpdate(
                                                  set={"Status": lit("DOWNLOADED"),
                                                       "volume_path": col("src.volume_path")})
