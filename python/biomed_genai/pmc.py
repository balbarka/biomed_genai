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

from .config import UC_Table, UC_Volume


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


def search_papers(keyword: str,
                  min_dte: str = "2022/01/01",
                  max_dte: Optional[str] = None,
                  ret_max: int = 5000) -> {str}:
    """Search to retrieve docs"""
    date_range = f"mindate={min_dte}&maxdate={max_dte or datetime.today().strftime('%Y/%m/%d')}"
    search_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&usehistory=y"
    search_page_url = f"{search_base_url}&term={keyword}&{date_range}&retmax={ret_max}" + "&retstart={i_pmid}"

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
    max_dte = max_dte or datetime.today().strftime('%Y/%m/%d')
    search_args_df = search_hist.spark.createDataFrame(data=[(kw.lower(), min_dte, max_dte) for kw in keywords],
                                                       schema="keyword STRING, min_dte STRING, max_dte STRING")
    args = search_args_df.alias('a').join(search_hist.df.alias('h'), "keyword", "outer") \
                         .withColumn('h_min', F.coalesce(F.col('h.min_dte'), F.col('a.min_dte'))) \
                         .withColumn('h_max', F.coalesce(F.col('h.max_dte'), F.col('a.max_dte'))) \
                         .select(F.col('keyword').alias('keyword'),
                                 F.when(F.col('a.min_dte < h_min'),
                                        F.col('a.min_dte')).otherwise(F.col('h_min')).alias('min_dte'),
                                 F.when(F.col('a.max_dte > h_max'),
                                        F.col('a.max_dte')).otherwise(F.col('h_max')).alias('max_dte')) \
                         .filter(F.col('min_dte <= max_dte')).collect()
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


def get_needed_pmids_df(search_hist: UC_Table,
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
    pmids = search_papers(**kwargs_list[0])
    for kwargs in kwargs_list[1:]:
        sleep(0.5)  # limit 2 api calls/sec
        pmids |= search_papers(**kwargs)

    print(f"Retrieved {len(pmids)} articles from api")

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
    metadata.dt.alias("tgt").merge(source=metadata_src.alias("src"),
                                   condition="src.AccessionID = tgt.AccessionID") \
        .whenMatchedUpdateAll() \
        .execute()

    # Update search_hist
    kwargs_df = pd.DataFrame(kwargs_list)
    search_hist.dt.alias("tgt").merge(source=search_hist.spark.createDataFrame(kwargs_df).alias("src"),
                                      condition="src.keyword = tgt.keyword") \
        .whenMatchedUpdateAll() \
        .whenNotMatchedInsertAll() \
        .execute()

    # TODO: Removal of RETRACTED Files
    # metadata_src will likely be inspected after return, with unpersist an undesired re-download would happen
    # Therefor we will redefine with a definition that does not call download_articles
    metadata_src.unpersist()
    return metadata.df.join(pmids_df, "AccessionId", "leftsemi")
