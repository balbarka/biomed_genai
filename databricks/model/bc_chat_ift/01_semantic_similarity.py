# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is first in a series that **generates synthetic data** for Instruction Fine Tuning (IFT).
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. Generate a set of breast cancer topics.
# MAGIC 2. For each topic, query from the Vector Store of breast cancer articles a set of chunks most semantically similar to the topic.
# MAGIC 3. Reformat the chunk set for subsequent Few Shot Prompting (FSP) in notebook `2_FSP`

# COMMAND ----------

import pandas as pd
import re, os, json
from typing import List

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query similar chunks as context for generating Q&A
# MAGIC Set up vector search index so we can query chunks relevant to a topic of interest

# COMMAND ----------

# MAGIC %run ../../workflow/pubmed_wf/_setup/setup_pubmed_wf

# COMMAND ----------

vs_index = pubmed_wf.vector_search.biomed.processed_articles_content_vs_index.index

# COMMAND ----------

topics = ["breast cancer biomarkers", 
          "breast cancer screening",
          "breast cancer diagnostic methods", 
          "breast cancer therapies",
          "breast cancer treatment recommendations",
          "breast cancer prognosis",
          "breast cancer risk factors",
          "breast cancer clinical trials",
          "emerging breast cancer therapies"]

# COMMAND ----------

def similar_chunks_from_topic_list(topics: List[str],
                                   vs=vs_index, 
                                   num_results: int = 5,
                                   min_score: float = 0.6,
                                   columns: List[str] = None,
                                   groupby: List[str] = ['topic']):
    # If columns are unspecified, auto-detect from vs
    if not columns:
        try:
            pk_column = vs.describe().get('primary_key')
            embedded_column = vs.describe().get('delta_sync_index_spec').get('embedding_source_columns')[0].get('name')
            columns=[pk_column, embedded_column]
        except:
            raise KeyError("vs is missing primary_key or embedding_source_columns. Either list in columns or re-index vs accordingly")
    
    chunkset = []
    for t in topics:
        # TODO: what if no results?
        similar_chunks = vs.similarity_search(query_text=t,
                        columns=columns,
                        num_results=num_results)
        # Pre-pend topic to each result
        similar_chunks_with_topic = [[t]+i for i in similar_chunks['result']['data_array']]
        chunkset += similar_chunks_with_topic

    # Reformat chunkset (list of lists) into pandas dataframe
    df_columns = ['topic'] + [i['name'] for i in similar_chunks['manifest']['columns']]
    df = pd.DataFrame(chunkset, columns=df_columns)

    # Filter on min_score
    df = df[df.score >= min_score]

    # Return (un)aggregated dataframe
    if groupby:
        return df.groupby(groupby).agg(list)
    else:
        return df

# COMMAND ----------

chunkset_df = similar_chunks_from_topic_list(topics, vs=vs_index, 
                                             num_results=10, min_score=0.62)
chunkset_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save as a table in UC

# COMMAND ----------

spark.createDataFrame(chunkset_df) \
    .write.format(‘delta’).saveAsTable(‘yen.syn_data_gen.chunksets’)
