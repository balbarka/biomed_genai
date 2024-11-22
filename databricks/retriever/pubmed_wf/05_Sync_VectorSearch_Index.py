# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Sync VectorSearch Index
# MAGIC
# MAGIC Databricks offers platform functionality across the entire GenAI development lifecycle including deployment. Now, that we have the content that we want to host for augmenting our eventual models, we'll need to serve that content using [Mosaic AI VectorSearch](https://docs.databricks.com/en/generative-ai/vector-search.html#mosaic-ai-vector-search). That is dependent upon two workspace entities that we'll want to maintain:
# MAGIC  - [EndPoint](https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.client.VectorSearchClient.get_endpoint): *dict* - Currently, there isn't an SDK class that comes with endpoint and that's ok becuase you'll typically create the endpoint once, assign permissions and not need to reference again in our genai apps. Thus, when you inspect this entity, you be working with a dict that contains the state and attributes of the endpoint.
# MAGIC  - [VectorSearchIndex](https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex): *databricks.vector_search.index.VectorSearchIndex* - This is a proper class with class methods that we'll use in our genai applications. There are three methods that we'll want to familiarize ourselves with:
# MAGIC     * [create_delta_sync_index](https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.client.VectorSearchClient.create_delta_sync_index) - Creates a delta sync index (option 1). We've include the create task as part of `biomed` config. This method will only be executed once, but it can take a cofee-walk time to build (~15 minutes). This method requires the following arguments:
# MAGIC         * **endpoint_name** - The name of the endpoint, can be found as `biomed.vector_search.biomed.endpoint['name']`.
# MAGIC         * **index_name** - The UC name that we assign to our index. Be aware that while this is a UC name, it is a workspace entity due to dependence on workspace compute.
# MAGIC         * **primary_key** - Single column name for the primary key of our delta table.
# MAGIC         * **source_table_name** - This is the UC name of the source delta table.
# MAGIC         * **pipeline_type** – The type of the pipeline. Must be `CONTINUOUS` or `TRIGGERED`. We are choosing `TRIGGERED` so a sync will not happened unless called. If `CONTINUOUS` is chosen a long lived job will make updates as the delta table is updated. This does require an additional long lived service that will not be explorered in this solution accelerator.
# MAGIC     * [sync](https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.client.VectorSearchClient.create_delta_sync_index) - Sync the index. This is used to sync the index with the source delta table. This only works with managed delta sync index when *pipeline type* is `TRIGGERED`.
# MAGIC     * [similarity_search](https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search) - Kinda the whole point of VectorSearch. This is the method we will call within our GenAI app to retrieve the most releveant documents.
# MAGIC
# MAGIC **NOTE**: Unlike all other `biomed` class entities so far, `vector_search` entities are workspace entities that happen to have UC refrences. Thus, be aware that there are workspace dependencies when working with vector_search. This matches our intuition, because ultimately serving is a compute component and therefor must belong within a workspace that includes other computer components like cluster compute and sql warehouses.
# MAGIC
# MAGIC **NOTE**: There are three different vector embedding management options for our vector search index. They differ on how much of embedding tasks is exposed for use and configuration. The [options](https://docs.databricks.com/en/generative-ai/vector-search.html#options-for-providing-vector-embeddings) are below in order from most managed to most configurable. This solutions accelerator only employs most managed, first, option below:
# MAGIC  - **Option 1, Managed Embeddings** (Used in this workflow) - you provide a text column and endpoint name and Databricks synchronizes the index with your Delta table.
# MAGIC  - **Option 2, Self Managed Embedding** - you compute the embeddings and save them as a field of your Delta Table, Databricks will then synchronize the index
# MAGIC  - **Option 3, Direct Index** - when you want to use and update the index without having a Delta Table

# COMMAND ----------

# DBTITLE 1,Initialize pubmed_wf Application Class
# MAGIC %run ./_setup/setup_pubmed_wf $SHOW_TABLE=false $SHOW_WORKFLOW=true

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Inspect Endpoint
# MAGIC
# MAGIC There isn't much that goes into Endpoint, but we still need to be aware of it's presence since scaling to many indexes will have implications of which indexes are grouped into the same endpoint (which won't be covered in this solution accelerator).

# COMMAND ----------

pubmed_wf.vector_search.biomed.endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Sync Index
# MAGIC
# MAGIC Sync Index triggers a DLT job. This will kick off a sync job and not await completion. Your index can still serve similarty search from current state while the index is updating.
# MAGIC
# MAGIC **TODO**: Add details about the managed DLT job and arch benefits.

# COMMAND ----------

pubmed_wf.vector_search.biomed.processed_articles_content_vs_index.index.sync()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## [OPTIONAL] Inspect VectorSearch Index via `similarity_search`
# MAGIC
# MAGIC [similarity_search]() returns a dict that can be coersed into a dataclass for ease of inspection. For now, won't don't need to manipulate our search results. Rather, just run a simple query to make sure that our service is up and running.

# COMMAND ----------

INSPECT_VS_INDEX = False

if INSPECT_VS_INDEX:
    query_text = "What are some proteins associated with breast cancer and what methods are available to detect these?"
    vs_index = pubmed_wf.vector_search.biomed.processed_articles_content_vs_index.index
    rslt = vs_index.similarity_search(query_text=query_text,
                                      columns=["id","content"],
                                      num_results=5)
    display(rslt)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## [OPTIONAL] Inspect VectorSearch Index as a LangChain Class
# MAGIC
# MAGIC While Databricks SDK is sufficient for writing RAG and Compound GenAI applications, many developers prefer to use genai chain libraries like [LangChain](https://python.langchain.com/v0.2/docs/introduction/#api-reference). LangChain is a framework that allows developers to focus on the genai app component design by working with component classes that handle the lower level details. This allows developers to write more succint code for application logic once classes have been instantiated. 
# MAGIC
# MAGIC In the LangChain framework components that retrieve data for context are called [retrievers](https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.retrievers). In this section we are going to test that we can instantiate and invoke a LangChain Retriever class for our Vector Search Index `processed_articles_content_vs_index`.
# MAGIC
# MAGIC **NOTE**: The **invoke** method returns a type of \[[Documents](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document)\]. Documents is a class specific to LangChain, that is the expected format for other chain components. By LangChain using these well defined method signatures, it simplifies development once components are learned.

# COMMAND ----------

INSPECT_LC_INDEX = False

if INSPECT_LC_INDEX:
    from langchain_community.vectorstores import DatabricksVectorSearch
    query_text = "What are some proteins associated with breast cancer and what methods are available to detect these?"

    retriever_config = {"chunk_template": "Passage: {chunk_text}\n",
                        "data_pipeline_tag": "biomed_workflow",
                        "parameters": {"k": 5, "query_type": "ann"},
                        "schema": {"chunk_text": "content",
                                   "document_uri": "url",
                                   "primary_key": "id"},
                        "vector_search_index": pubmed_wf.vector_search.biomed.processed_articles_content_vs_index.index.name}
    # Turn the Vector Search index into a LangChain retriever
    lc_retriever = DatabricksVectorSearch(
        index = pubmed_wf.vector_search.biomed.processed_articles_content_vs_index.index,
        text_column="content",
        columns=["id","content"]).as_retriever(**retriever_config)
    rslt = lc_retriever.invoke(query_text)
    print(rslt)

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO**: Look at https://docs.llamaindex.ai/en/stable/examples/vector_stores/DatabricksVectorSearchDemo/
