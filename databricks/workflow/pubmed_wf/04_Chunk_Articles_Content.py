# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Chunk Articles with `unstructured`
# MAGIC
# MAGIC We will use [unstructured](https://unstructured.io/) for our primary chunking library. We are going to use this for the actual body content and it is common to change the arguments of the unstructured [partitioning](https://docs.unstructured.io/open-source/core-functionality/partitioning) functions upon future iterations where we are improving our Dataset curation for pre-training or fine-tuning or our chunking strategy for our VS index.
# MAGIC
# MAGIC **NOTE**: Since we are working with XML data we are going to use the [partition-xml](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-xml) function. There are many libraries out there that can make use of the xml tags we left in our body column and they can excluded easily with regex or opensource xml parsing library. Thus, we left the xml in the body to allow for discovery of new / different parsing strategies in the future.
# MAGIC
# MAGIC **NOTE**: YES. We could have used [partition-xml](https://docs.unstructured.io/open-source/core-functionality/partitioning#partition-xml) function to parse from file instead of from the `curated_articles` delta table. Similar to the above note, we did this to make future iterative improvements faster as reading text from file in blob storage has a much larger I/O preformance cost. This was a deliberate architecture decision for future enhancements, not just to conform to a [Medallion Architecture](https://www.databricks.com/glossary/medallion-architecture)... although we are doing that as well.

# COMMAND ----------

# DBTITLE 1,Initialize pubmed_wf Application Class
# MAGIC %run ./_setup/setup_pubmed_wf $SHOW_TABLE=false $SHOW_WORKFLOW=true

# COMMAND ----------

# curated_articles_content will include all metadata fields we'll want in vectorsearches
# TODO: Put constraint so that id is a primary key

sql_path = pubmed_wf.processed_articles_content.sql_path
with open(sql_path, 'r') as file:
    sql = file.read()
    print(sql)

# COMMAND ----------

# Create a UDF that will chunk our article bodies
from unstructured.partition.xml import partition_xml
from pyspark.sql.types import ArrayType, StringType
import xml.etree.ElementTree as ET
import html2text

def chunk_xml_body(body: str, attrs: dict):
    text_maker = html2text.HTML2Text()
    root = ET.Element('root', attrib=attrs)
    root.text = body
    body_elements = partition_xml(text=str(ET.tostring(root, encoding='utf-8'), 'UTF-8'),
                                  xml_keep_tags = False,
                                  encoding='utf-8',
                                  include_metadata=False,
                                  languages=['eng',],
                                  date_from_file_object=None,
                                  chunking_strategy='by_title',
                                  multipage_sections=True,
                                  combine_text_under_n_chars=300,
                                  new_after_n_chars=1400,
                                  max_characters=1250)
    body_chunks = [text_maker.handle(str(be.text)) for be in body_elements if len(be.text) >= 110]
    return body_chunks

chunk_xml_body_udf = udf(chunk_xml_body, ArrayType(StringType()))

# COMMAND ----------

INSPECT_CURATED_ARTICLES = True

if INSPECT_CURATED_ARTICLES:
    display(pubmed_wf.curated_articles_xml.df)

# COMMAND ----------

# Get dataframe of new articles
from pyspark.sql.functions import col, lit, concat
from pyspark.sql.functions import xpath_string, explode, posexplode

# Insert all previously unprocessed articles
pubmed_wf.curated_articles_xml.df.alias("a") \
      .join(pubmed_wf.processed_articles_content.df.select(col("pmid")).distinct().alias("b"),
            col("a.AccessionID") == col("b.pmid"), "left_anti") \
      .withColumn('contents', chunk_xml_body_udf('body', 'attrs')) \
      .select(col('AccessionID').alias('pmid'),
                  xpath_string(col('front'),lit('front/article-meta/title-group/article-title')).alias('title'),
                  xpath_string(col('front'),lit('front/journal-meta/journal-title-group/journal-title')).alias('journal'),
                  lit('NEED DESIRED CITATION FORMAT').alias('citation'),
                  xpath_string(col('front'),lit('front/article-meta/pub-date/year')).alias('year'),
                  posexplode('contents').alias('content_pos', 'content')) \
      .withColumn('id', concat(col('pmid'), lit('-'), col('content_pos'))) \
      .drop('content_pos') \
      .write.mode('append').saveAsTable(pubmed_wf.processed_articles_content.name)
