# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC The creation and training for an instruction fine tuning example from breast cancer data

# COMMAND ----------

system_prompt_template = """
### Instruction
You are tasked to help a health care researcher determine the intention of a question. You are responsible for categorizing the following question into one of the following categories: 

"Diagnosis Inquiry"
"General Knowledge"
"Research"

Do not create or use any categories other than those explicitly listed above. Return only single category as response and nothing else. If there is confusion between multiple categories error on the side of assigning "Diagnosis Inquery" before "General Knowledge" and "General Knowledge" before "Research". 

###Example Input
I just found out I have high levels of protiens associated with breast cancer, what should I do?

###Response
Diagnosis Inquiry

###Example Input
What percentage of American Women will be diagnosed with breast cancer in the next 5 years?

###Response
General Knowledge

###Example Input
What are some current advancements in tomosynthesis that can be used to predict tumor growth rates?

###Response
Research

Based on the above, categorize the following question: \n\n"""

template = spark.createDataFrame([{"prompt_template": system_prompt_template}])

template.createOrReplaceTempView("template")

display(template)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC To create our dataset, we are going to randomly look through our own abstract topics and create 10 categorization examples on the topic. We'll first use pull 10 abstract segements to generate some topics:

# COMMAND ----------

# Get a sample of 10 abstracts
from pyspark.sql.functions import regexp_replace, xpath_string, col, lit, rand

abstracts = spark.table("biomed_genai.curated.articles_xml") \
                 .select(regexp_replace(
                         regexp_replace(
                             xpath_string(col('front'), lit('string(//abstract)')),
                             '<[^>]+>', ''),
                         '^Abstract\s*', '').alias('abstract')) \
                 .orderBy(rand()).limit(2)

display(abstracts)

abstracts.createOrReplaceTempView("abstracts")

# COMMAND ----------

diagnosis_prompt = """Read the following abstract and create a question in the first person related to the abstract. The question will be proposed as if it relates to the question asker who has just recieved a similar diagnosis. Propose the question in such a way that the it is clear the asker has recieved a related diagnosis and that the question is only 1-2 sentences long. Do not proceed the question with anything, especially the word Question. Abstract:"""

general_prompt = """Read the following abstract and create a question related to the abstract. Make sure that the question is 1-2 sentences long and uses terminology that is consistent with someone seeking general knowledge on a topic, but doesn't have any formal education on the abstract topic. The question should be written as if it were being asked by an 8th grader. Do not proceed the question with anything, especially the word Question. Abstract:"""

research_prompt = """Read the following abstract and create a question that can be answered with the following context provided. Make sure that the question is 1-2 sentences long and uses terminology consistant with a researcher who works in the field of the context. Do not proceed the question with anything, especially the word Question. The question should be written at the college level of reading. Abstract: """

prompts = spark.createDataFrame([{"catagory": "Diagnosis Inquiry",
                                  "prompt": diagnosis_prompt},
                                 {"catagory": "General Knowledge",
                                  "prompt": general_prompt},
                                 {"catagory": "Research",
                                  "prompt": research_prompt}])

prompts.createOrReplaceTempView("prompts")

display(prompts)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE biomed_genai.processed.ift_ds AS
# MAGIC SELECT
# MAGIC     ARRAY(STRUCT("user" AS role, 
# MAGIC                  CONCAT(prompt_template,
# MAGIC                         ai_query('databricks-llama-2-70b-chat',
# MAGIC                                  CONCAT(prompt, abstract))) AS content),
# MAGIC           STRUCT('assistant' AS role,
# MAGIC                  catagory AS content)) AS messages
# MAGIC FROM 
# MAGIC     prompts
# MAGIC CROSS JOIN
# MAGIC     abstracts
# MAGIC CROSS JOIN
# MAGIC     template;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM biomed_genai.processed.ift_ds;

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CREATE TABLE biomed_genai.processed.ift_ds;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # TRAINING
# MAGIC
# MAGIC This and everything below is training and will need to move into the training section of the notebooks.

# COMMAND ----------

# Let's start by installing our libraries
%pip install --quiet databricks-genai==1.0.8 mlflow==2.14.2
%pip install --quiet databricks-sdk==0.29.0
dbutils.library.restartPython()

# COMMAND ----------

def get_current_cluster_id():
  import json
  return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']

get_current_cluster_id()

# COMMAND ----------

from databricks.model_training import foundation_model as fm
import mlflow

mlflow.set_registry_uri("databricks-uc")

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

#Let's clean the model name
registered_model_name = "biomed_genai.models.biomed_qcat"
    
run = fm.create(
    data_prep_cluster_id=get_current_cluster_id(),  
    model=base_model_name,  
    train_data_path='biomed_genai.processed.ift_ds',
    task_type="CHAT_COMPLETION",  
    training_duration="5ep",  #opnly 5 epochs for the demo
    register_to=registered_model_name,
    learning_rate="5e-7",
)

print(run)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ticket_priority_training_dataset AS
# MAGIC SELECT 
# MAGIC     ARRAY(
# MAGIC         STRUCT('user' AS role, CONCAT('{system_prompt}', '\n', description) AS content),
# MAGIC         STRUCT('assistant' AS role, priority AS content)
# MAGIC     ) AS messages
# MAGIC FROM customer_tickets;
# MAGIC """)
# MAGIC
# MAGIC spark.table('ticket_priority_training_dataset').display()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC

# COMMAND ----------

# Now we need to write three separate intentions for each of the abstracts

from pyspark.sql.functions import ai_query, concat, lit

# Assuming 'abstracts' is the DataFrame from Cell 6
summary_df = abstracts.withColumn(
    "summary",
    ai_query(
        "databricks-llama-2-70b-chat",
        concat(lit("Summarize the following abstract: "), abstracts.abstract)
    )
)


display(summary_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT '${research_prompt}';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ai_query('databricks-llama-2-70b-chat',
# MAGIC                 'name 3 car models')

# COMMAND ----------


