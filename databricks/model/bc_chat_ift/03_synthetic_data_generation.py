# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook is third in a series that generates synthetic data for Instruction Fine Tuning (IFT).
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. From the seed.jsonl generated in the previous NB, perform data augmentation to generate more synthetic data.
# MAGIC 2. Merge seed.jsonl with the synthetic data. Together they will be used for training and evaluating IFT in subsequent NBs

# COMMAND ----------

# MAGIC %pip install langchain_databricks langchain_openai langchain_experimental
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re, os, json
import pyspark.pandas as ps
from pyspark.sql.functions import udf, col, flatten, desc, rank, rand, length
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.window import Window
from langchain_databricks import ChatDatabricks
from langchain_openai import ChatOpenAI
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import datasets
from typing import Optional
from pprint import pprint
from random import sample
import pandas as pd
from langchain_experimental.tabular_synthetic_data.openai import OPENAI_TEMPLATE, create_openai_data_generator
from typing_extensions import Annotated, TypedDict

# COMMAND ----------

# DBTITLE 1,load parameters and keys. Must only have %run and nothing else
# MAGIC %run ./_setup/params

# COMMAND ----------

# https://python.langchain.com/docs/integrations/chat/databricks/
class QA_augmented(TypedDict):
#    context: Annotated[str, ..., "Chunks of articles most similar to a topic queried from Vector Store"]
#    question: Annotated[str, ..., "Question provided"]
#    answer: Annotated[str, ..., "Answer to the provided question"]
#    prompt_variant: Annotated[str, ..., "Prompt to re-write question"]
    question_new: Annotated[str, ..., "New re-written question"]
    answer_new: Annotated[str, ..., "New answer to the re-written question"]

# COMMAND ----------

selected_fields = ["context", "question", "answer", "prompt_variant"]

# COMMAND ----------

seed_df = spark.createDataFrame(pd.read_json("data/seed.jsonl", lines=True))
display(seed_df)

# COMMAND ----------

prompt_variants = {
    "depth": "Re-write the question to ask in greater depth. The answer should correspondingly explain in greater detail or demonstrate step-by-step reasoning.",
    "complexity": "Re-write the question to be more complex such that the corresponding answers require the use of precise and specific technical terms and defining them as needed.",
    "conditional": "Re-write the question such that its answer is not general and appropriate states exceptions and conditions to demonstrate a deep and nuanced understanding of the question.",
    "diversity": "Draw inspiration from the provided context, question and answer to create a diverse range of questions and answers that are still in the same domain and based on the same context.",
    "paraphrase": "You can paraphrase the question and answer."}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross the seeds with prompt variants such that every seed is paired with every prompt variant

# COMMAND ----------

prompts_df = spark.createDataFrame(
    pd.DataFrame.from_dict(prompt_variants, orient="index", 
                           columns=['prompt_variant']).reset_index())
display(prompts_df)

# COMMAND ----------

seed_promptvariant = seed_df.crossJoin(prompts_df)
display(seed_promptvariant)

# COMMAND ----------

seed_dict = seed_promptvariant.select(*selected_fields).toPandas().to_dict("records")
seed_dict[0:10]

# COMMAND ----------

prefix = """You are an experienced linguistics expert for building datasets for large language models. Rewrite and paraphrase the question and answer provided following these rules:
1. The re-written question and answer must still be grounded in the provided context without extra information added.
2. {prompt_variant}

Examples are provided below delimited by ### to show how new questions and answers are re-written.
Examples:
###"""

suffix = """###
Return the re-written question and answer in the fields 'question_new' and 'answer_new' respectively.

Context:
{context}
Question:
{question}
Answer:
{answer}
"""

example_prompt = PromptTemplate.from_template("""Context: {context}
Question: {question}
Answer: {answer}
Question_new: {question_new}
Answer_new: {answer_new}""")

# COMMAND ----------

examples = [{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs",
"prompt_variant": "Re-write the question to ask in greater depth. The answer should correspondingly explain in greater detail or demonstrate step-by-step reasoning.",
"question_new": "Explain the advantages of the new ESTRO-ACROP guideline",
"answer_new":"Because ESTRO-ACROP uses modern volume-based planning techniques, it has dosimentric advantages, minimizing radiation exposure to cardiopulmonary organs, such as the heart, left ventricle, and LAD known to be associated long-term rates of high-grade coronary artery stenosis and acute coronary events. Minimizing such long term harm is important as many breast cancer patients are expected to have long-term survival."}]

# COMMAND ----------

llm = ChatDatabricks(endpoint = model, 
                  temperature=TEMPERATURE)
#llm = ChatOpenAI(model = "gpt-4o", temﬂperature=TEMPERATURE)
prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        input_variables=selected_fields,
        prefix=prefix,
        suffix=suffix)
structured_llm = llm.with_structured_output(QA_augmented)
chain = prompt | structured_llm

# COMMAND ----------

# MAGIC %md
# MAGIC Check the full prompt

# COMMAND ----------

print(prompt.format_prompt(prompt_variant='<prompt_variant>',
                           context='<context>', question='<question>', answer='<answer>').text)

# COMMAND ----------

seed_dict[8]

# COMMAND ----------

r = chain.invoke(seed_dict[8])
r

# COMMAND ----------

'What was the hypothesis regarding the role of Drosha- and Nfib mRNA-interacting proteins?',
'The hypothesis was that common Drosha- and Nfib mRNA-interacting proteins modulate Drosha activity toward the Nfib mRNA.'

# COMMAND ----------

responses = chain.batch(seed_dict[0:10],
                        config={"max_concurrency": 3})
responses

# COMMAND ----------


