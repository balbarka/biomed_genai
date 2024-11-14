# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook is second in a series that **generates synthetic data by data augmentation of seed data** for subsequent chat completion Fine Tuning (FT).
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. Read in the seed data generated in the previous NB
# MAGIC 2. Create prompt variants by [Evolve-Instruct](https://arxiv.org/abs/2304.12244). Each prompt variant creates a variant of the seed datum
# MAGIC by increasing in depth, complexity etc.
# MAGIC 3. Set up appropriate prompt templates with the prompt variants, seed data and examples in langchain for Few Shot Prompting
# MAGIC 4. Run Few Shot Prompting to evolve the seed data according to the prompt variants

# COMMAND ----------

# MAGIC %pip install langchain_databricks>=0.1.1 langchain-experimental==0.3.3 langchain==0.3.7 langchain-community==0.3.5 langchain-core==0.3.15
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip freeze

# COMMAND ----------

import re, os, json
import pyspark.sql
from langchain_databricks import ChatDatabricks
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import pandas as pd
from typing import List, Type
from typing_extensions import Annotated, TypedDict
from pyspark.sql.functions import col
from pprint import pprint
from _setup.params import *
from _setup.utils import write_jsonl_by_line

# COMMAND ----------

seed_table_name: str = "yen.syn_data_gen.seed"
evolved_table_name: str = "yen.syn_data_gen.evolved"

# TODO: set up volume for these files
outfile: str = 'data/evolved.jsonl'

model_evolve: str = 'databricks-meta-llama-3.1-405b-instruct'
temperature: float = 0.7
max_retries: int = 2
max_concurrency: int = 4

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read in seed data

# COMMAND ----------

seed_df = spark.table(seed_table_name)
display(seed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up class object to collect structured output

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

# MAGIC %md
# MAGIC ## 2. Prompt Variants
# MAGIC Set up prompt variants inspired by [Evolve-Instruct](https://arxiv.org/abs/2304.12244) to evolve the question and answer with the same context

# COMMAND ----------

prompt_variants = {
    "depth": "Re-write the question to ask in greater depth. The answer should correspondingly explain in greater detail or demonstrate step-by-step reasoning.",
    "complexity": "Re-write the question to be more complex such that the corresponding answers require the use of precise and specific technical terms and defining them as needed.",
    "conditional": "Re-write the question such that its answer is not general and appropriately states exceptions and conditions to demonstrate a deep and nuanced understanding of the question.",
    "diversity": "Draw inspiration from the provided context, question and answer to create a diverse range of questions and answers that are still in the same domain and based on the same context.",
    "paraphrase": "You can paraphrase the question and answer."}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cross the seeds with prompt variants such that every seed is paired with every prompt variant

# COMMAND ----------

prompts_df = spark.createDataFrame(
    pd.DataFrame.from_dict(prompt_variants, orient="index", 
                           columns=['prompt']).reset_index())
prompts_df = prompts_df.withColumnRenamed('index','variant')
display(prompts_df)

# COMMAND ----------

seed_promptvariant = seed_df.crossJoin(prompts_df)
display(seed_promptvariant)

# COMMAND ----------

selected_fields = ["id", "context", "question", "answer", "prompt"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Set up prompt template and chain

# COMMAND ----------

prefix = """You are an experienced linguistics expert for building datasets for large language models. Rewrite and paraphrase the question and answer provided following these rules:
1. The re-written question and answer must still be grounded in the provided context without extra information added.
2. {prompt}

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

# MAGIC %md
# MAGIC #### Set up examples for Few Shot Prompting and have them evolve by the prompt variant

# COMMAND ----------

examples = dict()
examples['depth'] = [{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs",
"prompt_variant": "Re-write the question to ask in greater depth. The answer should correspondingly explain in greater detail or demonstrate step-by-step reasoning.",
"question_new": "Explain the advantages of the new ESTRO-ACROP guideline",
"answer_new": "Because ESTRO-ACROP uses modern volume-based planning techniques, it has dosimentric advantages, minimizing radiation exposure to cardiopulmonary organs, such as the heart, left ventricle, and LAD known to be associated long-term rates of high-grade coronary artery stenosis and acute coronary events. Minimizing such long term harm is important as many breast cancer patients are expected to have long-term survival."},
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "Why is early detection important for breast cancer prognosis?",
"answer": "Early detection is important for breast cancer prognosis because it significantly influences treatment options and outcomes. Studies indicate that a large percentage of breast cancer-related deaths are due to metastasis, making early detection crucial for effective intervention. Mammography, a common screening method, has been shown to reduce breast cancer mortality by approximately 20%. Additionally, identifying abnormal DNA methylation patterns at early stages can provide valuable diagnostic information, potentially leading to more timely and targeted treatments. This underscores the necessity for improved detection methods to enhance early diagnosis and, consequently, patient prognosis.",
"prompt_variant": "Re-write the question to ask in greater depth. The answer should correspondingly explain in greater detail or demonstrate step-by-step reasoning.",
"question_new": "What specific factors contribute to the effectiveness of early detection in improving breast cancer prognosis, and how do traditional and emerging diagnostic methods address these factors?",
"answer_new": "Early detection enhances breast cancer prognosis by identifying tumors before they metastasize, which accounts for 60% to 90% of deaths. Mammography reduces mortality by about 20%, yet it has limitations like false positives. Emerging technologies, such as DNA methylation analysis, detect cancer earlier by identifying abnormal patterns that precede tumor development. This method offers greater diagnostic precision compared to traditional protein markers, facilitating timely and tailored treatments. By integrating both approaches, healthcare can significantly improve early diagnosis and patient outcomes."}]

examples['complexity'] = [{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs",
"prompt_variant": "Re-write the question to be more complex such that the corresponding answers require the use of precise and specific technical terms and defining them as needed.",
"question_new": "Why is ESTRO-ACROP guideline more beneficial in terms of breast implant placement and dosimetry?",
"answer_new": "ESTRO-ACROP guidelines adjust radiation therapy according to breast implant placements and exclude implants from the CTV, minimizing implant-related complications. Such new target volume delineation method heart, left anterior descending coronary artery (LAD), and ipsilateral lung while maintaining target coverage, delivery accuracy, and dose heterogeneity. Minimizing exposure to protect critical cardiopulmonary organs is important as many breast cancer patients are expected to enjoy long-term survival."},
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "Why is early detection important for breast cancer prognosis?",
"answer": "Early detection is important for breast cancer prognosis because it significantly influences treatment options and outcomes. Studies indicate that a large percentage of breast cancer-related deaths are due to metastasis, making early detection crucial for effective intervention. Mammography, a common screening method, has been shown to reduce breast cancer mortality by approximately 20%. Additionally, identifying abnormal DNA methylation patterns at early stages can provide valuable diagnostic information, potentially leading to more timely and targeted treatments. This underscores the necessity for improved detection methods to enhance early diagnosis and, consequently, patient prognosis.",
"prompt_variant": "Re-write the question to be more complex such that the corresponding answers require the use of precise and specific technical terms and defining them as needed.",
"question_new": "In what ways do specific early detection techniques, such as mammography and DNA methylation profiling, mitigate the risk of metastasis in breast cancer, and how do these methodologies inform the selection of personalized therapeutic interventions?",
"answer_new": "Early detection techniques, including mammography and DNA methylation profiling, are vital in reducing breast cancer metastasis risk. Mammography enables the identification of tumors at an early stage, which is critical for timely intervention before metastasis occurs, thus potentially reducing mortality by about 20%. DNA methylation profiling detects epigenetic alterations that signal early tumorigenesis, providing insights into tumor biology and guiding targeted therapies. By integrating these detection methods, clinicians can tailor personalized treatment strategies that address individual tumor characteristics, ultimately enhancing patient outcomes and survival rates."}]

examples['conditional'] = [{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs",
"prompt_variant": 'Re-write the question such that its answer is not general and appropriately states exceptions and conditions to demonstrate a deep and nuanced understanding of the question.',
"question_new": "In which specific patient groups or clinical scenarios does the ESTRO-ACROP guideline offer distinct dosimetric advantages, and how do these benefits vary based on factors such as the method of implant placement (prepectoral vs. subpectoral)",
"answer_new": "Among patients with left-sided breast cancer, the the new target volume delineation method in ESTRO-ACROP guideline significantly reduces radiation exposure to critical structures, such as the heart and left anterior descending artery. This reduction in exposure is crucial for minimizing the risk of long-term cardiovascular complications."},
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "Why is early detection important for breast cancer prognosis?",
"answer": "Early detection is important for breast cancer prognosis because it significantly influences treatment options and outcomes. Studies indicate that a large percentage of breast cancer-related deaths are due to metastasis, making early detection crucial for effective intervention. Mammography, a common screening method, has been shown to reduce breast cancer mortality by approximately 20%. Additionally, identifying abnormal DNA methylation patterns at early stages can provide valuable diagnostic information, potentially leading to more timely and targeted treatments. This underscores the necessity for improved detection methods to enhance early diagnosis and, consequently, patient prognosis.",
"prompt_variant": 'Re-write the question such that its answer is not general and appropriately states exceptions and conditions to demonstrate a deep and nuanced understanding of the question.',
"question_new": "What specific factors influence the effectiveness of early detection methods in breast cancer prognosis, and what are the exceptions where these methods may not significantly improve outcomes?",
"answer_new": "Early detection significantly influences breast cancer prognosis, primarily through techniques like mammography and DNA methylation analysis. However, their effectiveness can be limited by factors such as high breast density, which reduces mammography sensitivity and increases false positives. Additionally, DNA methylation markers may not be reliable in all cancer subtypes or early stages, resulting in missed diagnoses. In cases of aggressive tumors, early detection may not substantially alter prognosis, highlighting the need for individualized screening approaches."}]

examples['diversity'] = [{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs",
"prompt_variant": 'Draw inspiration from the provided context, question and answer to create a diverse range of questions and answers that are still in the same domain and based on the same context.',
"question_new": "What implications do the findings of the ESTRO-ACROP guideline have for patients undergoing subpectoral versus prepectoral implant placements?",
"answer_new":"The ESTRO-ACROP guideline suggests that it is particularly feasible for patients undergoing subpectoral reconstruction, as the rates of breast complications in this group are comparable to those in previous studies. In contrast, the differences in complications between prepectoral and subpectoral approaches remain controversial and require further investigation."},
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "Why is early detection important for breast cancer prognosis?",
"answer": "Early detection is important for breast cancer prognosis because it significantly influences treatment options and outcomes. Studies indicate that a large percentage of breast cancer-related deaths are due to metastasis, making early detection crucial for effective intervention. Mammography, a common screening method, has been shown to reduce breast cancer mortality by approximately 20%. Additionally, identifying abnormal DNA methylation patterns at early stages can provide valuable diagnostic information, potentially leading to more timely and targeted treatments. This underscores the necessity for improved detection methods to enhance early diagnosis and, consequently, patient prognosis.",
"prompt_variant": 'Draw inspiration from the provided context, question and answer to create a diverse range of questions and answers that are still in the same domain and based on the same context.',
"question_new": "What are some advantages and disadvantages of using DNA methylation for breast cancer diagnosis?",
"answer_new": "DNA methylation for breast cancer diagnosis has significant advantages, including the ability to detect abnormal patterns early in cancer progression, potentially improving early diagnosis over traditional imaging methods. However, limitations exist, such as diagnostic accuracy below 80% and a lack of standardized identification methods across different tumor types. Additionally, the biological mechanisms of methylation patterns are not fully understood, which complicates their application in clinical settings."}]

examples['paraphrase'] = [{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs",
"prompt_variant": 'You can paraphrase the question and answer.',
"question_new": "What benefits does the new ESTRO-ACROP guideline provide?",
"answer_new":"The ESTRO-ACROP guideline offers dosimetric advantages by reducing the risk of adverse events related to radiotherapy, such as radiation pneumonitis and fibrosis, while also decreasing unnecessary exposure to important cardiopulmonary organs."},
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "Why is early detection important for breast cancer prognosis?",
"answer": "Early detection is important for breast cancer prognosis because it significantly influences treatment options and outcomes. Studies indicate that a large percentage of breast cancer-related deaths are due to metastasis, making early detection crucial for effective intervention. Mammography, a common screening method, has been shown to reduce breast cancer mortality by approximately 20%. Additionally, identifying abnormal DNA methylation patterns at early stages can provide valuable diagnostic information, potentially leading to more timely and targeted treatments. This underscores the necessity for improved detection methods to enhance early diagnosis and, consequently, patient prognosis.",
"prompt_variant": 'Re-write the question such that its answer is not general and appropriately states exceptions and conditions to demonstrate a deep and nuanced understanding of the question.',
"question_new": "What makes early detection crucial for the prognosis of breast cancer?",
"answer_new": "Early detection is crucial for breast cancer prognosis as it greatly impacts treatment choices and outcomes. A significant number of breast cancer deaths result from metastasis, highlighting the need for prompt intervention. Mammography can lower mortality rates by about 20%. Furthermore, detecting abnormal DNA methylation patterns early can offer essential diagnostic insights, allowing for more timely and targeted therapies. This highlights the need for better detection methods to improve early diagnosis and patient outcomes."}]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evolve the seed data by Few Shot Prompting
# MAGIC Evolve the Q&A by one prompt variant at a time. Checkpoint augmented data every x inputs to .jsonl

# COMMAND ----------

def batch_structured_llm_with_checkpoints(df: pyspark.sql.DataFrame, selected_fields: List[str],
                                          structured_class: Type[QA_augmented],
                                          prompt_variants: dict, split_col: str,
                                          llm: Type[BaseChatModel],
                                          outfile: str, ans: List,
                                          save_every: int = 5, concurrency: int = 5, retries: int = 2,
                                          verbose: bool = False, debug_inputs: str = None) -> List:
    for k,v in prompt_variants.items():
        # Set up chain for batch inference
        prompt = FewShotPromptTemplate(
                examples=examples.get(k),
                example_prompt=example_prompt,
                input_variables=selected_fields,
                prefix=prefix,
                suffix=suffix)
        print(f'Evolving Q&A by {k} prompt...')
        if verbose:
            print(prompt.format_prompt(prompt_variant=v, context='<context>',
                                       question='<question>', answer='<answer>').text)
        structured_llm = llm.with_structured_output(structured_class)
        chain = prompt | structured_llm
        inputs = df.where(col(split_col)==k) \
                .select(*selected_fields) \
                .dropDuplicates() \
                .toPandas().to_dict("records")

        # batch run every x inputs
        for i in range(0, len(inputs), save_every):
            end = i + save_every
            subbatch = inputs[i:end]
            print(f"Evolving {i}th Q&A")
            # For debugging dupes
            if debug_inputs:
                write_jsonl_by_line(subbatch, debug_inputs)
            if verbose:
                print(f"Original question {subbatch[0].get('question')}")
            try:
                responses = chain.with_retry(stop_after_attempt=retries) \
                    .batch(subbatch, config={"max_concurrency": concurrency})

                # Store the context, original question and answer in the response dictionary (but not passed into the LLM unnecessarily)
                for i, r in zip(inputs, responses):
                    if r:
                        for k,v in i.items():
                            r[k] = v
                        if verbose:
                            pprint(r)

                responses = [r for r in responses if r and len(set(r.values()).intersection({None,'None','null'}))==0]
                # Write to jsonl after every x inputs
                write_jsonl_by_line(responses, outfile)
                ans.extend(responses)

            except Exception as e:
                print(f"Exception of type {type(e)}.\n{e}")

# COMMAND ----------

llm = ChatDatabricks(endpoint=model_evolve, temperature=temperature)

batches = []
batch_structured_llm_with_checkpoints(
    df=seed_promptvariant, selected_fields=selected_fields,
    structured_class=QA_augmented,
    prompt_variants=prompt_variants, split_col="variant",
    llm=llm, outfile=outfile, ans=batches,
    save_every=5, concurrency=max_concurrency, retries=max_retries, 
    verbose=False, debug_inputs="data/inputs.jsonl")

# COMMAND ----------

len(batches)

# COMMAND ----------

# Option 1 to save evolved data: Read from batches in memory
evolved_df = spark.createDataFrame(pd.DataFrame.from_records(batches))

# Option 2 to save evolved data: Read in from jsonl (if cluster stopped)
#evolved_df = spark.createDataFrame(pd.read_json("data/evolved.jsonl", lines=True))
display(evolved_df)

# COMMAND ----------

evolved_df.na.drop(how='any', subset=["context", "question", "answer", "question_new", "answer_new"]) \
    .dropDuplicates() \
    .write.mode("overwrite") \
    .saveAsTable(evolved_table_name)
display(spark.table(evolved_table_name))

# COMMAND ----------

evolved_df.count()

# COMMAND ----------

# For debugging
inputs_df = spark.createDataFrame(pd.read_json("data/inputs.jsonl", lines=True))
display(inputs_df)
