# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is second in a series that generates synthetic data for Instruction Fine Tuning (IFT).
# MAGIC
# MAGIC What this notebook does:
# MAGIC 1. From a chunkset as context, generate a question that could be posed.
# MAGIC 2. Generate an answer to the question using only the chunkset as a source.
# MAGIC 3. Pinpoint the sentence in the chunkset as the source that answers the question.
# MAGIC The steps 1-3 are done by Few Shot Prompting
# MAGIC 4. Periodically save the context, question, answer and source in a jsonl.

# COMMAND ----------

# MAGIC %pip install langchain==0.2.16 langchain-community==0.2.7 langchain-core==0.3.15 langchain_databricks>=0.1.1 langchain_openai==0.2.5
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re, os, json
import pandas as pd
from pandas.errors import EmptyDataError
from langchain_openai import ChatOpenAI
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from typing import Optional, List
from pydantic import BaseModel, Field
from typing import Type
from typing_extensions import Annotated, TypedDict
from _setup.params import *

# COMMAND ----------

#%run ./_setup/params

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query chunks as seed

# COMMAND ----------

df = spark.table("biomed_genai.processed.articles_content")
display(df)

# COMMAND ----------

#df.where(df.id=="PMC7616065-31").select('content').toPandas().values[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Re-format chunks most similar to a topic into an input dictionary to langchain
# MAGIC From the chunks queried around various topics, generate a dictionary of the form as input to langchain's chain.
# MAGIC ```
# MAGIC [{"context": concatenated_chunks1},
# MAGIC  {"context": concatenated_chunks2},
# MAGIC  ...]
# MAGIC  ```

# COMMAND ----------

chunk_sample =df.select('content').sample(0.02, seed=0).collect()
inputs = [{"context": row.content} for row in chunk_sample if len(row.content.split())>=50]
len(inputs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Few Shot Prompting (FSP)
# MAGIC #### Curate a list of examples for FSP

# COMMAND ----------

examples_gen = [   
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "What cancer antigen is used for detection of breast cancer?",
"answer": "CA 15-3. It is usually significantly elevated in the serum of late-stage breast cancer patients."},
{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs"}]

# COMMAND ----------

examples_judge = {"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "What cancer antigen is used for detection of breast cancer?",
"answer": "CA 15-3. It is usually significantly elevated in the serum of late-stage breast cancer patients.",
"domain_specificity": True,
"validity": True,
"relevance": True,
"correctness": True},
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "What cancer antigen is used for detection of breast cancer?",
"answer": "CA 123. It is usually significantly elevated in the serum of late-stage breast cancer patients.",
"domain_specificity": True,
"validity": True,
"relevance": True,
"correctness": False},
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "What cancer antigen is used for detection of skin cancer?",
"answer": "CA 15-3. It is usually significantly elevated in the serum of late-stage breast cancer patients.",
"domain_specificity": False,
"validity": True,
"relevance": False,
"correctness": False},
{"context": 'ref-type="bibr" rid="CIT0290">2017)  \nYes| ↓ systolic and diastolic blood pressure, ↔ arterial stiffness| Men with\nprediabetes, RT, n = 16| TRF (early)| (Sutton et al. 2018)  \nYes| ↓ systolic and diastolic blood pressure| Healthy adults, RCT, n = 185|\nCR| (Kraus et al. 2019)  \nAging and aging-associated pathology  \n↓ neurodegeneratione.g. ↑ cognitive function, ↓ amyloid pathology| CR – mouse,\nfemale (Qin et al. 2006); FMD – mouse, female (Brandhorst et al. 2015); IF –\nmouse (Li, Wang, and Zuo 2013); KD – mouse, Alzheimer’s model (Auwera et al.\n2005; Xu et al. 2022); rat, Parkinson’s model, male (Kuter et al. 2021); TRF –\nmouse, chronic cerebral hypoperfusion model, male (Selvaraji et al. 2022)|\nYes| ↑ cognitive subscale| Alzheimer’s disease patients, BCS, n = 10| KD|\n(Taylor et al. 2018)  \nYes| ↑ scale improvement for daily living activities and Addenbrookes\nCognitive Examination| Alzheimer’s disease patients, RCT/X, n = 21| KD|\n(Phillips et al. 2021)  \nYes| ↑ cognitive function (digit span test, Trail-Making Test B, and the\nglobal score)| Elderly non-demented individuals, BCT, n = 19| KD| (Ota et al.\n2016)  \n↓ senolytic cell burden | Senolytic treatment with dasatinib & quercetin (Thadathil et al. 2022)| Yes| ↓ senescent cell burden in adipose tissue and skin ↓ IL-1α, IL-6, MMP-9 and −12| Patients with diabetic kidney disease, BCS, n = 9| Senolytic (Dasatinib + Quercetin)| (Hickson et al. 2019)  \n↑ lifespan, ↓ non-neurodenegerative age-related disease and cancer*\n\n',
"question": "How many men have prediabetes?",
"answer": "16",
"domain_specificity": False,
"validity": False,
"relevance": True,
"correctness": True}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up a class for langchain structured output
# MAGIC Note that ChatDatabricks so far supports TypedDict, JSON but not pydantic schemas

# COMMAND ----------

# https://python.langchain.com/docs/integrations/chat/databricks/
class QA_context(TypedDict):
    context: Annotated[str, ..., "Chunks of articles most similar to a topic queried from Vector Store"]
    question: Annotated[str, ..., "Question generated"]
    answer: Annotated[str, ..., "Generated answer to the question"]

# COMMAND ----------

class QA_quality(TypedDict):
    domain_specificity: Annotated[bool, ..., "context is about breast cancer in human patients"]
    validity: Annotated[bool, ..., "context is not about numbers, a table, a figure caption, equation or code, or acknowledging authors' contributionsfor framing a question"]
    relevance: Annotated[bool, ..., "question is relevant to the context"]
    correctness: Annotated[bool, ..., "answer reasonably answers the question and relies only on the context and not contain links or extra information"]

# COMMAND ----------

# MAGIC %md
# MAGIC Set up the appropriate prompts for both the prompt to generate Q&A and then to judge it

# COMMAND ----------

prefix_gen = """Given the context below, generate a question that can be answered following these rules:
Rules:
1. The context should be 1-3 paragraphs of text from a medical journal. Otherwise, ignore and return None.
2. If the context is mostly about numbers, a recipe listing reagents, a table, a figure caption, equation or code, ignore and return None. 
3. If the context is mostly about acknowledging authors' contributions, ignore and return None.
4. The question should be fully answered from the given context.
5. The question should be reasonably understood and answerable by a trained scientist.
6. Do not ask highly contextual questions that require referencing to a specific study, for example "What are the main findings of the study" or "how many patients are enrolled in the study".  
7. Do not use phrases like 'provided context', etc. in the question.
8. Avoid framing questions using the word "and" that can be decomposed into more than one question.
9. The question should not be longer than 15 words.
10. The answer should be about 10-80 words long.
11. The question should be about breast cancer and not broadly about medical research, such as "what is a cell" or "what is an observational study".
12. The answer to the question should be based on the given context, not contain any links or extra information.
13. Be as precise as possible with answering the question.

Some examples are provided below.
Examples:
"""

suffix_gen = """To generate the question, first identify the most important or relevant part of the context. Then frame a question around that part that satisfies all the rules above and return the question in the "question" field. Lastly, return the answer in the "answer" field.

Context:
{context}
"""

example_prompt_gen = PromptTemplate.from_template(
"""Context: {context}
Question: {question}
Answer: {answer}""")

# COMMAND ----------

prefix_judge = """Given the context, question and corresponding answer, critique in terms of:
1. Domain-specificity: that the context is about medical science.
2. Validity: that the context is not about numbers, a table, a figure caption, equation or code, or acknowledging authors' contributions.
3. Relevance: that the question is relevant to the context
4. Correctness: the answer reasonably answers the question and relies only on the context and not contain links or extra information.

Return only True/False to the above 4 fields
"""

suffix_judge = """To generate the question, first identify the most important or relevant part of the context and return that in the "source" field. Then frame a question around that part that satisfies all the rules above and return the question in the "question" field. Lastly, return the answer in the "answer" field.

Context:
{context}

Question:
{question}

Answer:
{answer}
"""

example_prompt_judge = PromptTemplate.from_template(
    """Context: {context}
    Question: {question}
    Answer: {answer}""")

# COMMAND ----------

# MAGIC %md
# MAGIC Set up the LLM and chain

# COMMAND ----------

llm = ChatDatabricks(endpoint = model,
                  temperature=TEMPERATURE)
# llm = ChatOpenAI(#model = model,
#                  model="gpt-4o-mini",
#                  api_key=OPENAI_API_KEY,
#                  #base_url=BASE_URL,
#                  temperature=TEMPERATURE
#                  )

# COMMAND ----------

def create_chain(prefix, suffix, 
                 examples, example_prompt, 
                 input_var, structured_class):
    prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        input_variables=input_var,
        prefix=prefix,
        suffix=suffix)
    structured_llm = llm.with_structured_output(structured_class)
    chain = prompt | structured_llm
    return chain, prompt

# COMMAND ----------

# Chain to generate Q&A from context
chain_gen, prompt_gen = create_chain(prefix_gen, suffix_gen,
                 examples_gen, example_prompt_gen,
                 input_var=["context"],
                 structured_class=QA_context)
print(prompt_gen.format_prompt(context="<context>").text)


# COMMAND ----------

# Chain to judge Q&A from context
chain_judge, prompt_judge = create_chain(prefix_judge, suffix_judge,
                 examples=examples_judge,
                 example_prompt=example_prompt_judge,
                 input_var=["context", "question", "answer"],
                 structured_class=QA_quality)
print(prompt_judge.format_prompt(context="<context>",
                                 question="<question>",
                                 answer="<answer>").text)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Invocation
# MAGIC Test invocation with 1-3 inputs

# COMMAND ----------

# Invoke with single input
# response = chain_gen.invoke(inputs[0])

# View output
# response.dict() #pydantic
# response #TypedDict

# Batch invocation with multiple inputs
responses = chain_gen.batch(inputs[21:30],
                        config={"max_concurrency": 3})
responses

# COMMAND ----------

responses_wo_none = [r for r in responses if r and len(set(r.values()).intersection({None,'None','null'})) == 0]
critiques = chain_judge.batch(responses_wo_none,
                        config={"max_concurrency": 3})
critiques

# COMMAND ----------

# MAGIC %md
# MAGIC ### Concurrent batch invocation

# COMMAND ----------

# Checkpoint every few inputs
def generate_seed_data(inputs: List[dict], ans: List[dict],
                       chain_gen: Type[RunnableSequence], chain_judge: Type[RunnableSequence],
                       outfile: str = 'data/seed.jsonl', 
                       save_every: int = 10, concurrency: int = 2):
  for i in range(0, len(inputs), save_every):
    try:
      end = i + save_every
      subset = inputs[i:end]
      print(f"Generating Q & A for {i}th context")
      responses = chain_gen.batch(subset,
                              config={"max_concurrency": concurrency})
#      valid_responses = responses

      # Ensure the full context is used but not unneccessarily sent to and fro into llm
      for s, r in zip(subset, responses):
        if isinstance(r, dict) \
        and isinstance(s, dict) \
        and len(set(r.values()).intersection({None,'None','null'}))==0:
          r['context'] = s.get('context')
        else:
          responses.remove(r)
      # Ensure all None are removed
      responses = [r for r in responses if r \
        and len(set(r.values()).intersection({None,'None','null'}))==0]
      valid_responses = responses

      if chain_judge and responses:
        critiques = chain_judge.batch(responses,
                          config={"max_concurrency": concurrency})
        critique_mask = [all(c.values()) for c in critiques if c]
        valid_responses = [r for r, c in zip(responses, critique_mask) if c]
        #print(valid_responses)

    except Exception as e:
      print(f"Exception of type {type(e)}.\n{e}")

    # Write to jsonl after every x inputs
    # TODO: incrementally write to spark
    with open(outfile, 'a+') as out:
      for r in valid_responses:
          # r must not be None 
          # or contain 'None' values if r is a dict
          if r and len(set(r.values()).intersection({None,'None','null'}))==0:
            jout = json.dumps(r) + '\n'
            out.write(jout)
    ans.extend(valid_responses)

# COMMAND ----------

# TODO: Save to UC volume
ans = []
generate_seed_data(inputs, ans,
                   chain_gen, chain_judge,
                   outfile='data/seed.jsonl',
                   save_every=5, concurrency=5)

# COMMAND ----------

217/2094

# COMMAND ----------

seed_table_name = "yen.syn_data_gen.seed"
spark.createDataFrame(pd.DataFrame.from_records(ans)) \
    .write.mode("overwrite") \
    .saveAsTable(seed_table_name)
display(spark.table(seed_table_name))

# COMMAND ----------

# TODO: read in jsonl and save as pyspark
seed_df = spark.createDataFrame(pd.read_json("data/seed.jsonl", lines=True))
display(seed_df)

# COMMAND ----------

seed_table_name = "yen.syn_data_gen.seed"
seed_df.na.drop(how='any') \
    .dropDuplicates() \
    .write.mode("overwrite") \
    .saveAsTable(seed_table_name)
display(spark.table(seed_table_name))

# COMMAND ----------


