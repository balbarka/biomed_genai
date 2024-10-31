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

# MAGIC %pip install langchain_databricks langchain_openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re, os, json
import pandas as pd
from pandas.errors import EmptyDataError
from langchain_openai import ChatOpenAI
from langchain_databricks import ChatDatabricks
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from typing import Optional, List
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

# COMMAND ----------

# MAGIC %run ./_setup/params

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query chunks as seed

# COMMAND ----------

df = spark.table("biomed_genai.processed.articles_content")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Re-format chunks most similar to a topic into an input dictionary to langchain
# MAGIC From the chunks queried around various topics, generate a dictionary of the form as input to langchain's chain
# MAGIC ```
# MAGIC [{'context': concatenated_chunks1},
# MAGIC  {'context': concatenated_chunks2},
# MAGIC  ...]
# MAGIC  ```

# COMMAND ----------

chunk_sample =df.select('content').sample(0.0002, seed=0).collect()
inputs = [{'context': row.content} for row in chunk_sample if len(row.content.split())>=50]
len(inputs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Few Shot Prompting (FSP)
# MAGIC #### Curate a list of examples for FSP

# COMMAND ----------

examples = [   
{"context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "What cancer antigen is used for detection of breast cancer?",
"source": "For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients.",
"answer": "CA 15-3. It is usually significantly elevated in the serum of late-stage breast cancer patients."},
{"context": "study, reconstruction timing did not show a significant association with\nbreast complications, and the ESTRO-ACROP target volume delineation method did\nnot affect complications in either two-stage delayed reconstruction or\nimmediate reconstruction subgroups. For implant placement, the differences in\nbreast complications between prepectoral and subpectoral approaches are\ncontroversial yet (29–31). We do find it reassuring that the rates of breast\ncomplications observed in our cohort were generally comparable to those\nreported in previous studies. Our findings suggest that introducing the new\nESTRO-ACROP guideline is feasible for patients who underwent subpectoral\nreconstruction in terms of breast complications.\n\nBased on well-known randomized trials that established hypofractionated\nregimen as an effective alternative for adjuvant RT after breast-conserving\nsurgery and mastectomy (32–35), a multi-institutional study by the Korean\nRadiation Oncology Group evaluated the feasibility of hypofractionated RT\nafter breast reconstruction. It revealed that hypofractionated PMRT can\nimprove breast reconstruction outcomes (36). Other recent retrospective\nstudies also suggested that a hypofractionated regimen was comparable with a\nconventional fractionation in terms of breast-related complications,\nregardless of breast reconstruction type (14) and surgical extent (37).\n\nThe major difference between the conventional and the 2019 ESTRO-ACROP\nguidelines is in the definition of the CTV of the chest wall. Whereas prior\ncontouring guidelines generally included the whole implant, the new ESTRO-\nACROP guidelines removed it from the CTV in selected patients (16, 18). Of\nnote, in patients with subpectoral implant breast reconstruction, where\nimplants were inserted in the pocket between the pectoral major and minor, a\nconvex strip of subcutaneous and remnant breast tissue between the anterior\nand skin of the pectoral major was covered.\n\nThe new ESTRO-ACROP guideline has dosimetric benefits to adjacent normal\norgans when using modern volume-based planning techniques. Chang et al.\ncompared dosimetric characteristics of patients with left-sided breast cancer\nbetween two guidelines in VMAT planning. It revealed that the new target\nvolume delineation method significantly reduced exposure to the heart, left\nanterior descending coronary artery (LAD), and ipsilateral lung, maintaining\ntarget coverage, delivery accuracy, and dose heterogeneity compared with\nconventional delineation (17). Similarly, Milligan et al. also evaluated the\nchanges in normal organ sparing and target coverage with VMAT and pencil-beam\nscan planning, finding that the ESTRO target has dosimetric advantages to\ncardiopulmonary organs (18). Previous studies have shown that increasing\nradiation doses to the heart, left ventricle, and LAD are directly associated\nwith long-term rates of high-grade coronary artery stenosis and acute coronary\nevents (38–40). Also, radiation pneumonitis and radiation fibrosis are well-\nknown toxicities caused by RT in patients with breast cancer, which have a\ncorrelation with increasing radiation dose to the lung (41, 42). It is\nnoteworthy that the new guideline could minimize RT-induced adverse events, as\nmost patients with breast cancer are expected to have long-term survival.\n\nThere might be a concern about increasing recurrences at deep chest wall\nstructures, which\n\n",
"question": "What are the advantages of the new ESTRO-ACROP guideline?",
"source": "",
"answer": "ESTRO-ACROP has dosimentric benefits, minimizing radiotherapy-induced adverse events such as radiation pneumonitis and radiation fibrosis and unnecessary radiation exposure to cardiopulmonary organs"}]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up a class for langchain structured output
# MAGIC Note that ChatDatabricks so far supports TypedDict, JSON but not pydantic schemas

# COMMAND ----------

# https://python.langchain.com/docs/integrations/chat/databricks/
class QA_context_source(TypedDict):
    context: Annotated[str, ..., "Chunks of articles most similar to a topic queried from Vector Store"]
    source: Annotated[str, ..., "Key sentence in chunk as the source for framing a question"]
    question: Annotated[str, ..., "Question generated"]
    answer: Annotated[str, ..., "Generated answer to the question"]

# COMMAND ----------

# MAGIC %md
# MAGIC Set up the appropriate prompts

# COMMAND ----------

prefix = """Given the context below, generate a question that can be answered following these rules:
Rules:
1. The context should be 1-3 paragraphs of text from a medical journal. Otherwise, ignore and return None.
2. If the context is mostly about numbers, a table, a figure caption, equation or code, ignore and return None. 
3. If the context is mostly about acknowledging authors' contributions, ignore and return None.
4. The question should be fully answered from the given context.
5. The question should be reasonable and answerable by humans.
6. Do not use phrases like 'provided context', etc. in the question.
7. Avoid framing questions using the word "and" that can be decomposed into more than one question.
8. The question should not be longer than 15 words.
9. The question should be about breast cancer and not broadly about medical research, such as "what is a cell" or "what is an observational study".
10. The answer to the question should be based on the given context, not contain any links or extra information.
11. Be as precise as possible with answering the question.

Some examples are provided below.
Examples:
"""

suffix = """To generate the question, first identify the most important or relevant part of the context and return that in the "source" field. Then frame a question around that part that satisfies all the rules above and return the question in the "question" field. Lastly, return the answer in the "answer" field.

Context:
{context}
"""

example_prompt = PromptTemplate.from_template("""Context: {context}
                                              Question: {question}
                                              Source: {source}
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
prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        input_variables=["context"],
        prefix=prefix,
        suffix=suffix)
structured_llm = llm.with_structured_output(QA_context_source)
chain = prompt | structured_llm

# COMMAND ----------

# MAGIC %md
# MAGIC #### Invocation

# COMMAND ----------

# MAGIC %md
# MAGIC Test invocation with 1-3 inputs

# COMMAND ----------

# Invoke with single input
# response = chain.invoke(inputs[0])

# View output
# response.dict() #pydantic
# response #TypedDict

# Batch invocation with multiple inputs
responses = chain.batch(inputs[3:10],
                        config={"max_concurrency": 3})
responses

# COMMAND ----------

i=0
save_every_inputs=10
concurrency=2

end = i + save_every_inputs
subset = inputs[i:end]
responses = chain.batch(subset,
                    config={"max_concurrency": concurrency})
responses

# COMMAND ----------

len(responses)

# COMMAND ----------

for s, r in zip(subset, responses):
  if isinstance(r, dict): 
      r['context'] = s.get('context')
      if len(set(r.values()).intersection({None,'None','null'})) > 0:
        responses.remove(r)
responses

# COMMAND ----------

# Checkpoint every few inputs
def generate_seed_data(inputs: List[dict], outfile: str = 'data/seed.jsonl', 
                       save_every_inputs: int = 10, concurrency: int = 2):
  pieces = []
  for i in range(0, len(inputs), save_every_inputs):
    try:
      end = i + save_every_inputs
      subset = inputs[i:end]
      responses = chain.batch(subset,
                              config={"max_concurrency": concurrency})
    
      # Ensure the full context is used but not unneccessarily sent to and fro into llm
      for s, r in zip(subset, responses):
        if isinstance(r, dict): 
            r['context'] = s.get('context')

    except Exception as e:
      print("Exception of type {type(e)}.\n{e}")
    
    # Write to jsonl after every x inputs
    # TODO: incrementally write to spark
    with open(outfile, 'a+') as out:
      for r in responses:
          # r must not be None 
          # or contain 'None' values if r is a dict
          if r and len(set(r.values()).intersection({None,'None','null'})) == 0:
            jout = json.dumps(r) + '\n'
            out.write(jout)
    
    pieces.append(inputs[i:end])
  return pieces

# COMMAND ----------

len("Tet-on ctrl and Tet-on 3\u2019 UTR HP DG NSCs by nucleofection and the levels of EGFPd2 (GFP) protein and mRNA compared to the expression by NSCs expressing a CFP control construct (CAG::IRES-Cfp) after 48 hr of doxycycline induction".split())

# COMMAND ----------

# TODO: Save to UC volume
pieces = generate_seed_data(inputs, outfile='data/seed.jsonl', 
                            save_every_inputs=3, concurrency=2)

# COMMAND ----------

pieces

# COMMAND ----------

# TODO: read in jsonl and save as pyspark
