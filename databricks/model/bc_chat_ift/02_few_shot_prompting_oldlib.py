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

# MAGIC %pip freeze

# COMMAND ----------

# MAGIC %pip install langchain_databricks langchain_openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import re, os, json
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
# MAGIC ## Few Shot Prompting (FSP)
# MAGIC #### Curate a list of examples for FSP

# COMMAND ----------

examples = [   
{"topic": "breast cancer biomarkers",
 "context": "1 Introduction\n\nBreast cancer (BRCA) is one of the most common malignant tumors in women\nworldwide and a major cause of cancer-related deaths among women globally\n(Sung et al., 2021). The number of new BRCA cases is on the rise annually\nacross the world, particularly in developing countries. Studies have shown\nthat accounts for approximately 60%–90% of BRCA-related deaths are attributed\nto metastasis of tumor (Dillekas et al., 2019; Krishnan et al., 2021). Thus,\nearly detection is critical for BLCA treatment and prognosis. Mammography and\nultrasound have been utilized for standardizing breast lesion risk assessment,\namong which mammography screening reduced breast cancer mortality by ∼20%\n(Screening, 2012; Christiansen et al., 2022). However, they are susceptible to\nhigh false positive rates, resulting in unnecessary biopsies. Particularly in\ncase of high-density breast tissue, the detection sensitivity is compromised.\nTherefore, there is a need for new precise detection methods to compensate for\nthe deficiency in breast lesion detection.\n\nThe occurrence and progression of tumors are accompanied by a series of\nreconstruction processes of the genome and epigenome (Chakravarthi et al.,\n2016; Ushijima et al., 2021). Among them, DNA methylation is an epigenetic\nmechanism that regulates gene expression and chromatin structure in a complex\nway affecting gene expression. Studies have confirmed that abnormal\nmethylation patterns play an important role in the occurrence and progression\nof breast cancer and other malignant tumors (Kulis and Esteller, 2010; Ma et\nal., 2023). Since the DNA methylation modification process precedes protein\ntranslation, abnormal methylation patterns can be detected in the early stages\nof cancer development, and thus DNA methylation markers may have greater value\nin early diagnosis of breast cancer compared to detecting cancer-related\nprotein expression levels. Currently, the most widely used diagnostic\napplication related to methylation modification is the Sept9 methylation\ndetection for early diagnosis of colorectal cancer based on cell-free DNA (cf-\nDNA) in peripheral blood (Galanopoulos et al., 2017; Fu et al., 2018), but its\naccuracy for early diagnosis is less than 80%. For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients. Its effectiveness for early breast cancer diagnosis is also\nnot ideal. Therefore, it is more commonly applied for preoperative detection\nand monitoring disease progression after surgery. Developing new methods for\nearly breast cancer diagnosis, especially DNA methylation biomarkers, thus has\ngreat significance for the implementation of precision medicine for breast\ncancer.\n\nIn healthy individuals, 70%–80% of CpG sequences are in a methylated state,\nwhich is very important for maintaining body functions (Sulewska et al.,\n2007). In tumor cells, specific genes experience high methylation of CpG\nislands, termed CpG island methylator phenotype (CpG island methylator\nphenotype, CIMP). Different tumors have different detection sites for CIMP,\nbut the biological mechanisms and pathogenesis of most CIMPs have not been\nclearly studied. Therefore, there are no unified identification methods and\n\n",
"question": "What cancer antigen is used for detection of breast cancer?",
"source": "For detecting breast cancer,\nCA153 antigen detection is used as a breast cancer diagnostic method (Stefan-\nvan Staden and van Staden, 2013; Tang et al., 2016). The antigen detected by\nthis assay kit is significantly elevated in the serum of late-stage breast\ncancer patients.",
"answer": "CA 15-3. It is usually significantly elevated in the serum of late-stage breast cancer patients."}]

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
    topic: Annotated[Optional[str], ..., "Topic for querying similar chunks"]

# COMMAND ----------

# MAGIC %md
# MAGIC Set up the appropriate prompts

# COMMAND ----------

prefix = """Given the context below, generate a question about {topic} that can be answered following these rules:
Rules:
1. The question should make sense to humans even when read without the given context.
2. The question should be fully answered from the given context.
3. The question should be reasonable and answerable by humans.
4. Do not use phrases like 'provided context', etc. in the question.
5. Avoid framing questions using the word "and" that can be decomposed into more than one question.
6. The question should not be longer than 15 words.
7. The answer to the question should not contain any links.

Some examples are provided below.
Examples:
"""

suffix = """To generate the question, first identify the most important or relevant part of the context and return that in the "source" field. Then frame a question around that part that satisfies all the rules above and return the question in the "question" field. Lastly, return the answer in the "answer" field.

Context:
{context}
"""

example_prompt = PromptTemplate.from_template("""Topic: {topic}
                                              Context: {context}
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
        input_variables=["context", "topic"],
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
responses = chain.batch(inputs[0:3],
                        config={"max_concurrency": 3})

# COMMAND ----------

# Checkpoint every few inputs
def generate_seed_data(inputs: List[dict], outfile: str = 'data/seed.jsonl', 
                       save_every_inputs: int = 10, concurrency: int = 2):
  pieces = []
  for i in range(0, len(inputs), save_every_inputs):
    end = i + save_every_inputs
    subset = inputs[i:end]
    print(f"Generating Q&A for {subset[0]['topic']}")
    responses = chain.batch(subset,
                            config={"max_concurrency": concurrency})
    
    # Ensure the full context is used but not unneccessarily sent to and fro into llm
    for s, r in zip(subset, responses): 
      if r: 
          r['context'] = s['context']
    
    # Write to jsonl after every x inputs
    # TODO: incrementally write to spark
    with open(outfile, 'a+') as out:
      for r in responses:
          if r:
            jout = json.dumps(r) + '\n'
            out.write(jout)
    
    pieces.append(inputs[i:end])
  return pieces

# COMMAND ----------

# TODO: Save to UC volume
pieces = generate_seed_data(inputs, outfile='data/seed.jsonl', 
                            save_every_inputs=2, concurrency=2)

# COMMAND ----------

# TODO: read in jsonl and save as pyspark

# COMMAND ----------


