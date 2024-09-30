# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Breast Cancer Guided Evaluation Dataset - Simple
# MAGIC
# MAGIC Approach to developing an Evalaution Set will vary by user preference and experience. Regardless of conventions used to build your evaluation set, you will need to conform to the expected format described in Databricks Documentation on [Evaluation Sets](https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-set.html).
# MAGIC
# MAGIC We'll use Dataclasses (from <a href="$../../../python/biomed_genai/agent/eval.py" target="_blank">eval.py</a>) + notebook approach to build our Evaluation Set, but realize that while this approach is explicit and helpful for learning, it will be more verbose than other approaches such as writing directly to a file in a text editor or built from an existing structured data source like a table. We'll again take a look at these dataclasses when we iterate our eval dataset.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The reason that we develop an evaluation dataset is so that we can us it with a model to run [mlflow.evaluate](https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=mlflow%20evaluate#mlflow.evaluate).
# MAGIC
# MAGIC The evaluation process calculates metrics for two categories; Performance Metrics & Quality Metrics.
# MAGIC
# MAGIC ### Performance Metrics
# MAGIC
# MAGIC | Performance Metric.        | Description |
# MAGIC | -------------------------- | ----------- |
# MAGIC | `total_input_token_count`  | Served generative model input token count |
# MAGIC | `total_output_token_count` | Served generative model output token count |
# MAGIC | `total_token_count`        | Sum of input and output counts |
# MAGIC | `latency_seconds`          | Time of entire agent response |
# MAGIC
# MAGIC
# MAGIC ### Quality Metrics
# MAGIC
# MAGIC | Req'd Fields | Quality Metric | Judge Question | Note |
# MAGIC | ------------ | -------------- | -------------- | ---- |
# MAGIC | `request`    | relevance_to_query | *Is the response relevant to the request?* | |
# MAGIC | `request`    | safety | *Is there harmful content in the response?* | |
# MAGIC | `request`    | groundedness | *Is the response a hallucination or grounded in context?* | |
# MAGIC | `request`    | chunk_relevance | *Did the retriever find relevant chunks?* | Calculated only if model has a retriever component |
# MAGIC | `request`, `expected_response` | correctness | *Overall, did the agent generate a correct response?* | |
# MAGIC | `request`, `expected_response`,</br> `expected_retrieved_context` | document_recall | *How many of the known relevant documents did the retriever find?* | Calculated only if model has a retriever component |
# MAGIC
# MAGIC Depending on which fields in our Evaluation Set are populated, different metrics are provided.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook will make use of some dataclasses based upon the [Evaluation Sets](https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-set.html) schemas. Using that framework, we'll define 5 questions with the following characteristics:
# MAGIC
# MAGIC  * **Representative**: It should accurately reflect the range of requests the application will encounter in production.
# MAGIC  * **Challenging**: It should include difficult and diverse cases to effectively test the full range of the application’s capabilities.
# MAGIC  * **Continually updated**: It should be updated regularly to reflect how the application is used and the changing patterns of production traffic.
# MAGIC
# MAGIC  **NOTE**: While this agent application happens to have a single evaluation dataset, it is also viable to use multiple evaluation datasets which won't be explored in the `bc_eval_ds` agent application.
# MAGIC
# MAGIC  ---
# MAGIC
# MAGIC  **NOTE**: Because this application is interested in the categorization of the person asking the question, we'll want our evaluation dataset entries to cover the following question askers. We'll start with only two question per persona:
# MAGIC
# MAGIC   * **Patient** - give notice to consult a HCP
# MAGIC   * **Researcher** - Talk technical 
# MAGIC   * **General Inquery** - response as an 10th grade reading level
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Retrieve Agent Configs
# MAGIC
# MAGIC We are going to reuse the configs that we've already set in 01-Configuration, `bc_guided_chat`.

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_guided_chat $SHOW_AGENT_MODEL=true $SHOW_NOTEBOOK_TASKS=false

# COMMAND ----------

from biomed_genai.agent.eval import EvalSet, EvalSetEntry, EvalSetRequest

# The dataclass that we'll use to develop our evaluation dataset
bc_eval = EvalSet()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Evaluation Question 1
# MAGIC
# MAGIC **Topic**: Current Advancements / Early Detection 
# MAGIC
# MAGIC **Persona**: General Inquery
# MAGIC
# MAGIC **Rationale**: This is a more generic breast cancer question that assesses the model's ability to provide relevant information on modern screening techniques like 3D mammography, MRI, and ultrasound, as well as emerging methods like liquid biopsy and AI-enhanced imaging. It reflects the ongoing efforts to improve early detection rates and reduce false positives/negatives.

# COMMAND ----------

query = """What are the current advancements in breast cancer screening technologies, and how do they improve early detection? """

response = """Current advancements in breast cancer screening technologies include 3D mammography, which provides clearer and more detailed images of breast tissue, and AI-enhanced imaging, which improves the accuracy of detecting abnormalities. These technologies improve early detection by reducing false positives and negatives, enabling earlier intervention, and potentially leading to better patient outcomes."""

entry = EvalSetEntry(request_id="bc_1",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Evaluation Question 2
# MAGIC
# MAGIC **Topic**: BRCA Gene
# MAGIC
# MAGIC **Persona**: General Inquery
# MAGIC
# MAGIC **Rationale**: This question uses generic language to ask about the BRCA gene, making it clear it is from the general persona, not a patient or researcher.

# COMMAND ----------

query = """How does the BRCA mutation increase breast cancer risk?"""

response = """Genetic mutations in BRCA1 and BRCA2 significantly increase breast cancer risk by impairing the genes' ability to repair DNA, leading to a higher likelihood of cancerous cell growth. For high-risk individuals, preventive measures include enhanced screening protocols, lifestyle modifications, chemoprevention with medication, and surgeries."""


entry = EvalSetEntry(request_id="bc_2",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Evaluation Question 3
# MAGIC
# MAGIC **Topic**: Current Advancements / Early Detection 
# MAGIC
# MAGIC **Persona**: Patient
# MAGIC
# MAGIC **Rationale**: This question riffs off Question 1 above, but uses pronouns that indicate a person seeking personal health guidance rather than a general knowledge response. 

# COMMAND ----------

query = """What are some ways I could screen myself for early breast cancer detection?"""

response = """If you are concerned about your risk of breast cancer, you should set up an appointment to discuss with a Health Care Professional. Be sure to mention any family history.

Additionally, there are a few ways to screen yourself for breast cancer, including: 
* Visual inspection - Look for changes in the shape, size, or symmetry of your breasts. You can also check for dimpling, puckering, or skin texture changes, like an orange peel appearance. 
* Manual exam - Use the pads of your three middle fingers to feel for lumps, thickenings, or hardened knots in your breasts and armpits. You can use light, medium, and firm pressure. You can also squeeze your nipple to check for discharge. 
* Position changes - You can try performing your self-exam while standing, lying down, or in the shower. Lying down can help spread out your breast tissue, making it easier to feel for changes. 

You should check both breasts each month.
Breast self-exam is NOT a substitute for other screening methods, like: Mammography, Ultrasound, MRI (magnetic resonance imaging), and Regular physical exams by a doctor. 
Together, these screening tools can provide a fuller picture of any breast changes.
"""

entry = EvalSetEntry(request_id="bc_3",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Evaluation Question 4
# MAGIC
# MAGIC **Topic**: BRCA Gene
# MAGIC
# MAGIC **Persona**: Patient
# MAGIC
# MAGIC **Rationale**: This question riffs off of question 2 above but frames it from the POV of a patient who wants to check if they have the BRCA gene

# COMMAND ----------

query = """How would I know if I have the BRCA gene? And what are my options if I am positive for it?"""

response = """First, you should speak with your healthcare provider or a genetic counselor to determine if testing for the BRCA gene is right for you. 

To find out if you have a BRCA gene mutation, you need to undergo genetic testing. Here's a step-by-step process to determine if you carry the BRCA1 or BRCA2 mutation, which increases the risk of breast and ovarian cancers:

* Consult a Healthcare Provider: It’s recommended to first meet with a genetic counselor or healthcare provider who specializes in hereditary cancer. They can assess your family and personal medical history to evaluate whether you are a candidate for BRCA testing.

* Genetic Testing: BRCA testing is a simple blood test or a saliva test. The sample is sent to a lab that examines the BRCA1 and BRCA2 genes for mutations. Some companies offer direct-to-consumer genetic tests (like 23andMe), but these typically only screen for a limited number of mutations and may not provide comprehensive information. A healthcare provider can offer a broader, more reliable test.

* Results Interpretation: 
If you test positive for a BRCA mutation, it means you have an increased risk of developing breast, ovarian, and possibly other cancers. However, it doesn’t mean you will definitely develop cancer.
A negative test may indicate that you don’t carry a known BRCA mutation. But if you have a strong family history, other genetic factors could still be at play, and further testing or monitoring may be necessary.

* Next Steps After Testing: If you have a BRCA mutation, your doctor may recommend more frequent screenings, such as mammograms, MRIs, or ultrasounds, starting at a younger age. Some people with BRCA mutations consider preventive measures like medications, surgery (e.g., prophylactic mastectomy), or lifestyle changes to reduce cancer risk. A positive result may also have implications for your family members, as BRCA mutations are inherited. They may want to undergo testing as well.

Genetic testing is a powerful tool for understanding cancer risk and can guide decisions about personalized cancer prevention and early detection strategies."""

entry = EvalSetEntry(request_id="bc_4",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Evaluation Question 5 
# MAGIC
# MAGIC **Topic**: Current Advancements / Early Detection 
# MAGIC
# MAGIC **Persona**: Researcher
# MAGIC
# MAGIC **Rationale**: This question riffs off Question 1 above as well, but looks for a more technically detailed response around what advancements have already been developed for early detection and what are some future-oreiented advancements circulating in the field. This showcases how the same question can be framed 3 different ways for the 3 different persona categories. 

# COMMAND ----------

query = """For women with dense breast tissue, what are the latest methods for early breast cancer detection? Focus on screening and biomarker detection technologies."""

response = """Recent advancements in breast cancer screening have been focused on improving early detection, especially for women with dense breast tissue, which can make tumors harder to detect using traditional mammography. Several key areas have seen significant progress:

*3D Mammography (Tomosynthesis): Unlike traditional 2D mammograms, 3D mammography takes multiple X-ray images of the breast from different angles, creating a layered image. Studies have shown that tomosynthesis improves cancer detection rates and reduces false positives, particularly in women with dense breasts.

*Automated Breast Ultrasound (ABUS): ABUS is an FDA-approved technology that uses sound waves to create detailed images of the breast. It's particularly useful for women with dense breast tissue where mammograms may miss small cancers. ABUS is non-invasive and does not use radiation.

*Contrast-Enhanced Mammography (CEM): This technique combines traditional mammograms with a contrast agent that highlights blood flow patterns often seen in tumors. CEM has shown promise in detecting cancers missed by standard mammograms and can be a cost-effective alternative to MRI for certain high-risk populations.

*Molecular Biomarkers: Research is progressing in identifying specific biomarkers that can indicate the presence of breast cancer early. Blood-based liquid biopsies that detect circulating tumor DNA (ctDNA) or microRNA (miRNA) are being studied as non-invasive tools for early diagnosis. Some tests, like the Breast Cancer Index (BCI), are already available to assess the risk of recurrence, but their role in early detection is expanding.

*AI in Imaging: Artificial intelligence (AI) is being increasingly integrated into breast cancer screening. AI algorithms can assist radiologists by improving the accuracy of mammograms and detecting subtle abnormalities that might be overlooked. AI is particularly beneficial in reducing false positives and streamlining the reading of complex breast images."""

entry = EvalSetEntry(request_id="bc_5",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Evaluation Question 6
# MAGIC
# MAGIC **Topic**: BRCA Gene
# MAGIC
# MAGIC **Persona**: Researcher
# MAGIC
# MAGIC **Rationale**: This question also riffs off 2 and 4 above, but uses technical language to indicate that the person asking is a researcher. 
# MAGIC
# MAGIC https://www.sciencedirect.com/science/article/abs/pii/S1097276523007347?dgcid=coauthor

# COMMAND ----------

query = """ What is the difference between Multiplex Ligation-Dependent Probe Amplification and Digital PCR for BRCA mutation detection?"""

response = """
* Multiplex Ligation-Dependent Probe Amplification (MLPA) detects large-scale deletions/duplications in BRCA genes (copy number variants). It’s cost-effective and good for identifying large rearrangements, but cannot detect small mutations or low-frequency variants.

* Digital polymerase chain reaction (dPCR) is highly sensitive for detecting low-frequency point mutations and small indels, making it ideal for liquid biopsy and tumor monitoring. However, it’s more expensive and less effective for detecting large-scale rearrangements compared to MLPA.

Primer design - In MLPA, a single primer pair is used to amplify multiple targets, while in ddPCR, primers and probes are designed to hybridize with the target DNA. 
Reaction conditions - In MLPA, the ligated probes are amplified during the PCR reaction, while in ddPCR, the rules for designing primers and probes are similar to qPCR. 
Sensitivity - ddPCR has shown good linearity, efficiency, and sensitivity similar to PCR and qPCR. 
Sample preparation - ddPCR sample preparation can take 4-5 hours depending on the number of samples. 
Use cases - MLPA is often used to diagnose conditions caused by copy number variation, while ddPCR may improve the sensitivity to identify mutations in hereditary forms of breast and ovarian cancer. 

"""

entry = EvalSetEntry(request_id="bc_6",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Write eval_ds to a delta table
# MAGIC
# MAGIC You are not required to persist your data as a delta table, but there is already built-in mlflow methods for creating a dataset from a delta table very simple so it is recommended for the UC benefits of accessibility, security, and governance.
# MAGIC
# MAGIC The method we will call is `create_or_replace_delta`. Which writes the questions above into a delta table using the same schema as defined in [Evaluation Sets](https://docs.databricks.com/en/generative-ai/agent-evaluation/evaluation-set.html).
# MAGIC
# MAGIC **TODO**: Change Create or Replace with Create or Merge which will keep cleaner CDC, smaller writes.

# COMMAND ----------

spark.createDataFrame(bc_eval.as_df).write.format("delta").mode("overwrite").saveAsTable(bc_guided_chat.experiment.eval_ds.uc_name)

# COMMAND ----------

display(bc_guided_chat.experiment.eval_ds.df)

# COMMAND ----------

# TODO - @brad to make create DF function

# We'll want to have a create or replace method that will have the following behavior:
# An overwrite will clean up the delta version - the convention is that the eval dataset matches the 

# from pyspark.sql.utils import AnalysisException

# def create_or_replace_delta(self, uc_name:str, overwrite=False, release_version: int=None):
#     # Create or replace relta

#     bool_insert=False
#     try:
#         table = self.spark.table(uc_name)
#         if ~table.isEmpty():
#             bool_insert=True
#     except AnalysisException:
#         bool_insert=True
#     if bool_insert or overwrite:
#         print(f'Insert Overwrite {uc_name}.')
#         self.spark.createDataFrame(self.as_df).write.format("delta").mode("overwrite").saveAsTable(uc_name)
#     else:
#         print(f'No Action, {uc_name} is not empty.')
