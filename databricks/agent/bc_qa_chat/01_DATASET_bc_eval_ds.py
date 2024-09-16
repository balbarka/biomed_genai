# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Breast Cancer Evaluation Dataset - Simple
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

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Retrieve Agent Configs
# MAGIC
# MAGIC Similar to how we maintained a shared configuration across notebooks in our 'pubmed' workflow, we will again use shared configurations across all of the notebooks for the agent application `bc_qa_chat`.

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_qa_chat $SHOW_GOVERNANCE=true $SHOW_AGENT_DEPLOY=false

# COMMAND ----------

from biomed_genai.agent.eval import EvalSet, EvalSetEntry, EvalSetRequest

# The dataclass that we'll use to develop our evaluation dataset
bc_eval = EvalSet()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Evaluation Question 1 - Current Advancements
# MAGIC
# MAGIC **Rationale**: This question assesses the model's ability to provide information on modern screening techniques like 3D mammography, MRI, and ultrasound, as well as emerging methods like liquid biopsy and AI-enhanced imaging. It reflects the ongoing efforts to improve early detection rates and reduce false positives/negatives.

# COMMAND ----------

query = """What are the current advancements in breast cancer screening technologies, and how do they improve early detection?"""

response = """Current advancements in breast cancer screening technologies include 3D mammography (tomosynthesis), which provides clearer and more detailed images of breast tissue, and AI-enhanced imaging, which improves the accuracy of detecting abnormalities. These technologies improve early detection by reducing false positives and negatives, enabling earlier intervention, and potentially leading to better patient outcomes."""

entry = EvalSetEntry(request_id="bc_1",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluation Question 2 - Genetic Mutations
# MAGIC
# MAGIC **Rationale**: Understanding genetic predispositions is crucial for breast cancer research. This question evaluates the model's capability to explain the impact of specific genetic mutations on breast cancer risk and discuss preventive strategies like prophylactic surgeries, lifestyle changes, and targeted therapies.
# MAGIC
# MAGIC

# COMMAND ----------

query = """How do genetic mutations, such as BRCA1 and BRCA2, influence breast cancer risk, and what preventive measures are available for high-risk individuals?"""

response = """Genetic mutations in BRCA1 and BRCA2 significantly increase breast cancer risk by impairing the genes' ability to repair DNA, leading to a higher likelihood of cancerous cell growth. For high-risk individuals, preventive measures include enhanced screening protocols, lifestyle modifications, chemoprevention with medications like tamoxifen, and risk-reducing surgeries such as prophylactic mastectomy or oophorectomy."""

entry = EvalSetEntry(request_id="bc_2",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Evaluation Question 3 - Targeted Therapies and Immunotherapies
# MAGIC
# MAGIC
# MAGIC **Rationale**: This question focuses on assessing the model's knowledge of the latest treatments, such as PARP inhibitors, CDK4/6 inhibitors, and immune checkpoint inhibitors, comparing their efficacy, side effects, and application to traditional chemotherapy methods.

# COMMAND ----------

query = """What are the recent developments in targeted therapies and immunotherapies for breast cancer, and how do they compare to traditional chemotherapy?"""

response = """Recent developments in targeted therapies for breast cancer include drugs like CDK4/6 inhibitors and HER2 inhibitors, which specifically target cancer cell growth pathways, and immunotherapies like checkpoint inhibitors, which enhance the immune system's ability to attack cancer cells. Compared to traditional chemotherapy, these therapies tend to have fewer side effects and can be more effective for certain subtypes of breast cancer, offering personalized treatment options that improve patient outcomes."""

entry = EvalSetEntry(request_id="bc_3",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Evaluation Question 4 - Tumor Microenvironment
# MAGIC
# MAGIC **Rationale**: The tumor microenvironment plays a significant role in cancer development and treatment resistance. This question tests the model's understanding of the biological interactions within the tumor microenvironment and how they can be targeted to improve treatment outcomes.

# COMMAND ----------

query = """How does the tumor microenvironment contribute to breast cancer progression, and what are the implications for treatment?"""

response = """The tumor microenvironment in breast cancer, composed of immune cells, blood vessels, and extracellular matrix, supports tumor growth and metastasis by providing necessary signals and nutrients. Understanding this environment has led to treatments targeting these interactions, such as anti-angiogenic therapies and immune checkpoint inhibitors, which aim to disrupt the supportive network of the tumor."""

entry = EvalSetEntry(request_id="bc_4",
                     request=EvalSetRequest.from_query(query),
                     expected_response=response)
bc_eval.set.append(entry)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Evaluation Question 5 - Psychosocial Impacts
# MAGIC
# MAGIC **Rationale**: Beyond the biological and clinical aspects, the psychological impact of breast cancer is a crucial area of research. This question evaluates the model's ability to address the emotional and social challenges faced by patients and suggests ways healthcare providers can offer comprehensive support.

# COMMAND ----------

query = """What are the psychosocial impacts of breast cancer diagnosis and treatment, and how can healthcare providers support patients' mental health throughout the process?"""

response = """Breast cancer diagnosis and treatment can lead to significant psychosocial impacts, including anxiety, depression, body image issues, and a reduced quality of life, affecting both patients and their families. Healthcare providers can support patients' mental health by offering counseling services, facilitating support groups, and integrating psychosocial care into treatment plans to help patients cope with emotional challenges."""

entry = EvalSetEntry(request_id="bc_5",
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

@dataclass
class UC_Dataset(UC_Table):
    # A Dataset Class, same as a UC_Table, but with properties for the following conventions:
    # The release_version is maintained 
    release_version: int

    @property
    def ds(self):
        return mlflow.data.load_delta(table_name=self.name,
                                      name=self.ds_release_version_name,
                                      version=self.release_version)

    @property
    def ds_name(self):
        return self.name.split('.')[-1]

    @property
    def ds_release_version_name(self):
        return f'{self.ds_name}-{self.release_version:03}'

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM biomed_genai.agents.bc_eval_ds

# COMMAND ----------

from delta.tables import DeltaTable

dt = DeltaTable.forName(spark, "biomed_genai.agents.bc_eval_ds")

# COMMAND ----------

biomed_genai.agents.bc_eval_ds

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE biomed_genai.agents.bc_eval_ds
# MAGIC SET TBLPROPERTIES ('bc_qa_chat.release_verions' = '2');

# COMMAND ----------

dat = spark.sql("DESCRIBE FORMATTED biomed_genai.agents.bc_eval_ds")
display(dat)

# COMMAND ----------

dt.detail()

# COMMAND ----------

# We'll want to have a create or replace method that will have the following behavior:
# An overwrite will clean up the delta version - the convention is that the eval dataset matches the 
#
def create_or_replace_delta(self, uc_name:str, overwrite=False, release_version: int=None):
    # Create or replace relta

    bool_insert=False
    try:
        table = self.spark.table(uc_name)
        if ~table.isEmpty():
            bool_insert=True
    except AnalysisException:
        bool_insert=True
    if bool_insert or overwrite:
        print(f'Insert Overwrite {uc_name}.')
        self.spark.createDataFrame(self.as_df).write.format("delta").mode("overwrite").saveAsTable(uc_name)
    else:
        print(f'No Action, {uc_name} is not empty.')

# COMMAND ----------

bc_eval.__class__

# COMMAND ----------

# Display the local pandas datafame of eval_ds
display(bc_eval.as_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Evaluation Traces to populate `retrieved_context` [OPTIONAL]
# MAGIC
# MAGIC Above we completed the part of the Evaluation Dataset that can be provided by domain experts. However, it is a bit more tedious for domain experts to write the desired retrieved context. Thus, we can use the following to interactively pull set that retrieved_context we may want to add above.
# MAGIC
# MAGIC **NOTE**: This will not yield any results if this is your first iteration.
# MAGIC
# MAGIC **NOTE**: If you see more than one result, that is likely because multiple models versions or multiple dataset versions exist.
# MAGIC
# MAGIC **TODO**: Update eval dataclasses to include retrieved_context and provide an example above.

# COMMAND ----------

INTERACTIVE_TRACE = True

if INTERACTIVE_TRACE:
    import mlflow

    client = mlflow.tracking.MlflowClient()
    request_id = "bc_1"
    
    experiment_ids = [bc_qa_chat.experiment.experiment_id, ]
    filter_string = f'tags.eval.requestId = "{request_id}"'

    client.search_traces(experiment_ids = experiment_ids,
                         filter_string = filter_string)
    

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Validate Evaluation Dataset [OPTIONAL]
# MAGIC
# MAGIC It can be usefule to make sure that your evaluation dataset works, but we don't want to necessarily have to have a candidate model to do so. In this case, we'll simply evaluate our dataset and persist in this notebook experiment.
# MAGIC
# MAGIC **NOTE**: We don't want to save this mlflow experiemnt to our agent experiment because we don't consider this a candidate model. This experiemnt run is only to be able to inspect and validate our Evaluation Dataset is performing as desired.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Create a foundation model *function* to test evaluation set
# MAGIC
# MAGIC Since evaluation sets are really ever used with models, we'll create our first model evaluation with a simple chat foundation model with no retriver. We are then able to pass this pyfunc model as the model for evalution. 
# MAGIC
# MAGIC **Note**: We are only going to create a wrapper function for `dbrx` to run eval. `dbrx` is pre-existing, we are adding no additional functionality so therefore we will not be logging a model in this notebook.

# COMMAND ----------

VALIDATE_EVALUATION_DATASET = True

if VALIDATE_EVALUATION_DATASET:
    import mlflow
    import mlflow.deployments

    client = mlflow.deployments.get_deploy_client("databricks")

    @mlflow.trace()
    def dbrx_predict(model_input):
        return client.predict(endpoint="databricks-dbrx-instruct",
                              inputs=model_input)

    # We'll just use our bc_eval from above for a test example
    question = bc_eval.set[0].request.query
    input_example = {"messages": [{"role": "user",
                               "content": question}]}

    dbrx_predict(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Run an evaluation on a Foundation Model Function [OPTIONAL]
# MAGIC
# MAGIC Since we are not saving a model, we can go ahead and run our 
# MAGIC
# MAGIC To inspect the results, use the experiments icon on the right hand side (looks like a beaker).
# MAGIC
# MAGIC **NOTE**: This isn't a proper validation on `retrieved_context` since that field requires retriever and there isn't one in our function. 

# COMMAND ----------

from typing import Callable
from mlflow.models.evaluation.base import EvaluationResult

with mlflow.start_run():
    mlflow.evaluate(data=bc_qa_chat.experiment.eval_ds.ds,
                    model=dbrx_predict,
                    model_type="databricks-agent")
