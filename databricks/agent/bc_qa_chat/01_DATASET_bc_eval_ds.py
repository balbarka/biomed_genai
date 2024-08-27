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

# MAGIC %run ./_setup/setup_bc_qa_chat

# COMMAND ----------

from biomed_genai.agent.bc_qa_chat.agent_bc_qa_chat import Agent_model_bc_qa_chat

# The configurations for all of our bc_qa_chat agent application
bc_qa_chat = Agent_model_bc_qa_chat(**config_bc_qa_chat)

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

# COMMAND ----------

# Display the local pandas datafame of eval_ds
display(bc_eval.as_df)

# COMMAND ----------

bc_eval.create_or_replace_delta(bc_qa_chat.experiment.eval_ds.name,
                                overwrite=False)

# COMMAND ----------

# Display the spark dataframe of eval_ds
display(spark.table(bc_qa_chat.experiment.eval_ds.name))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC That's it! We jut created our first version of our evaluation dataset. We'll revisit this delta table again and iterate on it by simply creating newer delta table versions.
# MAGIC
# MAGIC To see our dataset in action, let's create our first agent application model. Check out: 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create a foundation model *function* to test evaluation set
# MAGIC
# MAGIC Since evaluation sets are really ever used with models, we'll create our first model evaluation with a simple chat foundation model with no retriver. We are then able to pass this pyfunc model as the model for evalution. It's common to call mlflow.evaluate run an evaluate at the same time as creating it and is the convention that we will use for all agent application notebooks.
# MAGIC
# MAGIC **Note**: We are only going to create a wrapper function for `dbrx` to run eval. `dbrx` is pre-existing and therefore we will not be logging a model in this notebook.
# MAGIC
# MAGIC **Note**: We could create a competing convention that would require us to create a wrapper class for a foundation model. I'm opposed to this convention since foundation models are already well defined, making multiple copies of them wrappend in additional classes that could create confusion. Thus, we are not creating a model that could potentially get promoted where the model is just a pass through to a model serving endpoint.

# COMMAND ----------

import mlflow
import mlflow.deployments

client = mlflow.deployments.get_deploy_client("databricks")

@mlflow.trace(name="dbrx_preidct_test")
def dbrx_predict(model_input):
    return client.predict(endpoint="databricks-dbrx-instruct",
                          inputs=model_input)

question = bc_eval.set[0].request.query
input_example = {"messages": [{"role": "user",
                               "content": question}]}

rslt = dbrx_predict(input_example)

# COMMAND ----------

import mlflow

# Start the parent run
with mlflow.start_run(run_name="Parent Run") as parent_run:
    # Log a parameter in the parent run
    mlflow.log_param("parent_param", "value1")

    # Start the first nested run
    with mlflow.start_run(run_name="Child Run 1", nested=True):
        mlflow.log_param("child_param_1", "value2")
        mlflow.log_metric("child_metric_1", 0.85)
        
        # Example of logging a model or artifact
        with open("output.txt", "w") as f:
            f.write("This is a sample artifact from Child Run 1")
        mlflow.log_artifact("output.txt")

    # Start the second nested run
    with mlflow.start_run(run_name="Child Run 2", nested=True):
        mlflow.log_param("child_param_2", "value3")
        mlflow.log_metric("child_metric_2", 0.92)

    # Log a metric in the parent run
    mlflow.log_metric("parent_metric", 0.95)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Run an evaluation on a Foundation Model Function
# MAGIC
# MAGIC **NOTE**: We are going to use a convention where we will have parent / child runs within an mlflow experiment. While it is possible and sometimes useful to write an evaluation expeiment run in the same experiment run, we will instead opt to write our evaluation runs as a nested run.
# MAGIC
# MAGIC **TODO**: Write explaination of artifacts that are created from the method `create_mlflow_eval`

# COMMAND ----------

# TODO: fix working directly with ds and pull arguments from biomed

from typing import Callable
from mlflow.models.evaluation.base import EvaluationResult

def create_mlflow_eval(predict_fn: Callable,
                       experiment_name='/experiments/biomed_app/qa_chat',
                       run_name='dbrx',
                       eval_ds_uc_name="",
                       allow_duplicates=False) -> EvaluationResult:
    mlflow.set_experiment(experiment_name=experiment_name)
    runs = mlflow.search_runs(experiment_names=[experiment_name,],
                              filter_string=f'tags.mlflow.runName = "{run_name}"')
    if (len(runs) == 0) or allow_duplicates:
        with mlflow.start_run(run_name="dbrx"):
            print(f'Creating "{run_name}" run.')
            return mlflow.evaluate(data=mlflow.data.load_delta(table_name=eval_ds_uc_name,
                                                               name=eval_ds_uc_name.split(".")[-1]),
                                   model=predict_fn,
                                   model_type="databricks-agent")
    else:
        print(f'"{run_name}" tags.mlflow.runName already exists.')
        metrics = mlflow.get_run(run_id=runs.loc[runs.end_time.idxmin()]['run_id']).data.to_dictionary()['metrics']
        return mlflow.models.EvaluationResult(metrics=metrics,artifacts=None)

eval_rslt = create_mlflow_eval(dbrx_predict,
                               experiment_name='/experiments/biomed_app/qa_chat',
                               run_name='dbrx',
                               eval_ds_uc_name="biomed_genai.processed.eval_ds")

# COMMAND ----------

eval_rslt = create_mlflow_eval(dbrx_predict,
                               experiment_name='../Volumes/biomed_genai/models/experiments/qa_chat/xxx',
                               run_name='dbrx',
                               eval_ds_uc_name="biomed_genai.processed.eval_ds")

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC
# MAGIC from typing import Callable
# MAGIC from mlflow.models.evaluation.base import EvaluationResult
# MAGIC from mlflow.data.spark_dataset import SparkDataset
# MAGIC
# MAGIC
# MAGIC def create_mlflow_eval(eval_ds: SparkDataset,
# MAGIC                        predict_fn: Callable,
# MAGIC                        experiment_name='/experiments/biomed_app/qa_chat',
# MAGIC                        run_name='dbrx',
# MAGIC                        allow_duplicates=False) -> EvaluationResult:
# MAGIC     mlflow.set_experiment(experiment_name=experiment_name)
# MAGIC     runs = mlflow.search_runs(experiment_names=[experiment_name,],
# MAGIC                               filter_string=f'tags.mlflow.runName = "{run_name}"')
# MAGIC     if (len(runs) == 0) or allow_duplicates:
# MAGIC         with mlflow.start_run(run_name="dbrx"):
# MAGIC             print(f'Creating "{run_name}" run.')
# MAGIC             return mlflow.evaluate(data=eval_ds,
# MAGIC                                    model=predict_fn,
# MAGIC                                    model_type="databricks-agent")
# MAGIC     else:
# MAGIC         print(f'"{run_name}" tags.mlflow.runName already exists.')
# MAGIC         metrics = mlflow.get_run(run_id=runs.loc[runs.end_time.idxmin()]['run_id']).data.to_dictionary()['metrics']
# MAGIC         return mlflow.models.EvaluationResult(metrics=metrics,artifacts=None)
# MAGIC
# MAGIC eval_rslt = create_mlflow_eval(eval_ds=biomed.processed_eval_ds.ds,
# MAGIC                                predict_fn=dbrx_predict,
# MAGIC                                experiment_name='/experiments/biomed_app/qa_chat',
# MAGIC                                run_name='dbrx')```
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Typically you will use the UI to inspect the metrics, but you can also inspect them from the EvaluationResult dataclass that is returned when you run an evaluation.

# COMMAND ----------

INSPECT_METRICS = True

if INSPECT_METRICS:
    for k,v in eval_rslt.metrics.items():
        print(f'{k}: {v}')
