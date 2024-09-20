# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create Sync Job [OPTIONAL]
# MAGIC
# MAGIC While we have defined all the steps necessary for refreshing our articles, it is a bit tedious to keep running with interactive notebook sessions. This notebook is completely optional for the solution, but it will create a job and schedule it for recurring execution every sunday morning. This is very valuable as it makes our retriver and therefor our eventual RAG application always current.
# MAGIC
# MAGIC **TODO**: Move reusable methods to \*.py file.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Define Job Cluster
# MAGIC
# MAGIC When running a job, you can use an existing interactive cluster or or a job cluster. We are going to opt to use the later since this will provide us some cost savings. We'll also opt to run only once weekly which will reduce cost by way of infrequent updates. Be aware that PMC does update is articles metadata daily so if the most current articles are critical, you can run this workflow daily.
# MAGIC
# MAGIC We'll define a method, `get_job_cluster`, which will return an appropriate job cluster definition based upon the current cloud provider.
# MAGIC
# MAGIC **TODO**: Add support for AWS, GCP job creation.

# COMMAND ----------

from databricks.sdk.service.jobs import JobCluster
from databricks.sdk.service.compute import ClusterSpec, AzureAttributes, AzureAvailability, DataSecurityMode, RuntimeEngine

def get_job_cluster():
    cloud = spark.conf.get("spark.databricks.cloudProvider")
    if cloud == "Azure":
        return JobCluster(job_cluster_key="BioMed_VS_Sync",
                         new_cluster=ClusterSpec(azure_attributes=AzureAttributes(first_on_demand=1,
                                                                                  availability=AzureAvailability(value="ON_DEMAND_AZURE"),
                                                                                  spot_bid_max_price=-1),
                                                 cluster_name="",
                                                 data_security_mode = DataSecurityMode(value="SINGLE_USER"),
                                                 node_type_id="Standard_D4ds_v5",
                                                 num_workers=0,
                                                 runtime_engine=RuntimeEngine(value="PHOTON"),
                                                 spark_conf={'spark.master': 'local[*, 4]'},
                                                 spark_version='14.3.x-scala2.12'))
    elif cloud == "AWS":
        pass
    elif cloud == "GCP":
        pass


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Define Workflow Tasks
# MAGIC
# MAGIC Of the 7 notebooks in **BioMed GenAI Workflow**, there are only 5 that we need to include as tasks to completely update our vector search index. In order of dependency:
# MAGIC  * <a href="$./01_Metadata_Sync" target="_blank">01_Metadata_Sync</a> Syncs the UC metadata table with PMC. This will ensure that we have a relatively local, fast, current listing of all PMC articles.
# MAGIC  * <a href="$./02_Articles_Ingest" target="_blank">02_Articles_Ingest</a> This notebook runs a job that ingests PMC raw files into a UC volume dependent upon search criteria.
# MAGIC  * <a href="$./03_Curate_Articles" target="_blank">03_Curate_Articles</a> This notebook creates an intermediate delta table with all xml articles. This provides a performant source, much faster than reading individual raw xml files to apply our chunking strategy.
# MAGIC  * <a href="$./04_Chunk_Articles_Content" target="_blank">04_Chunk_Articles_Content</a> This notebook applies a chunking strategy on articles to populate the source delta table for our vector serving index.
# MAGIC  * <a href="$./05_Sync_VectorSearch_Index" target="_blank">05_Sync_VectorSearch_Index</a> This notebook will sync our delta source table to our vector index.
# MAGIC
# MAGIC **NOTE**: Unlike Clusters which will have different configurations by cloud provider due to cloud providers having different cloud component offering, Tasks have a consistent UI across all cloud providers. We'll create a `tasks` variable with each of our required notebooks.

# COMMAND ----------

from databricks.sdk.service.jobs import Task, NotebookTask, TaskDependency
from databricks.sdk.service.compute import  Library, PythonPyPiLibrary

import os

workflow_dir = os.path.dirname(dbutils.entry_point.getDbutils().notebook().getContext().notebookPath().getOrElse(None))

tasks = [Task(task_key='01_Metadata_Sync',
              job_cluster_key="BioMed_VS_Sync",
              libraries=[Library(pypi=PythonPyPiLibrary(package='pyyaml==6.0')),
                         Library(pypi=PythonPyPiLibrary(package='mlflow==2.15.1')),
                         Library(pypi=PythonPyPiLibrary(package='databricks_vectorsearch==0.39'))],
              notebook_task=NotebookTask(notebook_path=os.path.join(workflow_dir, "01_Metadata_Sync"))),
         Task(task_key='02_Articles_Ingest',
              job_cluster_key="BioMed_VS_Sync",
              libraries=[Library(pypi=PythonPyPiLibrary(package='pyyaml==6.0')),
                         Library(pypi=PythonPyPiLibrary(package='mlflow==2.15.1')),
                         Library(pypi=PythonPyPiLibrary(package='databricks_vectorsearch==0.39'))],
              notebook_task=NotebookTask(notebook_path=os.path.join(workflow_dir, "02_Articles_Ingest")),
              depends_on=[TaskDependency(task_key='01_Metadata_Sync')]),
         Task(task_key='03_Curate_Articles',
              job_cluster_key="BioMed_VS_Sync",
              libraries=[Library(pypi=PythonPyPiLibrary(package='pyyaml==6.0')),
                         Library(pypi=PythonPyPiLibrary(package='mlflow==2.15.1')),
                         Library(pypi=PythonPyPiLibrary(package='databricks_vectorsearch==0.39'))],
              notebook_task=NotebookTask(notebook_path=os.path.join(workflow_dir, "03_Curate_Articles")),
              depends_on=[TaskDependency(task_key='02_Articles_Ingest')]),
         Task(task_key='04_Chunk_Articles_Content',
              job_cluster_key="BioMed_VS_Sync",
              libraries=[Library(pypi=PythonPyPiLibrary(package='pyyaml==6.0')),
                         Library(pypi=PythonPyPiLibrary(package='mlflow==2.15.1')),
                         Library(pypi=PythonPyPiLibrary(package='databricks_vectorsearch==0.39')),
                         Library(pypi=PythonPyPiLibrary(package='unstructured==0.15.1')),
                         Library(pypi=PythonPyPiLibrary(package='html2text==2024.2.26')),
                         Library(pypi=PythonPyPiLibrary(package='nltk==3.7'))],
              notebook_task=NotebookTask(notebook_path=os.path.join(workflow_dir, "04_Chunk_Articles_Content")),
              depends_on=[TaskDependency(task_key='03_Curate_Articles')]),
         Task(task_key='05_Sync_VectorSearch_Index',
              job_cluster_key="BioMed_VS_Sync",
              libraries=[Library(pypi=PythonPyPiLibrary(package='pyyaml==6.0')),
                         Library(pypi=PythonPyPiLibrary(package='mlflow==2.15.1')),
                         Library(pypi=PythonPyPiLibrary(package='databricks_vectorsearch==0.39')),
                         Library(pypi=PythonPyPiLibrary(package='langchain_community==0.2.7'))],
              notebook_task=NotebookTask(notebook_path=os.path.join(workflow_dir, "05_Sync_VectorSearch_Index")),
              depends_on=[TaskDependency(task_key='04_Chunk_Articles_Content')])]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create Scheduled Job
# MAGIC
# MAGIC We'll now proceed with getting or creating our job definition through the method, `get_or_create_job`.
# MAGIC
# MAGIC **NOTE**: This job will be created using the credentials of the user running this notebook.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import Job, Format, CronSchedule, PauseStatus, JobEditMode, JobRunAs, QueueSettings

def get_or_create_job(job_name: str) -> Job:
    client = WorkspaceClient()
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
    try:
        job:Job = next(client.jobs.list(expand_tasks=True, name=job_name))
        displayHTML(f'<a href=jobs/{job.job_id}>{job_name}</a> already exists.')
    except StopIteration:
        job_response = client.jobs.create(name=job_name,
                                          description="BioMed GenAI Workflow Synchronization",
                                          job_clusters=[get_job_cluster(),],
                                          format=Format(value="MULTI_TASK"),
                                          tasks=tasks,
                                          schedule=CronSchedule(quartz_cron_expression='1 0 5 ? * Sun',
                                                                timezone_id='America/New_York',
                                                                pause_status=PauseStatus("UNPAUSED")),                                  
                                          edit_mode=JobEditMode(value="EDITABLE"),
                                          run_as=JobRunAs(user_name=user),
                                          queue=QueueSettings(enabled=True),
                                          max_concurrent_runs=1,
                                          timeout_seconds=0)
        job:Job = client.jobs.get(job_response.job_id)
        displayHTML(f'<a href=jobs/{job.job_id}>{job_name}</a> has been created.')
    return job

job:Job = get_or_create_job(job_name="BioMed_VS_Sync")
