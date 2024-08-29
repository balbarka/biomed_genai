# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Agent Models - Candidate Model Development
# MAGIC
# MAGIC Unlike the previous notebook-tasks this notebook isn't doing the task so much as it is confirming a task is done. Specifically this notebook is going to verify that the following inner-loop notebook tasks are complete:
# MAGIC
# MAGIC   * **<a href="$./agent_model/02_01_Candidate_Runs" target="_blank">02_01_Candidate_Runs</a>**: This will make sure that the candidate experiment model runs that are listed in the candidate experiments are executed.
# MAGIC   * **<a href="$./agent_model/02_02_Score_Register" target="_blank">02_02_Score_Register</a>** Score Candidates and Register the Best as new model version.
# MAGIC   * **<a href="$./agent_models/02_03_Review_App_Feedback" target="_blank">02_03_Review_App_Feedback</a>**: Inspect feedback to detemine potential updates to parameters, scoring, or champion selection.
# MAGIC   * **<a href="$./agent_models/02_04_Designate_Champion" target="_blank">02_04_Designate_Champion</a>**: Evaluate if the Best Candidate model should be the new champion.
# MAGIC
# MAGIC **NOTE**: Be aware that agent model comparison happens twice! The first is to select the best candidate model. The second is to determine if the best candidate is good enough to unseat the existing champion. Since, version iterations do come with complexity costs, we want to be sure that assigning a new champion (production model) only happens when opportunity cost exceeds the complexity cost.
# MAGIC
# MAGIC The above inner-loop tasks are consistant with our governance diagram. However, now that we'll be interacting with our agent experiment, we can also show a graphical representation of the entities that are produced from out notebooks tasks known as the agent deploy graphic.

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_qa_chat $SHOW_GOVERNANCE=true $SHOW_AGENT_DEPLOY=true

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Validate Candidate Runs Complete
# MAGIC
# MAGIC We will know that candidate runs are complete when we verify that all have evaluation runs.
# MAGIC
# MAGIC **NOTE**: It is possible and likely more metrics could be considered and this method is updated.

# COMMAND ----------

candidate_models = bc_qa_chat.candidate_models
models = bc_qa_chat.experiment.models

def candidate_run_complete(candidate_models, models) -> bool:
    return set(candidate_models).issubset(set([m.model_name for m in models]))

candidate_run_success = candidate_run_complete(candidate_models, models)
candidate_run_success

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Validate Score Register Complete
# MAGIC
# MAGIC When score and register is complete, the following will be true:
# MAGIC  - We will have an evaluation run for each candidate model
# MAGIC  - We will have a registered model version that matches our release_version
# MAGIC

# COMMAND ----------

def score_register_complete(candidate_models, models):
    def model_eval_exists(model_name):
        return [m for m in models if m.model_name == model_name] != []
    return all([model_eval_exists(m) for m in candidate_models])

score_register_success = score_register_complete(candidate_models, models)
score_register_success

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Validate Review App Feedback
# MAGIC
# MAGIC Unlike other notebook-tasks, incorperating feedback won't have an inspectable item. Thus, for will only verify that the Review App is running and that feedback could have been provided.
# MAGIC
# MAGIC **Note**: There are ways to set a standard that a set number of review entries are created and have populated feedback fields, but that level of rigor isn't applied in this solution accelerator.

# COMMAND ----------

import requests

review_url = 'https://adb-830292400663869.9.azuredatabricks.net/ml/review/biomed_genai.agents.bc_qa_chat/5'

def review_app_complete(url):
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

# Check if the review_url is working
review_app_success = review_app_complete(review_url)
review_app_success

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Designate Champion Complete
# MAGIC
# MAGIC There are two conventions we can inspect to confirm that our candidate best model has been registered and we have a designated champion:
# MAGIC  - Is there a single condidate experiment model run tagged as `best` for the given release_version?
# MAGIC  - Is the candidate experiment model run now a registered model?
# MAGIC  - Is there a single registered model tagged as 'prod'?
# MAGIC
# MAGIC **TODO**: Write the complete method for these three criteria. In the interim, these conditions have been inspected manually and assigned a `True` value.

# COMMAND ----------

designate_champion_complete = True
designate_champion_success = designate_champion_complete

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # All Complete & Commit
# MAGIC
# MAGIC Now we are at the point where we have all assets in the proper state. All code for these notebooks will be in a state that the model can be replicated. Thus, if the all complete checks are successful, we should commit all code to this point to our release branch.

# COMMAND ----------

all_agent_model_success = all([candidate_run_success,
                               score_register_success,
                               review_app_success,
                               designate_champion_success])
all_agent_model_success
