# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Agent Configurations
# MAGIC
# MAGIC These notebooks are numbered to take us through the agent development lifecycle. The first step in the agent development life cycle is to **Gather Requirements**.
# MAGIC ![Agent LLMOps](https://docs.databricks.com/en/_images/llm-ops-simplified.png)
# MAGIC
# MAGIC **Gather Requirements** is typically done outside of our application code base, but it indirectly relates to a notebook-task in our LLM Ops governance notebooks, this notebook about configurations. Additionally, many of the other steps in the LLM Ops governance tasks will be like this. So while we'll keep the above graphic in mind for what business process tasks we are executing and how they map to Databricks features, we'll actually be following a code implementation of this as notebooks-tasks within our project code. Here is what our entire LLM Ops Governance Life Cycle looks like for an example agent application, `bc_qa_chat`:

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_qa_chat $SHOW_GOVERNANCE=true $SHOW_AGENT_DEPLOY=true

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Agent `ba_qa_chat` Configurations
# MAGIC
# MAGIC ### Configuration Hierarchy:
# MAGIC
# MAGIC When we developing agent applications there is an impressive five levels of configurations that we need to concern ourselves with:
# MAGIC  * **Project Configurations** - Project level configurations like catalog shared project asset directories.
# MAGIC
# MAGIC    * **Agent Shared Configurations** - Projects can and will have many agents assets. These shared agent configs will provide the root directories in UC and Workspace for where to store these assets.
# MAGIC
# MAGIC      * **Agent Specific Configurations** - Individual Agents will need their own config space. However, since Agents are continually improved, that means that we will have some Agent Specific Configurations by Agent Version. Since these are not static, we really don't want to persist them as static configurations. Fortunately, Databricks already provides a solution for managing these agent specific configurations as [UC Registered Models](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html).
# MAGIC
# MAGIC        * **Model Shared Configurations** - Think configurations that are the same across all candidate models classes. If we are using [MLFlow](https://docs.databricks.com/en/mlflow/index.html), that means for each candidate model class we'll have an experiment path. However, when we adopt the convention that every candidate model class uses a shared experiment path, we can set a single experiment path in our agent specific configurations and remove this configuration level.
# MAGIC
# MAGIC          * **Model Specific Configurations** - Model Specific configurations will include all those artifacts and parameters that go into defining a model class instance. This is the primary feature of [MLFlow](https://docs.databricks.com/en/mlflow/index.html) and so therefore we will manage all of our model specific configurations in MLFlow experiment runs.
# MAGIC          
# MAGIC ### Configuration Management:
# MAGIC
# MAGIC Thus, if we consolidate *Project Configurations*, *Agent Shared Configurations*, and *Agent Shared Configurations* into a single [YAML](https://en.wikipedia.org/wiki/YAML) file while adopting the Databricks provided features of [UC Registered Models](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html) and [MLFlow](https://docs.databricks.com/en/mlflow/index.html), our configuration hierarchy simplies into the following:
# MAGIC
# MAGIC  * **Static Configurations**: [YAML](https://en.wikipedia.org/wiki/YAML)
# MAGIC    * **Agent-Version Configurations**: [UC Registered Models](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)
# MAGIC      * **Model-Verion Configurations**: [MLFlow](https://docs.databricks.com/en/mlflow/index.html)
# MAGIC
# MAGIC Thus to simplify configuration retrieval, we'll maintain a single setup notebook, <a href="$./_setup/setup_bc_qa_chat" target="_blank">setup_bc_qa_chat</a> that will retrieve the relevant subset of configurations from <a href="$../../_config/config_biomed_genai.yaml" target="_blank">config_biomed_genai.yaml</a>. From there, we'll use a dataclass, [Agent_bc_qa_chat](../../../python/biomed_genai/agent/bc_qa_chat/agent_bc_qa_chat.py), that is integrates with [UC Registered Models](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html) and [MLFlow](https://docs.databricks.com/en/mlflow/index.html) to provide a single entry configuration dataclass ready for use in any `bc_qa_bot` notebook. This was already done above at the same time we called our LLM Ops Governance graphic.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We'll explore in follow-on notebooks how the `bc_qa_chat` dataclass is used.
# MAGIC
# MAGIC # Agent Conventions
# MAGIC
# MAGIC Above we introduced conventions to simplify our some of our required configurations. For ease of reference we'll bulletize them here:
# MAGIC
# MAGIC   * **Consolidated YAML File** - Our YAML file is actually hierarchtical which is how we are able to maintain a single file for everything static configuration in our project.
# MAGIC   * **Single Experiment Path** - We'll use a single experiment path for all models, all evaluations in `bc_qa_bot`. This will also allow us to make the most use out of evaluation features built into MLFlow for model and evaluation comparisons. **NOTE**: We are not using notebook experiment paths.
# MAGIC   * **Single Model Serving Endpoint** - Not covered above, we will have a convention that we will only utilize a single model serving endpoint for our agent application.
# MAGIC   * **Secrets** - Also not covered above, but enterprise agent applications have a dependency on secrets. We'll cover this in more detail in the following section.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # Agent Secret Convention
# MAGIC
# MAGIC # !!! **--->     NEVER PUT A SECRET IN A CONFIG FILE     <---** !!!
# MAGIC
# MAGIC # !!! **--->     NEVER PUT A SECRET IN INTO MLFLOW      <---** !!!
# MAGIC
# MAGIC # !!! **--->     NEVER WRITE A SECRET IN A NOTEBOOK CELL     <---** !!!
# MAGIC
# MAGIC When a secret is needed for an agent, the right thing to do is to write whatever configurations are needed for the retrieval into configurations. Yes, secret scope and secret key are appropriate configurations. No, secret values, tokens and other sensitive items are not appropriate configurations.
# MAGIC
# MAGIC The common convention is to write the secret scope and in the following format: `{{secrets/<scope-name>/<secret-name>}}`
# MAGIC
# MAGIC **NOTE**: It is both common and viable for customers to use thier own enterprice KMS or cloud service. The `biomed_genai` project uses only [Databricks Secrets](https://docs.databricks.com/en/security/secrets/secrets.html) because of it's integration across databricks features.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC # Start a RELEASE Branch
# MAGIC
# MAGIC If you are updating your Configs, you have started a new agent release. Once the configs are complete, create a release branch and commit the following files in your new release branch:
# MAGIC  - **00_CONFIG_bc_qa_chat_config** - This notebook
# MAGIC  - **config_biomed_genai.yaml** - Configurations for 
# MAGIC
# MAGIC  **NOTE**: In an agent release the one file you will get merge conflicts on is config_biomed_genai.yaml, thus be sure to coordinate with other teams if you are both making iterative improvments on different componets at the same time. If implemented correctly, this should be the single file vulnerable to a merge conflict.
# MAGIC
# MAGIC Congrats, you've completed your first notebook-task of a new agent development cycle.
