# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC It will feel odd to create a publsuh notebook, but this notebook serves a purpose of capturing a Release. The tasks below are:
# MAGIC
# MAGIC  * Write the active production verions of all agent entities.
# MAGIC  * Package code if necessary. This may seem like a curious task, but during our development we use code that was in a specific state. To be able to rerun this release as it was, it is possible to use a git commit, but it is also viable to use a packaged release of code.
# MAGIC  * Make a final commit and PR.
# MAGIC
# MAGIC **NOTE**: this notebook task overlaps with release management task that may normally not be done in notebook. Thus, this notebook may reference executable deploy code or run it - however it is likely that this would be done through an existing process and this notebook will only verify if a release has been made.
# MAGIC  

# COMMAND ----------


