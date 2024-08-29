# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Designate Champion
# MAGIC
# MAGIC This is the stage where we compare the best candidate model to the existing production model. In the case that this is the first production model, the best candidate model wins. 
# MAGIC
# MAGIC **Note**: This selection doesn't need to be the exact same scoring as was done for best model select. It should include some threshold for how much better a candidate model needs to be to replace a champion model.
# MAGIC
# MAGIC **Note**: It is possible to apply a current release_version evaluation dataset to a previous release version to get more current metrics. Make sure when this is done to do in the share experiment and not in a notebook experiment. Since there is a nested experiment run convention in this solution accelerator, we can easily run evaluations this way.
# MAGIC
# MAGIC **Note**: Regardless if the best candidate model becomes the champion, we will still create a new version of our agent. This convention means that our agent version will be consistant with our RELEASE verions. Meaning each RELEASE will have a single best candidate as a version.
# MAGIC
# MAGIC **TODO**: Write the update funciton for champion - it must be able to handle first pass and follow-on passes.
