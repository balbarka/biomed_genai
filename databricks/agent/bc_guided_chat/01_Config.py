# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This will orient to the agent intent and explain how configs are managed.
# MAGIC
# MAGIC | Tool Concept     | Existing UC Feature               |
# MAGIC | ---------------- | --------------------------------- |
# MAGIC | Tool calls       | FMAPI / AI Playground             |
# MAGIC | Tool definitions | UC UDFs: SQL, Python, [Remote]    |
# MAGIC | Tool metadata    | UC function metadata              |
# MAGIC | Tool governance  | UC / Serverless egress control    |
# MAGIC | Tool compute     | DBSQL / Safe Spark / [BrickStore] |
# MAGIC | Tool sharing     | [Delta Sharing]                   |
# MAGIC | Tool discovery   | Global search                     |
# MAGIC
# MAGIC **REMOVE BEFORE COMMIT REF**: 
# MAGIC  * [Mosaic Tools](https://docs.google.com/presentation/d/11_48jYFH7UzW1LW_pCmJrLc9yPxF7uSZ7ZNCNIEmcyA/edit#slide=id.g2d1778726a2_0_67)
# MAGIC  * [Catalyst Guidelines](https://docs.google.com/presentation/d/1VjbRUCeO7buwYuL6oRXAbQ2O32NUcZQ4kJv2iMjF9cM/edit#slide=id.g2f7415a0983_0_341)
# MAGIC  * [Catalyst JIRA (FEIP-133)](https://databricks.atlassian.net/browse/FEIP-133)
# MAGIC  * [Mosaic AI Keynote](https://docs.google.com/presentation/d/1JpdsyUWEvhklKygHzUWSd8Q259pv6MxNhuxjqhgI7sA/edit#slide=id.p24)
# MAGIC  * [balbarka/biomed_genai](https://github.com/balbarka/biomed_genai)
# MAGIC  * [AI Cookbook Tool Ref Code #1](https://github.com/databricks/genai-cookbook/blob/ae2f30fb2345e343c84141ca0775dc34220f4b9f/agent_app_sample_code/agents/function_calling_agent_w_retriever_tool.py#L352)
# MAGIC  * [AI Cookbook Tool Ref Code #2](https://github.com/databricks/genai-cookbook/blob/ae2f30fb2345e343c84141ca0775dc34220f4b9f/agent_app_sample_code/03_agent_proof_of_concept.py#L55)
# MAGIC  * [AI Cookbook Tool Ref Code #3](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb)
# MAGIC  * [Mosaic AI Tools L200 Deck](https://docs.google.com/presentation/d/1oTZ9w-GsRhz0OfDZzB5lJve0H1CLMyg5ljYULVhJHKU/edit#slide=id.g276db250ef1_32_2492)
# MAGIC
# MAGIC  **TODO**: Update graphics with reference urls

# COMMAND ----------

# MAGIC %run ./_setup/setup_bc_guided_chat $SHOW_AGENT_MODEL=true $SHOW_NOTEBOOK_TASKS=false

# COMMAND ----------


