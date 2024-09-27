# Databricks notebook source
from mlflow.types.llm import ChatResponse, ChatChoice, ChatMessage, ChatChoiceLogProbs, TokenUsageStats

# COMMAND ----------

from mlflow.types.llm import ChatResponse, ChatChoice, ChatMessage, ChatChoiceLogProbs, TokenUsageStats

chatMessage = ChatMessage(role="user",
                          content="test_message")

chatChoiceLogProbs = ChatChoiceLogProbs(content=None)

chatChoice = ChatChoice(index=0,
                        message=chatMessage,
                        finish_reason="stop",
                        logprobs=chatChoiceLogProbs)

tokenUsageStats = TokenUsageStats(prompt_tokens=10,
                                  completion_tokens=10,
                                  total_tokens=10)

chatResponse = ChatResponse(choices=[chatChoice,],
                            usage=tokenUsageStats,
                            id=None,
                            model=None,
                            object="chat.completion")  

# COMMAND ----------

ChatResponse(**chatResponse.to_dict())

# COMMAND ----------

from datetime import datetime

# Convert the string date to a datetime object
created_date = datetime.strptime("2022-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")

# COMMAND ----------

ChatResponse(**{'choices': [{'index': 0,
                             'message': {'role': 'user', 'content': 'test_message'},
                             'finish_reason': 'stop',
                             'logprobs': {}}], 
                'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 10},
                'object': 'chat.completion',
                 'created': 1724206409})

# COMMAND ----------

# MAGIC %sh
# MAGIC curl \
# MAGIC -u token:$DATABRICKS_TOKEN \
# MAGIC -X POST \
# MAGIC -H "Content-Type: application/json" \
# MAGIC -d '{
# MAGIC   "messages": [
# MAGIC     {
# MAGIC       "role": "system",
# MAGIC       "content": "You are a helpful assistant."
# MAGIC     },
# MAGIC     {
# MAGIC       "role": "user",
# MAGIC       "content": " What is a mixture of experts model?"
# MAGIC     }
# MAGIC   ]
# MAGIC }' \
# MAGIC https://<workspace_host>.databricks.com/serving-endpoints/databricks-dbrx-instruct/invocations \

# COMMAND ----------

import requests
import os

os.environ['OPENAI_API_KEY'] = dbutils.secrets.get('biomed_genai', 'bc_qa_chat_api_key')
os.environ['OPENAI_HOST'] = dbutils.secrets.get('biomed_genai', 'host')

# Define the endpoint URL
url = os.environ['OPENAI_HOST']

# Define the headers
headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}

# Define the payload
payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is a mixture of experts model?"
        }
    ]
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Print the response
print(response.json())

# COMMAND ----------

import requests
import os
import json

# Define the endpoint URL
url = f"{url}/serving-endpoints/databricks-dbrx-instruct/invocations"

# Define the headers
headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}

payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is a mixture of experts model?"
        }
    ]
}

# Make the POST request
response = requests.post(url, headers=headers, json=payload)

# Print the response
print(response.json())

# COMMAND ----------


