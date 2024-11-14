import json
from typing import List
from databricks.sdk.runtime import dbutils

#Return the current cluster id to use to read the dataset and send it to the fine tuning cluster. See https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html#cluster-id
def get_current_cluster_id():
  return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']


def get_latest_model_version(model_name):
    from mlflow.tracking import MlflowClient
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


def write_jsonl_by_line(responses: List, outfile: str, no_none=True) -> None:
    # Write to jsonl line by line
    with open(outfile, 'a+') as out:
        for r in responses:
            if r:
                if no_none:
                    if len(set(r.values()).intersection({None,'None','null'}))==0:
                        jout = json.dumps(r) + '\n'
                else:
                    jout = json.dumps(r) + '\n'
                out.write(jout)


# For completion model
def make_completion_prompt(context, question, system_prompt):
    system_prompt = "You are a medical expert answering questions about biomedical research. Please answer the question below based on only the provided context. If you do not know, return nothing."
    return f"""{system_prompt}
### Question: {question}
### Context: {context}
### Answer:
"""


# For chat model
def make_chat_prompt(context: str, question: str, answer: str, mistral=False) -> List[Dict[str, str]]:
    system_prompt = f"""You are a medical expert answering questions about biomedical research. Please answer the question below based on only the provided context. If you do not know, return nothing."""
    user_input = f"""{question}. Answer this question using only this context: 
{context}."""
    if mistral:
        #Mistral doesn't support system prompt
        return [
            {"role": "user", "content": f"{system_prompt} \n{user_input}"},
            {"role": "assistant", "content": answer}]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer}]


@pandas_udf("array<struct<role:string, content:string>>")
def make_chat_udf(content: pd.Series, question: pd.Series, answer: pd.Series) -> pd.Series:
    return pd.Series([make_chat_prompt(c, q, a) for c, q, a in zip(content, question, answer)])


