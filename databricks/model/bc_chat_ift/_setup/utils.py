import json
from typing import List

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


def write_jsonl_by_line(responses: List, outfile: str) -> None:
    # Write to jsonl line by line
    with open(outfile, 'a+') as out:
        for r in responses:
            if r and len(set(r.values()).intersection({None,'None','null'}))==0:
                jout = json.dumps(r) + '\n'
                out.write(jout)

