import pyspark
import delta
import mlflow
from dataclasses import dataclass
from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from mlflow.exceptions import RestException
from functools import cached_property
import re
import os
import json

from pandas import Series, DataFrame as Pdf
from mlflow.entities.experiment import Experiment
#from mlflow.entities import Run, RunInfo, RunInputs
from mlflow.entities.run import Run
from mlflow.entities.run_data import RunData
from mlflow.tracking import MlflowClient
from mlflow.models.evaluation.base import EvaluationResult

from databricks.vector_search.client import VectorSearchClient


@dataclass
class WS_JSON_Entity:
    # This is a base class for WS entities that can be instantiated from JSON file config
    ws_name: str
    json_file: str
    json_folder: str

    @cached_property
    def vs_client(self) -> VectorSearchClient:
        # TODO: Enable Authentication from secrets (instead of from notebook credentials)
        return VectorSearchClient()

    @cached_property
    def json_path(self) -> str:
        return "/".join([self.json_folder, self.json_file])

    @cached_property
    def json_relative_url(self) -> str:
        return self.json_path.replace("/Workspace/", "#workspace/")

    @cached_property
    def json_dict(self) -> dict:
        with open(self.json_path, 'r') as file:
            return json.load(file)


@dataclass
class WS_Endpoint(WS_JSON_Entity):

    @cached_property
    def ws_relative_url(self) -> str:
        return '/compute/vector-search/' + self.name

    @property
    def endpoint(self) -> dict:
        return get_or_create_endpoint(**self.json_dict)

    @cached_property
    def name(self) -> str:
        return self.endpoint.get('name')

def get_or_create_endpoint(name: str,
                           endpoint_type='STANDARD'):
    kwargs = locals()
    vsc = VectorSearchClient(disable_notice=True)
    try:
        return vsc.get_endpoint(name)
    except Exception as e:
        return vsc.create_endpoint_and_wait(**kwargs)

def get_or_create_index(endpoint_name: str,
                        index_name: str,
                        primary_key: str,
                        source_table_name: str,
                        pipeline_type: str,
                        embedding_dimension=None,
                        embedding_vector_column=None,
                        embedding_source_column=None,
                        embedding_model_endpoint_name=None,
                        sync_computed_embeddings=False):
    kwargs = locals()
    vsc = VectorSearchClient(disable_notice=True)
    try:
        return vsc.get_index(endpoint_name, index_name)
    except Exception as e:
        return vsc.create_delta_sync_index_and_wait(**kwargs)
    

@dataclass
class WS_Index(WS_JSON_Entity):

    @cached_property
    def ws_relative_url(self) -> str:
        return f"/explore/data/{'/'.join(self.ws_name.split('.'))}/"

    @property
    def index(self) -> VectorSearchIndex:
        kwargs = self.json_dict
        # Convention is that source_table_name + '_vs_index' = index_name
        kwargs["source_table_name"] = self.ws_name[:9]
        kwargs["index_name"] = self.ws_name
        return get_or_create_index(**kwargs)

    @cached_property
    def name(self) -> str:
        return self.index.name


@dataclass
class UC_SQL_Entity:
    # This is a base class for UC entities that can be instantiated with SQL DDL: Catalog, Schema, Table, Volume
    uc_name: str
    sql_file: str
    sql_folder: str

    @cached_property
    def catalog(self) -> str:
        return self.uc_name.split('.')[0]
    
    @cached_property
    def schema(self) -> str:
        names = self.uc_name.split('.')
        if len(names) < 2:
            return ""
        else:
            return names[1]   

    @cached_property
    def spark(self) -> SparkSession:
        return SparkSession.builder.getOrCreate()

    @cached_property
    def sql_path(self) -> str:
        return "/".join([self.sql_folder, self.sql_file])

    @cached_property
    def uc_relative_url(self) -> str:
        if hasattr(self, "_path_value"):
            # _path_value is only used in volumes and is a means to create subdirectories in a volume
            return ('/explore/data/volumes/' + '/'.join(self.name.split('.'))).replace('`', '')
        else:
            return ('/explore/data/' + '/'.join(self.name.split('.'))).replace('`', '')

    @cached_property
    def sql_relative_url(self) -> str:
        return self.sql_path.replace("/Workspace/", "#workspace/")

    @cached_property
    def create_sql(self) -> str:
        with open(self.sql_path, 'r') as f:
            return f.read()

    @cached_property
    def name(self) -> str:
        # As convention, ddl sql executed first time name is called
        # As convention, use IF NOT EXISTS in ddl
        sql = self.create_sql
        kwargs = {k: getattr(self, k) for k in
                  set(self.__dir__()).intersection(set(re.findall(r"{(.*?)}", sql)))}
        self.spark.sql(sql.format(**kwargs))
        if hasattr(self, "_path_value"):
            # _path_value is only used in volumes and is a means to create subdirectories in a volume
            os.makedirs("/".join((['', 'Volumes',] + self.uc_name.split('.') + [self._path_value,]))
                           .replace('`', ''), exist_ok=True)
        return self.uc_name


@dataclass
class UC_Table(UC_SQL_Entity):
    # UC Table Class
    
    @cached_property
    def table(self) -> str:
        return self.uc_name.split('.')[2]

    @property
    def df(self) -> pyspark.sql.DataFrame:
        return self.spark.table(self.name)

    @property
    def dt(self) -> delta.tables.DeltaTable:
        return delta.tables.DeltaTable.forName(self.spark, self.name)

    @property
    def dt_version(self) -> int:
        return int(self.dt.history().orderBy("version", ascending=False).first().version)

@dataclass
class UC_Dataset(UC_Table):
    # A Dataset Class, same as a UC_Table, but with convention properties
    release_version: int

    @property
    def ds(self):
        return mlflow.data.load_delta(table_name=self.name,
                                      name=self.ds_release_version_name,
                                      version=self.release_version)

    @property
    def ds_name(self):
        return self.name.split('.')[-1]

    @property
    def ds_release_version_name(self):
        return f'{self.ds_name}-{self.release_version:03}' 

@dataclass
class WS_ExperimentGenAI_Eval():
    def __init__(self, eval_run: Run):
        self.experiment_id = eval_run.info.experiment_id
        self.model_run_id = eval_run.data.tags.get('mlflow.parentRunId')
        self.eval_run_id = eval_run.info.run_id
        self.eval_run_name = eval_run.info.run_name
        self.eval_ds_name = eval_run.inputs.dataset_inputs[0].dataset.name
    
    @cached_property
    def eval_run_version(self) -> int:
        return int(self.eval_run_name.split('-')[-1])

    @cached_property
    def ws_relative_url(self):
        return f"/ml/experiments/{self.experiment_id}?searchFilter=dataset.name+%3D+%22{self.eval_ds_name}%22"

class WS_ExperimentGenAI_Model():
    def __init__(self, release_version: int,
                       model_run: Run,
                       eval_run: Run):
        self.experiment_id = model_run.info.experiment_id
        self.model_run_id = model_run.info.run_id
        self.model_run_name = model_run.info.run_name
        self.model_run_info = model_run.info
        self.eval = WS_ExperimentGenAI_Eval(eval_run) if eval_run is not None else None
    
    @cached_property
    def model_name(self):
        return '-'.join(self.model_run_name.split('-')[0:-1])

    @cached_property
    def model_version(self):
        return int(self.model_run_name.split('-')[-1])

    @cached_property
    def ws_relative_url(self):
        return f"/ml/experiments/{self.experiment_id}?searchFilter=attributes.run_name%3D%22{self.model_run_name}%22"

@dataclass
class WS_GenAI_Agent_Experiment():
    """This experiment run has a convention where the model is saved to the parent run
    and the eval dataset is saved to the child run. This convention is better suited for 
    GenAI which has a developer pattern of iterative evaluations as eval dataset evolve
    with newer delta table versions."""
    release_version: int
    agent_name: str
    experiments_workspace_folder: str
    eval_ds: UC_Dataset = None
    default_model_name: str = None

    @cached_property
    def experiment_name(self):
        return f'{self.experiments_workspace_folder}/{self.agent_name}'

    @cached_property
    def experiment_id(self):
        # We run this to make sure that the experiment workspace folder exists even if empty
        os.makedirs(f'/Workspace/{self.experiments_workspace_folder}', exist_ok=True)
        return (mlflow.get_experiment_by_name(self.experiment_name) or
                mlflow.get_experiment(mlflow.create_experiment(self.experiment_name))).experiment_id

    @property
    def models(self) -> [WS_ExperimentGenAI_Model]:
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id,],
                                  filter_string=f"tags.release_version = '{self.release_version}'",
                                  output_format='list')
        if len(runs)>0:
                model_runs = [r for r in runs if 'mlflow.parentRunId' not in r.data.tags]
                eval_runs = [r for r in runs if 'mlflow.parentRunId' in r.data.tags]
                def get_eval_run(m):
                    eval_run = [e for e in eval_runs if e.data.tags['mlflow.parentRunId']==m.info.run_id]
                    return None if len(eval_run)==0 else eval_run[0]
                return [WS_ExperimentGenAI_Model(self.release_version, m, get_eval_run(m)) for m in model_runs]
        else:
            return []

    def model(self, model_run_name: str=None) -> WS_ExperimentGenAI_Model:
        model_run_name = model_run_name or self.model_run_name
        return next((m for m in self.models if m.model_run_name == model_run_name), None)

    def create_model_run(self, model_name: str=None, overwrite: bool=False, nb_experiment: bool=False):
        """When you run create model run, you are creating a model with a set experiment and model name.
        If model overwrite is True, it will overwrite any existing model. If overwrite is false and a model already 
        exists an exception will be thrown. If doing local iterations, you can set nb_eperiment to True. The parameter 
        overwrite has no effect if nb_experiment is True."""
        if nb_experiment:
            return mlflow.start_run(experiment_id=self.experiment_id)
        else:
            model = self.model(model_name)
            if model:
                if overwrite:
                    # Return run to overwrite existing, current model
                    run = mlflow.start_run(run_id=model.model_run_id)
                else:
                    if self.model():
                        # This is to mitigate unintentional overwrites
                        raise ValueError("The model run {model.model_name} already exists.")
                    else:
                        # Return potential new or overwrite
                        run = mlflow.start_run(experiment_id=self.experiment_id,
                                               run_name=self.model_run_name)
            else:
                # Return new, first model run
                run =  mlflow.start_run(experiment_id=self.experiment_id,
                                        run_name=self.model_run_name)
            mlflow.set_tag("release_version", self.release_version)
            return run

    def evaluate(self, model_name: str=None) -> EvaluationResult:
        # There is no overwrite funcitonality for evaluate, if the model is replaced, must manually delete the eval run
        # Overwrite is convenience to iterate fast when encountering failed experiments and don't want the runs clutter
        model = self.model(model_name)
        if model:
            if model.eval:
                print("An evaluation run {self.eval_ds.ds_release_version_name} already exists. Skipping evaluate.")
            else:
                with mlflow.start_run(experiment_id=self.experiment_id,
                                      run_name=model.model_run_name,
                                      nested=True,
                                      parent_run_id=model.model_run_id) as eval_run:
                    eval_rslt =  mlflow.evaluate(data=self.eval_ds.ds,
                                                 model=f"runs:/{model.model_run_id}/model",
                                                 model_type="databricks-agent")
                    mlflow.set_tag("release_version", self.release_version)
                    return eval_rslt
        else:
            raise ValueError("You must first create model {model.model_name} before you run evaluate.")

    @cached_property
    def ws_relative_url(self):
        return f"/ml/experiments/{self.experiment_id}?searchFilter=&lifecycleFilter=Active"

    @cached_property
    def ws_exp_release_relative_url(self):
        # Returns a filter url for the current model version
        return f"/ml/experiments/{self.experiment_id}?searchFilter=tags.release_version+%3D{self.release_version}"

    @cached_property
    def model_run_name(self): 
        return f'{self.default_model_name or "MODEL"}-{self.release_version:03}'

@dataclass
class UC_Registered_Model():
    uc_name: str

    @cached_property
    def mlflow_client(self) -> MlflowClient:
        return MlflowClient()

    @property
    def latest_version(self):
        model_versions = self.mlflow_client.search_model_versions(f"name='{self.name}'")
        if model_versions:
            latest_version = max([int(mv.version) for mv in model_versions])
            return latest_version
        else:
            return 0

    @cached_property
    def name(self) -> str:
        mlflow.set_registry_uri("databricks-uc")
        try:
            self.mlflow_client.get_registered_model(self.uc_name)
        except RestException:
            self.mlflow_client.create_registered_model(self.uc_name)
        return self.uc_name

    @cached_property
    def catalog(self):
        return self.name.split('.')[0]
    
    @cached_property
    def schema(self):
        return self.name.split('.')[1]

    @cached_property
    def model_name(self):
        return self.name.split('.')[-1]    

    @cached_property
    def uc_relative_url(self):
        return f'/explore/data/models/{self.catalog}/{self.schema}/{self.model_name}'


@dataclass
class UC_Volume(UC_SQL_Entity):
    # Our volume class which uses our base class, every volume in our application should get this class assigned
    _path_value: str = ""

    @cached_property
    def volume_root(self):
        vol_path_list = ['', 'Volumes',]
        vol_path_list += self.name.split('.')
        return '/'.join(vol_path_list).replace('`', '')
    
    @cached_property
    def path(self):
        # Returns the complete path which is the ultimate value we will use in our application workflow
        if self._path_value == "":
            return self.volume_root
        else:
            return self.volume_root + '/' + self._path_value


    @cached_property
    def spark(self) -> SparkSession:
        return SparkSession.builder.getOrCreate()
