from functools import cached_property
from dataclasses import dataclass
from biomed_genai.config import UC_SQL_Entity, UC_Volume
from huggingface_hub import login, snapshot_download
from databricks.sdk.runtime import dbutils
import mlflow

import os

@dataclass
class WS_GenAI_HF_Experiment_Run():
    """This class will be created from a WS_GenAI_HF_Experiment ensuring that all paths have already been instantiated"""
    experiment_id: str
    model_run_id: str
    model_name: str
    model_revision: str
    experiments_workspace_folder: str
    volume_hf_hub_root: str
    hf_login_token_secret: str
    local_hf_hub_cache: str = "/root/.cache/huggingface/hub"

    def __post_init__(self):
        # We'll explicitly set the local cache to the default local cache
        os.environ["HF_HOME"] = self.local_hf_hub_cache
        os.environ["TRANSFORMERS_CACHE"] = self.local_hf_hub_cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = self.local_hf_hub_cache
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "True"
        login(token=self._hf_login_token, add_to_git_credential=True)

    @property
    def _hf_login_token(self):
        secret_scope, secret_key = [s for s in self.hf_login_token_secret.translate(str.maketrans("", "", "{}")).split("/")][-2:]
        return dbutils.secrets.get(secret_scope, secret_key)

    @cached_property
    def rev_folder(self) -> str:    
        return '/'.join(['', '--'.join(['models',] +  self.model_name.split('/')), 'snapshots', self.model_revision])
    
    @cached_property
    def cache_src(self) -> str:
        return self.volume_hf_hub_artifact_path + self.rev_folder

    @cached_property
    def cache_tgt(self) -> str:
        return self.local_hf_hub_cache + self.rev_folder
    
    @cached_property
    def hf_url_commit(self) -> str:
        return f'{self.hf_url_model_card}/commit/{self.model_revision}'
    
    @cached_property
    def hf_url_model_card(self) -> str:
        return f'https://huggingface.co/{self.model_name}'

    def hf_cache_snapshot_download(self):
        """This will sync the volume snapshot for the given model_name, model_revision"""
        try: 
            snapshot_location = snapshot_download(repo_id=self.model_name,
                                                  revision=self.model_revision,
                                                  local_dir_use_symlinks=False,
                                                  cache_dir=self.local_hf_hub_cache,
                                                  token=self._hf_login_token,
                                                  resume_download=True,)
            print(f"Successfully completed up-to-date download of model {self.model_name}, revision {self.model_revision} " +
                  f"in {self.local_hf_hub_cache}")
        except Exception as e: 
            print(f"Error: {e}")

    def hf_cache_artifact_download(self):
        """This will overwrite local cache and copy experiment artifacts into local cache"""
        pass

    def hf_cache_volume_sync(self, model_name: str = None, model_revision: str = None, sync_volume = False):
        if sync_volume:
            self.sync_volume_snapshot(model_name, model_revision)
        self._sync_local_snapshot(model_name, model_revision)

    def start_log_model_run(self):
        """Hidden method to sync the local snapshot for the given model_name, model_revision"""
        return mlflow.start_run(run_id=model.model_run_id)


@dataclass
class WS_GenAI_HF_Experiment():
    """This experiment has a convention where every huggingface model name, model revision is an experiment run."""
    experiments_workspace_folder: str
    volume_hf_hub_root: str
    hf_login_token_secret: str
    local_hf_hub_cache: str = "/root/.cache/huggingface/hub"

    @cached_property
    def experiment_name(self):
        ws_folder = self.experiments_workspace_folder 
        os.makedirs("/Workspace" + ws_folder, exist_ok=True)
        return ws_folder + '/huggingface'

    @cached_property
    def experiment_id(self):
        # We'll use the experiment_id as the cached property that will check if we have an experiment already
        return (mlflow.get_experiment_by_name(self.experiment_name) or
                mlflow.get_experiment(mlflow.create_experiment(name=self.experiment_name,
                                                               artifact_location=f'dbfs:{self.volume_hf_hub_root}'))).experiment_id

    def get_or_create_hf_model_run(self, hf_model_name, hf_model_revision) -> WS_GenAI_HF_Experiment_Run:
        """This method will return an instantiated hf_model_run class"""
        def model_search_runs():
            return mlflow.search_runs(
                experiment_ids=[self.experiment_id,],
                filter_string=f"tags.hf_model_name = '{hf_model_name}' and tags.hf_model_revision = '{hf_model_revision}'",
                output_format='list')
        runs = model_search_runs()
        if len(runs)==0:
            # We are creating the run id for a given hf_model_name, hf_model_revision
            with mlflow.start_run(experiment_id=self.experiment_id,
                                  run_name=hf_model_name.split('/')[-1]) as run:
                mlflow.set_tag("hf_model_name", hf_model_name)
                mlflow.set_tag("hf_model_revision", hf_model_revision)
            runs = model_search_runs()
        model_run_id = runs[0].info.run_id
        return WS_GenAI_HF_Experiment_Run(experiment_id=self.experiment_id,
                                          model_run_id=model_run_id,
                                          model_name=hf_model_name,
                                          model_revision=hf_model_revision,
                                          experiments_workspace_folder=self.experiments_workspace_folder,
                                          volume_hf_hub_root=self.volume_hf_hub_root,
                                          hf_login_token_secret=self.hf_login_token_secret,
                                          local_hf_hub_cache=self.local_hf_hub_cache)

    @cached_property
    def ws_relative_url(self):
        return f"/ml/experiments/{self.experiment_id}?searchFilter=&lifecycleFilter=Active"


@dataclass
class Model_hf_cache():
    # This is the application class that includes, catlog, schema, volume, and experiment classes
    hf_login_token_secret: str
    config_ddl_folder: str
    experiments_workspace_folder: str
    volume_hf_hub_cache_catalog: str
    volume_hf_hub_cache_schema: str
    volume_hf_hub_cache_volume: str = "huggingface"
    volume_hf_hub_cache_dir: str = "hub"
    local_hf_hub_cache: str = "/root/.cache/huggingface/hub"

    def __post_init__(self):
        setattr(self, 'catalog', UC_SQL_Entity(uc_name=self.volume_hf_hub_cache_catalog,
                                               sql_file="CREATE_CATALOG_biomed_genai.sql",
                                               sql_folder=self.config_ddl_folder))
        setattr(self, 'schema', type('Schema', (object,), {}))
        schema = getattr(self, 'schema')
        setattr(schema, 'models', UC_SQL_Entity(uc_name=f"{self.catalog.name}.{self.volume_hf_hub_cache_schema}",
                                                sql_file="CREATE_SCHEMA_models.sql",
                                                sql_folder=self.config_ddl_folder))
        setattr(self, 'hub_volume', UC_Volume(uc_name=f'{schema.models.name}.{self.volume_hf_hub_cache_volume}',
                                          sql_file="CREATE_VOLUME_huggingface_hub.sql",
                                          sql_folder=self.config_ddl_folder,
                                          _path_value=self.volume_hf_hub_cache_dir))
        setattr(self, 'experiment', WS_GenAI_HF_Experiment(experiments_workspace_folder=self.experiments_workspace_folder,
                                                           volume_hf_hub_root=self.hub_volume.path,
                                                           hf_login_token_secret=self.hf_login_token_secret,
                                                           local_hf_hub_cache=self.local_hf_hub_cache))