from dataclasses import dataclass
from biomed_genai.config import UC_SQL_Entity, WS_GenAI_Model_Experiment, UC_Dataset, UC_Registered_Model, UC_Table, WS_GenAI_Agent_Experiment

@dataclass
class Model_bc_chat_ift:
    # This is the class we'll use to consolidate our our model bc_chat_ift variables
    # NOTE: we will use this class across all notebooks within bc_chat_ift
    # NOTE: There is a dependency on the workflow that has brought in bc articles, wf_pubmed which must be instantiated and passed to this dataclass

    release_version: int
    config_ddl_folder: str
    experiments_workspace_folder: str
    catalog_name: str
    models_schema_name: str
    ft_model_name: str
    base_model_hf_name: str
    teacher_model_ep_name: str
    # dependent_classes: 
    source_table: UC_Table = None
    eval_ds: UC_Dataset = None
    
    def __post_init__(self):
        setattr(self, 'catalog', UC_SQL_Entity(uc_name=self.catalog_name,
                                               sql_file="CREATE_CATALOG_biomed_genai.sql",
                                               sql_folder=self.config_ddl_folder))
        setattr(self, 'schema', type('Schema', (object,), {}))
        schema = getattr(self, 'schema')
        setattr(schema, 'models', UC_SQL_Entity(uc_name=f"{self.catalog.name}.{self.models_schema_name}",
                                                sql_file="CREATE_SCHEMA_models.sql",
                                                sql_folder=self.config_ddl_folder))
        setattr(self, 'experiment', WS_GenAI_Model_Experiment(release_version=self.release_version,
                                                              model_name=self.ft_model_name,
                                                              experiments_workspace_folder=self.experiments_workspace_folder,
                                                              eval_ds=self.eval_ds,
                                                              default_model_name=self.ft_model_name))
        setattr(self, 'ift_seed', UC_Table(uc_name=f"{schema.models.name}.ift_seed",
                                           sql_file="CREATE_TABLE_ift_seed.sql",
                                           sql_folder=self.config_ddl_folder))
        setattr(self, 'ift_train', UC_Table(uc_name=f"{schema.models.name}.ift_train",
                                           sql_file="CREATE_TABLE_ift_train.sql",
                                           sql_folder=self.config_ddl_folder))
        setattr(self, 'experiment', WS_GenAI_Agent_Experiment(release_version=self.release_version,
                                                              agent_name=self.ft_model_name,
                                                              experiments_workspace_folder=self.experiments_workspace_folder,
                                                              eval_ds=self.eval_ds,
                                                              default_model_name=self.ft_model_name))
        setattr(self, 'registered_model', UC_Registered_Model(uc_name=f"{schema.models.name}.{self.ft_model_name}"))


