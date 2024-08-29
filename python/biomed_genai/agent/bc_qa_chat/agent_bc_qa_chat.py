from biomed_genai.config import *

@dataclass
class Agent_model_bc_qa_chat:
    # This is the class we'll use to consolidate our our agent bc_qa_chat variables
    # NOTE: we will use this class across all model types and model notebooks within bc_qa_chat
    release_version: int
    catalog_name: str
    agents_schema_name: str
    agent_name: str
    experiments_workspace_folder: str
    eval_ds_name: str
    config_ddl_folder: str
    candidate_models: [str]
    default_model_name: str = None
    
    def __post_init__(self):
        if self.default_model_name not in self.candidate_models:
            self.default_model_name = None
        setattr(self, 'catalog', UC_SQL_Entity(uc_name=self.catalog_name,
                                               sql_file="CREATE_CATALOG_biomed_genai.sql",
                                               sql_folder=self.config_ddl_folder))
        setattr(self, 'schema', type('Schema', (object,), {}))
        schema = getattr(self, 'schema')
        setattr(schema, 'agents', UC_SQL_Entity(uc_name=f"{self.catalog.name}.{self.agents_schema_name}",
                                                sql_file="CREATE_SCHEMA_agents.sql",
                                                sql_folder=self.config_ddl_folder))
        setattr(self, 'experiment', WS_GenAI_Agent_Experiment(release_version=self.release_version,
                                                              agent_name=self.agent_name,
                                                              experiments_workspace_folder=self.experiments_workspace_folder,
                                                              eval_ds=UC_Dataset(uc_name=f"{self.schema.agents.name}.{self.eval_ds_name}",
                                                                                 sql_file="CREATE_TABLE_agents_bc_eval_ds.sql",
                                                                                 sql_folder=self.config_ddl_folder,
                                                                                 release_version=self.release_version),
                                                              default_model_name=self.default_model_name))
        setattr(self, 'registered_model', UC_Registered_Model(uc_name=f"{self.catalog.name}.{self.agents_schema_name}.{self.agent_name}"))
