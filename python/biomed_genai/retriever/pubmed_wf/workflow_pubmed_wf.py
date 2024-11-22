from biomed_genai.config import *

@dataclass
class Workflow_pubmed_wf:
    # This is the class we'll use to consolidate our uc application entities into a single configuration
    catalog_name: str = 'biomed_genai'
    raw_schema_name: str = 'raw'
    curated_schema_name: str = 'curated'
    processed_schema_name: str = 'processed'
    config_ddl_folder: str = "./ddl"
    config_vs_folder: str = "./ddl"

    @classmethod
    def increment_class_variable(cls, value):
        cls.class_variable += value  # Access class variable

    def __post_init__(self):
        setattr(self, 'catalog', UC_SQL_Entity(uc_name=self.catalog_name,
                                               sql_file="CREATE_CATALOG_biomed_workflow.sql",
                                               sql_folder=self.config_ddl_folder))
        setattr(self, 'schema', type('Schema', (object,), {}))
        schema = getattr(self, 'schema')
        setattr(schema, 'raw', UC_SQL_Entity(uc_name=f"{self.catalog_name}.{self.raw_schema_name}",
                                             sql_file="CREATE_SCHEMA_raw.sql",
                                             sql_folder=self.config_ddl_folder))
        setattr(schema, 'curated', UC_SQL_Entity(uc_name=f"{self.catalog_name}.{self.curated_schema_name}",
                                                 sql_file="CREATE_SCHEMA_curated.sql",
                                                 sql_folder=self.config_ddl_folder))
        setattr(schema, 'processed', UC_SQL_Entity(uc_name=f"{self.catalog_name}.{self.processed_schema_name}",
                                                   sql_file="CREATE_SCHEMA_processed.sql",
                                                   sql_folder=self.config_ddl_folder))
        setattr(self, 'raw_metadata_xml', UC_Table(uc_name=f"{schema.raw.name}.metadata_xml",
                                                   sql_file="CREATE_TABLE_raw_metadata_xml.sql",
                                                   sql_folder=self.config_ddl_folder))
        raw_metadata_xml = getattr(self, 'raw_metadata_xml')
        setattr(raw_metadata_xml, 'cp', UC_Volume(uc_name=f"{schema.raw.name}._checkpoints",
                                                  sql_file="CREATE_VOLUME_raw_checkpoints.sql",
                                                  sql_folder=self.config_ddl_folder,
                                                  _path_value=f'metadata_xml'))
        setattr(self, 'raw_search_hist', UC_Table(uc_name=f"{schema.raw.name}.search_hist",
                                                  sql_file="CREATE_TABLE_raw_search_hist.sql",
                                                  sql_folder=self.config_ddl_folder))
        setattr(self, 'raw_articles_xml', UC_Volume(uc_name=f'{schema.raw.name}.articles',
                                                    sql_file="CREATE_VOLUME_raw_articles_xml.sql",
                                                    sql_folder=self.config_ddl_folder,
                                                    _path_value=f'all/xml'))
        setattr(self, 'curated_articles_xml', UC_Table(uc_name=f'{schema.curated.name}.articles_xml',
                                                       sql_file="CREATE_TABLE_curated_articles_xml.sql",
                                                       sql_folder=self.config_ddl_folder))
        curated_articles_xml = getattr(self, 'curated_articles_xml')
        setattr(curated_articles_xml, 'cp', UC_Volume(uc_name=f'{schema.curated.name}._checkpoints',
                                                      sql_file="CREATE_VOLUME_curated_checkpoints.sql",
                                                      sql_folder=self.config_ddl_folder,
                                                      _path_value=f'articles_xml'))
        setattr(self, 'processed_articles_content', UC_Table(uc_name=f"{schema.processed.name}.articles_content",
                                                             sql_file="CREATE_TABLE_processed_articles_content.sql",
                                                             sql_folder=self.config_ddl_folder))
        processed_articles_content = getattr(self, 'processed_articles_content')
        setattr(processed_articles_content, 'cp', UC_Volume(uc_name=f'{schema.processed.name}._checkpoints',
                                                            sql_file="CREATE_VOLUME_processed_checkpoints.sql",
                                                            sql_folder=self.config_ddl_folder,
                                                            _path_value=f'articles_content_xml'))
        setattr(self, 'vector_search', type('Vector_Search', (object,), {}))
        vector_search = getattr(self, 'vector_search')
        setattr(vector_search, 'biomed', WS_Endpoint(ws_name='biomed',
                                                     json_file="ENDPOINT_biomed.json",
                                                     json_folder=self.config_vs_folder))
        biomed = getattr(vector_search, 'biomed')
        setattr(biomed, 'processed_articles_content_vs_index',
                WS_Index(ws_name=f'{schema.processed.name}.articles_content_vs_index',
                         json_file="INDEX_processed_articles_content_vs_index.json",
                         json_folder=self.config_vs_folder))