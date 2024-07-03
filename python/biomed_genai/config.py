import pyspark
import delta
from dataclasses import dataclass
from pyspark.sql import SparkSession
from functools import cached_property
import re
import os


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


@dataclass
class BioMedConfig:
    # This is the class we'll use to consolidate our uc application entities into a single configuration
    _catalog_name: str = 'biomed_genai'
    _schema_raw_name: str = 'raw'
    _schema_curated_name: str = 'curated'
    _schema_processed_name: str = 'processed'
    _config_sql_folder: str = "./ddl"
    _config_json_folder: str = "./ddl"

    def __post_init__(self):
        setattr(self, 'catalog', UC_SQL_Entity(uc_name=self._catalog_name,
                                               sql_file="CREATE_CATALOG_biomed_pipeline.sql",
                                               sql_folder=self._config_sql_folder))
        setattr(self, 'schema', type('Schema', (object,), {}))
        schema = getattr(self, 'schema')
        setattr(schema, 'raw', UC_SQL_Entity(uc_name=f"{self._catalog_name}.{self._schema_raw_name}",
                                             sql_file="CREATE_SCHEMA_raw.sql",
                                             sql_folder=self._config_sql_folder))
        setattr(schema, 'curated', UC_SQL_Entity(uc_name=f"{self._catalog_name}.{self._schema_curated_name}",
                                                 sql_file="CREATE_SCHEMA_curated.sql",
                                                 sql_folder=self._config_sql_folder))
        setattr(schema, 'processed', UC_SQL_Entity(uc_name=f"{self._catalog_name}.{self._schema_processed_name}",
                                                   sql_file="CREATE_SCHEMA_processed.sql",
                                                   sql_folder=self._config_sql_folder))
        setattr(self, 'raw_metadata_xml', UC_Table(uc_name=f"{schema.raw.name}.metadata_xml",
                                                   sql_file="CREATE_TABLE_raw_metadata_xml.sql",
                                                   sql_folder=self._config_sql_folder))
        raw_metadata_xml = getattr(self, 'raw_metadata_xml')
        setattr(raw_metadata_xml, 'cp', UC_Volume(uc_name=f"{schema.raw.name}._checkpoints",
                                                  sql_file="CREATE_VOLUME_raw_checkpoints.sql",
                                                  sql_folder=self._config_sql_folder,
                                                  _path_value=f'metadata_xml'))
        setattr(self, 'raw_search_hist', UC_Table(uc_name=f"{schema.raw.name}.search_hist",
                                                  sql_file="CREATE_TABLE_raw_search_hist.sql",
                                                  sql_folder=self._config_sql_folder))
        setattr(self, 'raw_articles_xml', UC_Volume(uc_name=f'{schema.raw.name}.articles',
                                                    sql_file="CREATE_VOLUME_raw_articles_xml.sql",
                                                    sql_folder=self._config_sql_folder,
                                                    _path_value=f'all/xml'))
        setattr(self, 'curated_articles_xml', UC_Table(uc_name=f'{schema.curated.name}.articles_xml',
                                                       sql_file="CREATE_TABLE_curated_articles_xml.sql",
                                                       sql_folder=self._config_sql_folder))
        curated_articles_xml = getattr(self, 'curated_articles_xml')
        setattr(curated_articles_xml, 'cp', UC_Volume(uc_name=f'{schema.curated.name}._checkpoints',
                                                      sql_file="CREATE_VOLUME_curated_checkpoints.sql",
                                                      sql_folder=self._config_sql_folder,
                                                      _path_value=f'articles_xml'))
        setattr(self, 'processed_articles_content', UC_Table(uc_name=f"{schema.processed.name}.articles_content",
                                                             sql_file="CREATE_TABLE_processed_articles_content.sql",
                                                             sql_folder=self._config_sql_folder))
        processed_articles_content = getattr(self, 'processed_articles_content')
        setattr(processed_articles_content, 'cp', UC_Volume(uc_name=f'{schema.processed.name}._checkpoints',
                                                            sql_file="CREATE_VOLUME_processed_checkpoints.sql",
                                                            sql_folder=self._config_sql_folder,
                                                            _path_value=f'articles_content_xml'))

    @cached_property
    def spark(self) -> SparkSession:
        return SparkSession.builder.getOrCreate()
