#from openai.types.chat.chat_completion_message import ChatCompletionMessage
from mlflow.types.llm import ChatMessage
from dataclasses import dataclass, field
from typing import Union, List, Optional

from delta.tables import DeltaTable
import pandas as pd
import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException

# From docs: A messages field that follows the OpenAI chat completion schema and can encode the full conversation.
# We could use ChatCompletionMessage, but it has some post init constraints we may not want
# We'll adopt a convention of using an existing MLFlow class already based upon OpenAI schema, ChatMessage

@dataclass
class MessagesEntity():
    messages: List[ChatMessage]

@dataclass
class QueryHistEntity():
    query: str
    history: List[ChatMessage]

@dataclass
class EvalSetRequest(QueryHistEntity):

    @classmethod
    def from_query(cls, query:str):
        return cls(query=query, 
                   history=[])
    
    @classmethod
    def from_messages(cls, messages:MessagesEntity):
        return cls(query=messages[-1],
                   history=messages[:-1])
    
    @property
    def as_query(self) -> str:
        return self.query

    @property
    def as_messages(self) -> MessagesEntity:
        return MessagesEntity([self.query,] + self.history)

    @property
    def as_query_hist(self) -> QueryHistEntity:
        return self
    
    @property
    def as_dict(self) -> dict:
        return {"query": self.query,
                "history": [m.__dict__ for m in self.history]}

    @property
    def as_str(self) -> dict:
        if len(self.history)==0:
            return self.query
        else:
            return str(self.as_dict)


@dataclass
class EvalSetRetrieverContext():
    content: str
    doc_uri: str




@dataclass
class EvalSetEntry():
    request_id: Optional[str] = field(default=None)
    request: EvalSetRequest = field(default="")
    expected_retrieved_context: Optional[List[EvalSetRetrieverContext]] = field(default=None)
    expected_response: Optional[str] = field(default="")
    response: Optional[str] = field(default=None)
    retrieved_context: Optional[List[EvalSetRetrieverContext]] = field(default=None)
    trace: Optional[str] = field(default=None)

    @property
    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d['request'] = self.request.as_str
        return d

    
@dataclass
class EvalSet():
    set: List[EvalSetEntry]=field(default_factory=list)

    @property
    def spark(self) -> SparkSession:
        return SparkSession.builder.getOrCreate()


    @property
    def as_df(self):
        # Simple Dataframe format for simple eval demo
        return pd.DataFrame([e.as_dict for e in self.set])
    
    @property
    def as_ds(self):
        return mlflow.data.from_pandas(self.as_df)
    
    def create_or_replace_delta(self, uc_name:str, overwrite=False):
        bool_insert=False
        try:
            table = self.spark.table(uc_name)
            if ~table.isEmpty():
                bool_insert=True
        except AnalysisException:
            bool_insert=True
        if bool_insert or overwrite:
            print(f'Insert Overwrite {uc_name}.')
            self.spark.createDataFrame(self.as_df).write.format("delta").mode("overwrite").saveAsTable(uc_name)
        else:
            print(f'No Action, {uc_name} is not empty.')