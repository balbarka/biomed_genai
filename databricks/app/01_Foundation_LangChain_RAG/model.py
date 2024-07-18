import sys
import mlflow
import importlib

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter


## Enable MLflow Tracing
mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config="config.yaml")

@RunnableLambda
def last_user_content(messages: [dict]):
    return messages[-1]["content"]


def get_lc_retriever(endpoint_name: str, index_name: str, k=3):
    return DatabricksVectorSearch(index=VectorSearchClient(disable_notice=True).get_index(endpoint_name, index_name),
                                  text_column="content",
                                  columns=["id", "content"]).as_retriever(search_kwargs={"k": k})


@RunnableLambda
def format_search_results(docs: [Document]):
    return "".join([f"Passage: {d.page_content}\n" for d in docs])


def get_lc_prompt_template(llm_prompt_template: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", llm_prompt_template),
                                             ("user", "{question}")])


def get_lc_model(endpoint_name: str) -> ChatDatabricks:
    return ChatDatabricks(endpoint=endpoint_name,
                          extra_params={"temperature": 0.01, "max_tokens": 500})


def get_lc_chain(endpoint_name: str,
                 index_name: str,
                 prompt_template: str,
                 model_name: str):
    lc_retriever = get_lc_retriever(endpoint_name, index_name)
    lc_prompt = get_lc_prompt_template(prompt_template)
    lc_model = get_lc_model(model_name)
    return ({"question": itemgetter("messages") | last_user_content,
             "context": itemgetter("messages") | last_user_content | lc_retriever | format_search_results}
            | lc_prompt | lc_model | StrOutputParser())

chain = get_lc_chain(endpoint_name=model_config.get('vector_search_endpoint_name'),
                     index_name=model_config.get('vector_search_index_name'),
                     prompt_template=model_config.get('llm_prompt_template'),
                     model_name=model_config.get('llm_model_serving_endpoint_name'))

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)