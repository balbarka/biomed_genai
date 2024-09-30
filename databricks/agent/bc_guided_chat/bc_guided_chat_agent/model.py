import mlflow
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from operator import itemgetter


class LangchainCustomModel:
    def __init__(self, conf_path="./config.yaml"):
        # Enable MLflow tracing for the model
        self._enable_mlflow_tracing()
        # Load model configuration from a YAML file
        self.model_config = mlflow.models.ModelConfig(development_config=conf_path)
        # Setup vector search for retrieving relevant documents
        self._setup_vector_search()
        # Prepare the chat model for generating responses
        self._prepare_model()
        # Define personas descriptions to assist in generating personalized responses
        self.personas_descriptions = self._personas_descriptions()
        # Prepare the prompt template for the chat model
        self._prepare_prompt()
        # Create the processing chain for handling queries
        self.create_chain()
        # Register the model with MLflow
        self._set_mlflow_model()

    def _enable_mlflow_tracing(self):
        # Automatically log model details with MLflow
        mlflow.langchain.autolog(disable=False)

    def _setup_vector_search(self):
        # Initialize vector search index from configuration
        self.vs_index = VectorSearchClient(disable_notice=True).get_index(
            endpoint_name=self.model_config.get("vector_search_endpoint_name"),
            index_name=self.model_config.get("vector_search_index_name"),
        )
        # Configure vector search as a retriever for documents
        self.vector_search_as_retriever = DatabricksVectorSearch(
            self.vs_index,
            text_column="content",
            columns=["id", "content"],
        ).as_retriever(search_kwargs={"k": 5})
        # Set schema for the retrieved documents
        mlflow.models.set_retriever_schema(primary_key="id", text_column="content")

    def _prepare_model(self):
        # Initialize the chat model with endpoint and parameters from configuration
        self.model = ChatDatabricks(
            endpoint=self.model_config.get("llm_model_serving_endpoint_name"),
            extra_params={"temperature": 0.01, "max_tokens": 500},
        )

    def _personas_descriptions(self):
        # Return descriptions for different personas to assist in generating personalized responses
        return self.model_config.get("persona_description_template")

    def _prepare_prompt(self):
        # Prepare the prompt template for generating responses based on user persona and context
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.model_config.get("llm_prompt_template"),
                ),
                ("user", "{question}"),
            ]
        )

    def classify_persona(self, question):
        # Classify the user's question into one of the predefined personas
        classification_prompt = self.model_config.get("llm_persona_classifier_prompt_template")

        persona = self.model.invoke(input=classification_prompt).content.strip()
        return persona

    def _format_context(self, docs):
        # Format the retrieved documents into a string for inclusion in the prompt
        chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
        return "".join(chunk_contents)

    def _extract_user_query_string(self, chat_messages_array):
        # Extract the latest user query from a list of chat messages
        return chat_messages_array[-1]["content"]

    def create_chain(self):
        # Create a processing chain for handling queries, including persona classification and response generation
        self.chain = (
            {
                "question": itemgetter("messages")
                | RunnableLambda(self._extract_user_query_string),
                "persona": itemgetter("messages")
                | RunnableLambda(self._extract_user_query_string)
                | RunnableLambda(self.classify_persona),
                "personas_descriptions": RunnableLambda(
                    lambda _: self.personas_descriptions
                ),
                "context": itemgetter("messages")
                | RunnableLambda(self._extract_user_query_string)
                | self.vector_search_as_retriever
                | RunnableLambda(self._format_context),
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def get_chain(self):
        # Return the processing chain
        return self.chain

    def _set_mlflow_model(self):
        # Register the processing chain as a model with MLflow
        mlflow.models.set_model(model=self.chain)


# if __name__ == "__main__":
    # TODO
    # 1. test wrap up this in a python main exectuable

# instantiate
chain = LangchainCustomModel(conf_path="./config.yaml").get_chain()    
mlflow.models.set_model(model=chain)

  
