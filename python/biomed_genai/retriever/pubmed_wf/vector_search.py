"""Collection of Vectorsearch Helper Functions"""

from databricks.vector_search.client import VectorSearchClient


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
