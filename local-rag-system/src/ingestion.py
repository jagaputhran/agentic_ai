import json
import logging
from typing import Any, Dict, List, Tuple

from opensearchpy import OpenSearch, helpers

from src.constants import ASSYMETRIC_EMBEDDING, EMBEDDING_DIMENSION, OPENSEARCH_INDEX
from src.opensearch import get_opensearch_client
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


def load_index_config() -> Dict[str, Any]:
    """
    Loads the index configuration from a JSON file.

    Returns:
        Dict[str, Any]: The index configuration as a dictionary.
    """
    with open("src/index_config.json", "r") as f:
        config = json.load(f)

    # Replace the placeholder with the actual embedding dimension
    config["mappings"]["properties"]["embedding"]["dimension"] = EMBEDDING_DIMENSION
    logger.info("Index configuration loaded from src/index_config.json.")
    return config if isinstance(config, dict) else {}


def create_index(client: OpenSearch) -> None:
    """
    Creates an index in OpenSearch using settings and mappings from the configuration file.

    Args:
        client (OpenSearch): OpenSearch client instance.
    """
    index_body = load_index_config()
    if not client.indices.exists(index=OPENSEARCH_INDEX):
        response = client.indices.create(index=OPENSEARCH_INDEX, body=index_body)
        logger.info(f"Created index {OPENSEARCH_INDEX}: {response}")
    else:
        logger.info(f"Index {OPENSEARCH_INDEX} already exists.")


def delete_index(client: OpenSearch) -> None:
    """
    Deletes the index in OpenSearch if it exists.

    Args:
        client (OpenSearch): OpenSearch client instance.
    """
    if client.indices.exists(index=OPENSEARCH_INDEX):
        response = client.indices.delete(index=OPENSEARCH_INDEX)
        logger.info(f"Deleted index {OPENSEARCH_INDEX}: {response}")
    else:
        logger.info(f"Index {OPENSEARCH_INDEX} does not exist.")


def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """
    Indexes multiple documents into OpenSearch in bulk.

    Args:
        documents (List[Dict[str, Any]]): List of document dictionaries with 'doc_id', 'text', 'embedding', and 'document_name'.

    Returns:
        Tuple[int, List[Any]]: Tuple with the number of successfully indexed documents and a list of any errors.
    """
    actions = []
    client = get_opensearch_client()

    for doc in documents:
        doc_id = doc["doc_id"]
        embedding_list = doc["embedding"].tolist()
        document_name = doc["document_name"]

        # Prefix each document's text with "passage: " for the asymmetric embedding model
        if ASSYMETRIC_EMBEDDING:
            prefixed_text = f"passage: {doc['text']}"
        else:
            prefixed_text = f"{doc['text']}"

        action = {
            "_index": OPENSEARCH_INDEX,
            "_id": doc_id,
            "_source": {
                "text": prefixed_text,
                "embedding": embedding_list,  # Precomputed embedding
                "document_name": document_name,
            },
        }
        actions.append(action)

    # Perform bulk indexing and capture response details explicitly
    success, errors = helpers.bulk(client, actions)
    logger.info(
        f"Bulk indexed {len(documents)} documents into index {OPENSEARCH_INDEX} with {len(errors)} errors."
    )
    return success, errors


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """
    Deletes documents from OpenSearch where 'document_name' matches the provided value.

    Args:
        document_name (str): Name of the document to delete.

    Returns:
        Dict[str, Any]: Response from the delete-by-query operation.
    """
    client = get_opensearch_client()
    query = {"query": {"term": {"document_name": document_name}}}
    response: Dict[str, Any] = client.delete_by_query(
        index=OPENSEARCH_INDEX, body=query
    )
    logger.info(
        f"Deleted documents with name '{document_name}' from index {OPENSEARCH_INDEX}."
    )
    return response
