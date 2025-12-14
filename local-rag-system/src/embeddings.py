import logging
from typing import Any, List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.constants import EMBEDDING_MODEL_PATH
from src.utils import setup_logging

# Initialize logger
setup_logging()  # Configures logging for the application
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the embedding model.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    logger.info(f"Loading embedding model from path: {EMBEDDING_MODEL_PATH}")
    return SentenceTransformer(EMBEDDING_MODEL_PATH)


def generate_embeddings(chunks: List[str]) -> List[np.ndarray[Any, Any]]:
    """
    Generates embeddings for a list of text chunks.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        List[np.ndarray[Any, Any]]: List of embeddings as numpy arrays for each chunk.
    """
    model = get_embedding_model()
    embeddings = [np.array(model.encode(chunk)) for chunk in chunks]
    logger.info(f"Generated embeddings for {len(chunks)} text chunks.")
    return embeddings
