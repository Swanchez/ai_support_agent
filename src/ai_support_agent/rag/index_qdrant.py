"""Initialize both Qdrant collections without sending a question to an LLM."""

from ai_support_agent.config import load_vector_store_config
from ai_support_agent.exceptions import ConfigurationError, EmbeddingRequestError, VectorStoreError
from ai_support_agent.rag.runtime import create_gemini_vector_retriever, create_external_reference_gemini_vector_retriever


def main() -> None:
    try:
        if load_vector_store_config().backend != "qdrant":
            raise ConfigurationError("Set RAG_VECTOR_BACKEND=qdrant before indexing.")
        for factory in (create_gemini_vector_retriever, create_external_reference_gemini_vector_retriever):
            retriever = factory()
            try:
                store = retriever.vector_store
                print(f"Ready: {store.collection_name}")
            finally:
                retriever.close()
    except (ConfigurationError, VectorStoreError) as error:
        raise SystemExit(str(error)) from error
    except EmbeddingRequestError as error:
        raise SystemExit("Embedding provider is unavailable; indexing was not completed.") from error


if __name__ == "__main__":
    main()
