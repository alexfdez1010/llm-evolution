from typing import Protocol, List, Dict, Any, runtime_checkable


@runtime_checkable
class VectorDatabase(Protocol):
    """Protocol for vector databases."""

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ) -> None:
        """
        Add items to the vector database.

        Args:
            ids: Unique identifiers for the items.
            embeddings: Embedding vectors for the items.
            metadatas: Metadata for each item.
            documents: Original documents or descriptions.
        """
        ...

    def query(
        self, query_embeddings: List[List[float]], n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar items.

        Args:
            query_embeddings: Embedding vectors to search for.
            n_results: Number of results to return.

        Returns:
            List[Dict[str, Any]]: List of query results.
        """
        ...
