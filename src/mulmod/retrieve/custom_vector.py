from abc import abstractmethod
from typing import Any, Optional

from langchain.retrievers.multi_vector import (
    MultiVectorRetriever,
    SearchType,
)
from langchain_core.callbacks import (
    CallbackManagerForRetrieverRun,
    Callbacks,
)
from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.retrievers import BaseRetriever


class MyBaseRetriever(BaseRetriever):
    """
    Copied from langchain_core.retrievers and slightly modified so it
    returns also similarities scores in similarity search. In mmr it returns
    dummy zeroes.
    """

    @abstractmethod
    def _get_relevant_documents_with_score(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[tuple[Document, float]]:
        """Get documents relevant to a query with their score.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents and their score.
        """

    def get_relevant_documents_with_score(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Retrieve documents relevant to a query with their score.
        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
        Returns:
            List of relevant documents and their score.
        """
        from langchain_core.callbacks.manager import CallbackManager

        callback_manager = CallbackManager.configure(
            callbacks,
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            query,
            name=run_name,
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents_with_score(
                    query, run_manager=run_manager, **_kwargs
                )
            else:
                result = self._get_relevant_documents_with_score(query, **_kwargs)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                [t[0] for t in result],
            )
            return result


class MyMultiVectorRetriever(MultiVectorRetriever, MyBaseRetriever):
    """
    Copied from langchain.retrievers.multi_vector._get_relevant_documents and
    slightly modified so it returns also similarities scores in similarity search.
    In mmr it returns dummy zeroes.
    """

    def _get_relevant_documents_with_score(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[tuple[Document, float]]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents with their scores
        """
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
            sub_docs_with_sims = [(d, 0) for d in sub_docs]  # add dummy sim
        else:
            sub_docs_with_sims = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )

        # We do this to maintain the order of the ids that are returned
        ids = []
        sims = []
        for d, sim in sub_docs_with_sims:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
                sims.append(sim)
        docs = self.docstore.mget(ids)
        return [(d, s) for d, s in zip(docs, sims) if d is not None]
