import uuid
from dataclasses import dataclass
from typing import ClassVar

from langchain.storage import InMemoryStore
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from mulmod.extract import Extraction
from mulmod.retrieve.custom_vector import (
    MyMultiVectorRetriever,
)

RetrievalResult = list[tuple[Document, float]]


@dataclass
class Retriever:
    """
    Retriever class responsible for retrieving relevant extractions.

    Attributes:
    top_k (int):
        Number of top documents to retrieve.
    id_key (ClassVar[str]):
        Class variable representing the key keyword for document IDs.
    """

    top_k: int = 3
    id_key: ClassVar[str] = "doc_id"

    def __post_init__(self) -> None:
        """
        Create chroma vector db with cosine distance and multivector retriver.
        """

        vectorstore = Chroma(
            collection_name="single-doc-retriever",
            embedding_function=GPT4AllEmbeddings(),
            collection_metadata={"hnsw:space": "cosine"},
        )

        self.retriever = MyMultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=InMemoryStore(),
            id_key=Retriever.id_key,
            search_kwargs={"k": self.top_k},
        )

    def add_docs_from_texts(self, keys: list[str], values: list[str]) -> None:
        """
        Adds documents from text content to the retriever.

        Args:
        keys (List[str]):
            List of keys (e.g., summaries).
        values (List[str]):
            List of text content corresponding to the keys.
        """

        if len(values) == 0:
            return

        ids = [str(uuid.uuid4()) for _ in values]

        doc_keys = Retriever.text2doc(keys, ids)
        doc_values = Retriever.text2doc(values, ids)

        self.retriever.vectorstore.add_documents(doc_keys)
        self.retriever.docstore.mset(list(zip(ids, doc_values)))

    def add_imgs_from_extract(
        self, summaries: list[str], paths: list[Extraction]
    ) -> None:
        """
        Adds documents from image extractions to the retriever.

        Args:
        summaries (List[str]):
            List of text summaries corresponding to the images.
        paths (List[Extraction]):
            List of Extraction objects with image paths.
        """

        if len(paths) == 0:
            return

        ids = [str(uuid.uuid4()) for _ in paths]

        doc_keys = Retriever.text2doc(summaries, ids)
        doc_values = Retriever.extr_img2doc(paths, ids)

        self.retriever.vectorstore.add_documents(doc_keys)
        self.retriever.docstore.mset(list(zip(ids, doc_values)))

    def retrieve(self, query: str, treshold: float = 0.65) -> RetrievalResult:
        """
        Retrieves relevant documents based on a text query.

        Args:
        query (str):
            Query string to retrieve relevant documents.
        treshold (float):
            Threshold value for document relevance. If higher then do not return it.

        Returns:
            RetrievalResult: List of tuples containing Document objects and their similarity scores.
        """

        rel_docs = self.retriever.get_relevant_documents_with_score(query)

        rel_docs = [e for e in rel_docs if e[1] <= treshold]

        return rel_docs

    @classmethod
    def extr_img2doc(cls, imgs: list[Extraction], ids: list[str]) -> list[Document]:
        return [
            Document(
                page_content=e.content,
                metadata={"img_path": e.content, cls.id_key: ids[i]},
            )
            for i, e in enumerate(imgs)
        ]

    @classmethod
    def text2doc(cls, texts: list[str], ids: list[str]) -> list[Document]:
        return [
            Document(page_content=s, metadata={cls.id_key: ids[i]})
            for i, s in enumerate(texts)
        ]
