import sys

from mulmod.extract import PdfExtractor
from mulmod.logger import get_logger
from mulmod.retrieve.retriever import RetrievalResult, Retriever
from mulmod.summary import Summarizer

USAGE = """\
Usage:
  python main.py <filepath> <mode>
Arguments:
  <filepath>    : Path to the PDF file.
  <mode>        : 0: retrieval_only
                  1: rag
"""

STOP_TOKEN = "<stop>"
INTRO_RET_MSG = f"""\
Ask a query about the input PDF to print relevant texts and images for it.
To stop simply type {STOP_TOKEN}.
"""

logger = get_logger(__name__)


def print_relevant(retrieval: RetrievalResult) -> None:
    if len(retrieval) == 0:
        print("Did not find relevant documents for this query. Try to rephrase it.")
        return

    print("Found this relevant texts and images :")
    for e in retrieval:
        print()
        print(f"Similarity score {e[1]} for:")
        print()
        print(e[0].page_content)
        print()


def retrieval_only(pdf_path: str) -> None:
    extractor = PdfExtractor(
        max_characters=600,
        new_after_n_chars=550,
        combine_text_under_n_chars=500,
    )
    extractions = extractor.extract(filepath)

    texts = [e.content for e in extractions.texts]
    tables = [e.content for e in extractions.tables]

    logger.info("Started generating summaries for images.")
    image_summarizer = Summarizer(num_words=10)
    image_summaries = image_summarizer.get_summary(extractions.images)
    logger.info("Finished generating summaries for images.")

    logger.info("Started creating vector database for retrieval.")
    retriever = Retriever()
    retriever.add_docs_from_texts(texts, texts)
    retriever.add_docs_from_texts(tables, tables)
    retriever.add_imgs_from_extract(image_summaries, extractions.images)
    logger.info("Finished creating vector database for retrieval.")

    print(INTRO_RET_MSG)
    while True:
        query = input("Query: ")
        if STOP_TOKEN in query:
            break

        print_relevant(retriever.retrieve(query, treshold=1.0))


def rag(pdf_path: str) -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        mode = int(sys.argv[2])
    except ValueError:
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    if mode == 0:
        retrieval_only(filepath)
    else:
        rag(filepath)
