import sys

from mulmod.extract import Extractions, PdfExtractor
from mulmod.logger import get_logger
from mulmod.retrieve.rag import Rag
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
INTRO_CHAT_MSG = f"""\
Ask a query about the input PDF to get answer from the chatbot.
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


def remove_empty(extracts: Extractions) -> Extractions:
    return Extractions(
        texts=[e for e in extracts.texts if e.content != ""],
        tables=[e for e in extracts.tables if e.content != ""],
        images=extracts.images,
    )


def get_retriever(
    filepath: str,
    max_characters: int,
    new_after_n_chars: int,
    combine_text_under_n_chars: int,
    img_summary_num_words: int,
    summarize_texts=False,
    txt_summary_num_words: int = 50,
) -> Retriever:
    extractor = PdfExtractor(
        max_characters=max_characters,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine_text_under_n_chars,
    )
    extractions = extractor.extract(filepath)
    extractions = remove_empty(extractions)

    texts = [e.content for e in extractions.texts]
    tables = [e.content for e in extractions.tables]

    logger.info("Started generating summaries for images.")
    image_summarizer = Summarizer(num_words=img_summary_num_words)
    image_summaries = image_summarizer.get_summary(extractions.images)
    logger.info("Finished generating summaries for images.")

    retriever = Retriever()

    if summarize_texts:
        text_summarizer = Summarizer(num_words=txt_summary_num_words)
        logger.info("Started generating summaries for texts and tables.")
        text_summaries = text_summarizer.get_summary(extractions.texts)
        table_summaries = text_summarizer.get_summary(extractions.tables)
        logger.info("Finished generating summaries for texts and tables.")
    else:
        text_summaries = texts
        table_summaries = tables

    logger.info("Started creating vector database for retrieval.")
    retriever.add_docs_from_texts(text_summaries, texts)
    retriever.add_docs_from_texts(table_summaries, tables)
    retriever.add_imgs_from_extract(image_summaries, extractions.images)
    logger.info("Finished creating vector database for retrieval.")

    return retriever


def retrieval_only(pdf_path: str) -> None:
    retriever = get_retriever(
        filepath=pdf_path,
        max_characters=600,
        new_after_n_chars=550,
        combine_text_under_n_chars=500,
        img_summary_num_words=50,
        summarize_texts=False,
    )

    print(INTRO_RET_MSG)
    while True:
        query = input("Query: ")
        if STOP_TOKEN in query:
            break

        print_relevant(retriever.retrieve(query, treshold=1.0))


def rag(pdf_path: str) -> None:
    retriever = get_retriever(
        filepath=pdf_path,
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        img_summary_num_words=50,
        summarize_texts=True,
        txt_summary_num_words=50,
    )
    ai = Rag()

    print(INTRO_CHAT_MSG)
    while True:
        query = input("Query: ")
        if STOP_TOKEN in query:
            break
        answer = ai.answer(query, retriever)
        print(answer)


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
