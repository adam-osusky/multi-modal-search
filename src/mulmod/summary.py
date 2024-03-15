from dataclasses import dataclass
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from mulmod.extract import Extraction, ExtractionType
from mulmod.img import get_img_base64

SUMMARY_PROMPT_TEXT = """In the context of machine learning, summarize the following text \
chunk in {num_words} words, highlighting the most important information which can be \
extracted from it. Text chunk: {extraction_content}"""

SUMMARY_PROMPT_TABLE = """In the context of machine learning, summarize the following table in \
{num_words} words, highlighting the most important information which can be extracted from it.\
Table: {extraction_content}"""

SUMMARY_PROMPT_IMAGE = """In the context of machine learning, summarize the following image in \
{num_words} words, highlighting the most important information which can be extracted from it."""

SYSTEM_PROMPT = """In the field of machine learning and large language models, you excel at \
summarizing research papers. You can analyze text excerpts, tables, or images and extract the \
key points in a clear and concise way."""


@dataclass
class Summarizer:
    """Class for generating summaries of texts, tables and images. Currently it use ollama \
    models.

    It can have different asking prompts for imgs, tables and texts. These prompts can use \
    variable `extraction_content` for texts and `num_words` for limiting the length of a \
    summary.

    Attributes:
    text_prompt:
        Prompt template for text summarization.
    table_prompt:
        Prompt template for table summarization. 
    image_prompt:
        Prompt template for image summarization.
    system_prompt:
        System message prompt template used for introducing the LLMs role.
    model:
        Name of the LLM model to be used for summarization from Ollama.
    num_words: 
        Target number of words for the generated summary.
    """

    text_prompt: str = SUMMARY_PROMPT_TEXT
    table_prompt: str = SUMMARY_PROMPT_TABLE
    image_prompt: str = SUMMARY_PROMPT_IMAGE
    system_prompt: str = SYSTEM_PROMPT
    model: str = "llava"
    num_words: int = 100

    def __post_init__(self) -> None:
        self.prompts: dict[ExtractionType, str] = {
            ExtractionType.TEXT: self.text_prompt,
            ExtractionType.TABLE: self.table_prompt,
            ExtractionType.IMAGE: self.image_prompt,
        }

    def get_summary(self, extractions: list[Extraction]) -> list[str]:
        """Generates list of summaries. Simple zero-shot chain."""

        if len(extractions) == 0:
            return []

        llm = ChatOllama(model=self.model)

        chain = self.get_prompt | llm | StrOutputParser()

        summaries = chain.batch(extractions, {"max_concurrency": 5})

        return summaries

    def get_prompt(self, extraction: Extraction) -> list[BaseMessage]:
        """Creates message chat with system initialization and query."""

        msgs = []

        temp_kwargs = {
            "extraction_content": extraction.content,
            "num_words": self.num_words,
        }

        system_temp = SystemMessagePromptTemplate.from_template(self.system_prompt)
        msgs.append(system_temp.format(**temp_kwargs))

        msgs.append(self.get_query_msg(extraction.type, **temp_kwargs))

        return msgs

    def get_query_msg(self, type: ExtractionType, **kwargs: Any) -> BaseMessage:
        """Prepares query message. For including image the image must be in base64."""

        if type == ExtractionType.IMAGE:
            temp = HumanMessagePromptTemplate.from_template(self.prompts[type])
            text = temp.format(**kwargs).content

            img_path = Summarizer.from_kwargs("extraction_content", **kwargs)
            image_b64 = get_img_base64(img_path)

            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
            }

            content_parts = []

            text_part = {"type": "text", "text": text}

            content_parts.append(image_part)
            content_parts.append(text_part)

            return HumanMessage(content=content_parts)

        else:
            temp = HumanMessagePromptTemplate.from_template(self.prompts[type])
            return temp.format(**kwargs)

    @staticmethod
    def from_kwargs(kw: str, **kwargs: Any) -> Any:
        """Get specific argument from keyword args."""

        if kw in kwargs:
            return kwargs[kw]
        else:
            raise RuntimeError(f"Expected to have {kw} in {kwargs}")
