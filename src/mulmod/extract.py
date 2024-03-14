import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

from pydantic import BaseModel
from unstructured.documents.elements import CompositeElement, Element, Table
from unstructured.partition.pdf import partition_pdf

from mulmod.logger import get_logger

logger = get_logger(__name__)


class ExtractionType(Enum):
    TEXT = 1
    TABLE = 2
    IMAGE = 3


class Extraction(BaseModel):
    type: ExtractionType
    content: str


@dataclass
class Extractions:
    texts: List[Extraction]
    tables: List[Extraction]
    images: List[Extraction]


@dataclass
class PdfExtractor:
    """
    Extractor for PDF documents that partitions and extracts elements from the PDF.

    Attributes:
    - max_characters: Maximum number of characters to extract per chunk.
    - new_after_n_chars: Threshold for starting a new chunk during extraction.
    - combine_text_under_n_chars: Threshold for combining text chunks.
    - img_dir: Directory to store extracted images.
    """

    max_characters: int = 4000
    new_after_n_chars: int = 3800
    combine_text_under_n_chars: int = 3000

    img_dir: str = "./resources/figs"

    def extract(self, filepath: str) -> None:
        """
        TODO WIP
        """
        logger.info(f"Started extraction on {filepath}.")

        self.img_dir = os.path.join(self.img_dir, Path(filepath).stem)

        pdf_elements: List[Element] = partition_pdf(
            filename=filepath,
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=self.max_characters,
            new_after_n_chars=self.new_after_n_chars,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            extract_image_block_output_dir=self.img_dir,
        )

        texts, tables = self.categorize(pdf_elements)

        logger.info(
            f"Extracted {len(texts)} texts, {len(tables)} tables and {len(os.listdir(self.img_dir))} images."
        )
        logger.info(f"The extracted images are in {self.img_dir}")

        for t in texts:
            print(len(t.content))

    def categorize(
        self, elements: List[Element]
    ) -> tuple[list[Extraction], list[Extraction]]:
        """
        Categorizes elements into texts and tables.

        Parameters:
        - elements: List of elements extracted from the PDF.

        Returns:
        A tuple containing lists of text and table extractions.
        """

        texts = []
        tables = []

        for element in elements:
            if isinstance(element, CompositeElement):
                texts.append(Extraction(type=ExtractionType.TEXT, content=element.text))
            elif isinstance(element, Table):
                tables.append(
                    Extraction(type=ExtractionType.TABLE, content=element.text)
                )

        return texts, tables
