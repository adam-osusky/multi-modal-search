import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
    """
    Representation of an extraction.

    Attributes:
        type (ExtractionType):
            Type of the extraction.
        content (str):
            Content of the extraction. If the type is IMAGE then the content is path
            to that image.
    """

    type: ExtractionType
    content: str


@dataclass
class Extractions:
    texts: list[Extraction]
    tables: list[Extraction]
    images: list[Extraction]


@dataclass
class PdfExtractor:
    """
    Extractor for PDF documents that partitions and extracts elements from the PDF.

    Attributes:
    max_characters:
        Maximum number of characters to extract per chunk.
    new_after_n_chars:
        Threshold for starting a new chunk during extraction.
    combine_text_under_n_chars:
        Threshold for combining text chunks.
    img_dir:
        Directory to store extracted images.
    """

    max_characters: int = 1000
    new_after_n_chars: int = 800
    combine_text_under_n_chars: int = 500

    base_img_dir: str = "./resources/figs"

    def extract(self, filepath: str) -> Extractions:
        """
        Extracts texts, tables, images from the PDF document.

        Parameters:
            filepath (str):
                Path to the PDF file.

        Returns:
            Extractions:
                Collection of extracted elements.
        """
        logger.info(f"Started extraction on {filepath}.")

        self.img_dir = os.path.join(self.base_img_dir, Path(filepath).stem)

        pdf_elements: list[Element] = partition_pdf(
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

        texts, tables = PdfExtractor.categorize(pdf_elements)

        logger.info(
            f"Extracted {len(texts)} texts, {len(tables)} tables and {len(os.listdir(self.img_dir))} images."
        )
        logger.info(f"The extracted images are in {self.img_dir}")

        images = PdfExtractor.get_imgs(self.img_dir)

        return Extractions(texts=texts, tables=tables, images=images)

    @staticmethod
    def categorize(
        elements: list[Element],
    ) -> tuple[list[Extraction], list[Extraction]]:
        """
        Categorizes elements into texts and tables.

        Parameters:
        elements:
            List of elements extracted from the PDF.

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

    @staticmethod
    def get_imgs(img_dir: str) -> list[Extraction]:
        images = []

        for filename in os.listdir(img_dir):
            filepath = os.path.join(img_dir, filename)
            if filename.endswith(".jpg"):
                images.append(Extraction(type=ExtractionType.IMAGE, content=filepath))

        return images
