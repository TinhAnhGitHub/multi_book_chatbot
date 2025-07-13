import os
from tqdm import tqdm
from llama_index.core import Document
import pymupdf4llm
import pymupdf

from ..utils.logger_config import SimpleLogger


logger = SimpleLogger(__name__)

def load_books(book_paths: str, index_folder: str) -> tuple[list[Document], list[str], list[str]]:

    exists_index_file = set(os.listdir(index_folder))  # Files already indexed
    all_pdf_paths = []

    for root, _, files in os.walk(book_paths):
        for file in files:
            if file.endswith('.pdf'):
                all_pdf_paths.append(os.path.join(root, file))

    new_book_paths = []
    existing_books = []

    for pdf_path in all_pdf_paths:
        pdf_name = os.path.basename(pdf_path).split('.')[0]
        if pdf_name not in exists_index_file:
            new_book_paths.append(pdf_path)
        else:
            existing_books.append(pdf_path)

    logger.info(f"{len(new_book_paths)} new books to process.")
    logger.info(f"{len(existing_books)} books already indexed (skipped).")

    book_docs: list[Document] = []

    for book_path in tqdm(new_book_paths, desc="Processing books into Document..."):
        try:
            document = pymupdf.open(book_path)
            metadata_pdf = document.metadata

            metadata = {
                'producer': metadata_pdf.get('producer'),
                'author': metadata_pdf.get('author'),
                'keywords': metadata_pdf.get('keywords'),
                'title': metadata_pdf.get('title'),
                'subject': metadata_pdf.get('subject'),
                'file_name': os.path.basename(book_path)
            } if metadata_pdf else {}

            if not metadata_pdf:
                logger.warning(f"Book {os.path.basename(book_path)} has no metadata.")

            markdown_content = pymupdf4llm.to_markdown(book_path)

            book_doc = Document(
                text=markdown_content,
                metadata=metadata
            )
            book_docs.append(book_doc)

        except Exception as e:
            logger.exception(f"Error processing {book_path}: {e}")

    return book_docs, [os.path.basename(p) for p in new_book_paths], [os.path.basename(p) for p in existing_books]



        
