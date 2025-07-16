import sys
import os
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_DIR)


from enum import Enum
import pickle
from pathlib import Path
from typing import cast
from llama_index.llms.gemini import Gemini
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    SummaryIndex,
    load_index_from_storage
)
import uuid


from fastapi import UploadFile, HTTPException
from src.utils.logger_config import SimpleLogger
from src.config.settings import AppConfig
from src.api.api_models import  FileDocument, FileStatus
from src.services.file_processing_service import FileStorageService, FileValidationService, DocumentProcessingService
from src.services.index_service import IndexService
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from src.services.database_service import  DatabaseService, FileService, VectorService


logger = SimpleLogger(__name__)






async def get_book_summary(summary_index: SummaryIndex, summary_path: str, llm: Gemini) -> str:

    assert os.path.exists(summary_path) or summary_index, "The summary path does not exist, require summary index"
    
    if os.path.exists(summary_path):
        with open(summary_path, 'rb') as f:
            return pickle.load(f)
    
    query = "Extract a concise 1-2 line summary of this document, focusing on its main themes and topics."
    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", llm=llm)

    summary = str(await summary_query_engine.aquery(query))
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'wb') as f:
        pickle.dump(summary,f)
    
    return summary


class UploadMessageStatus(Enum):
    SUCCESSFUL = "Upload Successful"
    WARNING_DUPLICATE = "Duplicated file detected! Upload Successful"


class UploadReturn:
    def __init__(self, message: UploadMessageStatus, file: FileDocument):
        self.message = message
        self.file = file


class IngestionPipeline:
    """
    Service for Ingestion Data Orchestration
    1. Control services
    2. File Management
    """

    def __init__(self, config: AppConfig, embed_mode: GoogleGenAIEmbedding):
        self.config = config
        self.index_service = IndexService(config, embed_mode)
        self.file_storage_service = FileStorageService(config)
        self.file_validation_service = FileValidationService(config)
        self.document_proc_service = DocumentProcessingService(config)

        
        self.db = DatabaseService(config)
        self.file_db_service = FileService(config)
        self.vector_db_service = VectorService(config)



    async def handle_file_upload(self, file: UploadFile, user_id: str) -> UploadReturn:

        assert self.db.mongo_db is not None, "Connect to the db first"

        return_message = UploadMessageStatus.SUCCESSFUL
        file_id = str(uuid.uuid4())

        get_all_metadata_file = await self.file_db_service.get_all_file_metadata()
        original_filename_list = [f.original_filename for f in get_all_metadata_file]
        
        original_filename = cast(str, file.filename)
        if original_filename_list and original_filename in original_filename_list:
            return_message = UploadMessageStatus.WARNING_DUPLICATE
        
        try:
            validation_result = await self.file_validation_service.validate_file(file)
            if not validation_result.is_valid:
                raise HTTPException(status_code=400, detail=validation_result.error_message)
            
            storage_path = await self.file_storage_service.save_uploaded_file(file, user_id, file_id)
            
            file_doc = FileDocument(
                file_id=file_id,
                filename=f"{file_id}_{file.filename}",
                original_filename=original_filename,
                file_type=cast(str, validation_result.file_type),
                file_size=cast(int, validation_result.file_size),
                status=FileStatus.UPLOADING,
                storage_path=storage_path
            )

            await self.file_db_service.save_file_metadata(file_doc=file_doc)

            

            return UploadReturn(message=return_message, file=file_doc)

        except Exception as e:
            logger.error(f"File upload error: {e}")
            await self.file_db_service.update_file_status(
                file_id, FileStatus.FAILED, error_message=str(e)
            )
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    async def _indexing(self, file_doc: FileDocument):
        




        
    
    



        

    