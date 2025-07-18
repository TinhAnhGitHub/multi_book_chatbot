import sys
import os
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_DIR)


from enum import Enum
from typing import cast
import uuid

from fastapi import UploadFile, HTTPException
from src.utils.logger_config import SimpleLogger
from src.config.settings import AppConfig
from src.api.api_models import  FileDocument, FileStatus
from src.services.file_processing_service import FileStorageService, FileValidationService, DocumentProcessingService
from src.services.index_service import IndexService
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import StorageContext
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore


from src.services.database_service import  DatabaseService, FileService, VectorService


logger = SimpleLogger(__name__)





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
    3. Embedding
    4. Storage
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

    async def initialize(self):
        await self.db.connect()

        self.file_db_service.mongo_client = self.db.mongo_client
        self.file_db_service.mongo_db = self.db.mongo_db
        self.file_db_service.milvus_vector_store = self.db.milvus_vector_store
        
        self.vector_db_service.mongo_client = self.db.mongo_client
        self.vector_db_service.mongo_db = self.db.mongo_db
        self.vector_db_service.milvus_vector_store = self.db.milvus_vector_store
        

    async def cleanup(self):
        await self.db.disconnect()

    async def handle_file_upload(self, file: UploadFile) -> UploadReturn:

        assert self.db.mongo_db is not None, "Connect to the db first"

        return_message = UploadMessageStatus.SUCCESSFUL
        file_id = str(uuid.uuid4())

        existing_files = await self.file_db_service.get_all_file_metadata()
        original_filename = cast(str, file.filename)

        for existing_file in existing_files:
            if (existing_file.original_filename == original_filename and 
                existing_file.status == FileStatus.COMPLETED):
                return_message = UploadMessageStatus.WARNING_DUPLICATE
                logger.info(f"Duplicate file detected: {original_filename}")
                break
        
        try:
            validation_result = await self.file_validation_service.validate_file(file)
            if not validation_result.is_valid:
                raise HTTPException(status_code=400, detail=validation_result.error_message)
            
            storage_path = await self.file_storage_service.save_uploaded_file(file, file_id)
            
            file_doc = FileDocument(
                file_id=file_id,
                filename=f"{file_id}_{file.filename}",
                original_filename=original_filename,
                file_type=cast(str, validation_result.file_type),
                file_size=cast(int, validation_result.file_size),
                status=FileStatus.UPLOADING,
                storage_path=storage_path,
                milvus_collection=self.config.milvus.collection_name
            )
            
            await self.file_db_service.save_file_metadata(file_doc=file_doc)
            
            logger.info(f"File uploaded successfully: {file_id}")
            return UploadReturn(message=return_message, file=file_doc)

        except Exception as e:
            logger.error(f"File upload error: {e}")
            await self.file_db_service.update_file_status(
                file_id, FileStatus.FAILED, error_message=str(e)
            )
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


        
    async def process_file(self, file_id: str) -> bool:
        try:
            file_doc = await self.file_db_service.get_file_metadata(file_id)
            if not file_doc:
                logger.error(f"File not found: {file_id}")
                return False
        
            if file_doc.status == FileStatus.COMPLETED:
                logger.info(f"File already processed: {file_id}")
                return True
            
            await self.file_db_service.update_file_status(file_id, FileStatus.PROCESSING)
            logger.info(f"Processing File: {file_doc.original_filename} with id: {file_doc.file_id}")

            document = self.document_proc_service.from_file_to_document(file_doc.storage_path)
            ctx = self.db.storage_context(file_id)

            index_interface = await self.index_service.create_vector_summary_index(document, ctx) # save nodes and summary index
            nodes = [ node for node in index_interface.vector_index.docstore.docs.values() ]
            chunk_count = await self.vector_db_service.store_embeddings(file_id, nodes)
            await self.file_db_service.update_file_status(
                file_id,
                FileStatus.COMPLETED,
                chunk_count=str(chunk_count)
            )

            logger.info(f"File processed successfully: {file_id} ({chunk_count} chunks)")
            return True
        except Exception as e:
            logger.error(f"Error processing file {file_id}: {e}")
            await self.file_db_service.update_file_status(
                file_id, FileStatus.FAILED, error_message=str(e)
            )
            return False


    async def delete_files(self, file_id: str) -> tuple[bool, int]:
        try:
            file_doc = await self.file_db_service.get_file_metadata(file_id)
            if not file_doc: 
                logger.warning(f"File not found in database: {file_id}")
                return False, 0

            deleted_chunks = await self.vector_db_service.delete_file_vectors(file_id)

            await self.db.storage_context_delete(file_id)

            file_deleted = await self.file_storage_service.delete_file(file_doc.storage_path)

            if not file_deleted:
                logger.warning(f"Could not delete physical file: {file_doc.storage_path}")

            deleted_file_doc = await self.file_db_service.delete_file_metadata(file_id)
            if deleted_file_doc:
                logger.info(f"File deleted successfully: {file_id}")
                return True, deleted_chunks
            else:
                logger.error(f"Failed to delete file metadata: {file_id}")
                return False, 0
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False, 0

    
    async def get_file_status(self, file_id: str) -> FileDocument | None:
        return await self.file_db_service.get_file_metadata(file_id)


    







        
    
    



        

    