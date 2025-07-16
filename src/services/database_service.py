import sys
import os
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_DIR)

from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from llama_index.vector_stores.milvus import MilvusVectorStore, IndexManagement
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode, MetadataFilter, FilterOperator, MetadataFilters
from llama_index.core.schema import Document, BaseNode
from typing import Any
from pymilvus import DataType
from typing import  cast

from src.utils.logger_config import SimpleLogger
from src.config.settings import AppConfig
from src.api.api_models import FileDocument, FileStatus, ChatHistoryDocument



logger = SimpleLogger(__name__)




class DatabaseService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.mongo_client: AsyncIOMotorClient | None = None
        self.mongo_db: AsyncIOMotorDatabase | None = None
        self.milvus_vector_store: MilvusVectorStore | None = None

    async def connect(self):
        await self._connect_mongodb()
        await self._connect_milvus()

    async def disconnect(self):
        if self.mongo_client:
            self.mongo_client.close()

    async def _connect_mongodb(self):
        try:
            self.mongo_client = AsyncIOMotorClient(self.config.mongo.mongodb_uri)
            self.mongo_db = self.mongo_client[self.config.mongo.db_name]
            await self._create_mongo_indexed()
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    async def _create_mongo_indexed(self):
        file_collection = self.mongo_db[self.config.mongo.files_collection]
        chat_collection = self.mongo_db[self.config.mongo.collection_name]

        await file_collection.create_index('file_id', unique=True)
        await file_collection.create_index('status')
        await file_collection.create_index('upload_timestamp')


        await chat_collection.create_index('session_id')
        await chat_collection.create_index('timestamp')

    async def _connect_milvus(self):
        try:
            self.milvus_vector_store = MilvusVectorStore(
                uri=f"{self.config.milvus.host}:{self.config.milvus.port}",
                token=self.config.milvus.token,
                collection_name=self.config.milvus.collection_name,
                overwrite=False,
                upsert_mode=True,
                doc_id_field=self.config.milvus.doc_id_field,
                text_key='text',
                scalar_field_names=['file_id', 'chunk_index'], # The names of the extra scalar fields to be included in the collection schema.
                scalar_field_types=[DataType.VARCHAR, DataType.INT64],
                enable_dense=True,
                dim=self.config.milvus.vector_dim,
                embedding_field="embedding",
                enable_sparse=False,
                index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
                index_config={
                    'index_type': self.config.milvus.index_type,
                    'metric_type': self.config.milvus.metric_type,
                    'params': {'nlist': 1024}
                },
                search_config={
                    'nprobe': 10
                },
                similarity_metric=self.config.milvus.metric_type
            )
            logger.info("Milvus vector initialized and connected successfully!")
        except Exception as e:
            logger.error(f"Milvus connection/initialization failed: {e}")
            raise
    

class FileService(DatabaseService):
    """
    Service for file operations
    """

    async def save_file_metadata(self, file_doc: FileDocument) -> str:
        """Save file metadata to MongoDB

        Args:
            file_doc (FileDocument): the File Document Type

        Returns:
            str: return the unique file_id
        """

        files_collection = self.mongo_db[self.config.mongo.files_collection]
        
        await files_collection.insert_one(file_doc.model_dump())
        
        return file_doc.file_id

    async def get_file_metadata(self, file_id: str) -> FileDocument | None:
        """Get filemetadata by fileId

        Args:
            file_id (str): the file id

        Returns:
            FileDocument | None: If exists then return the FileDocument, else return Nothing
        """
        files_collection = self.mongo_db[self.config.mongo.files_collection]

        file_data = await files_collection.find_one({
            'file_id': file_id
        })
        if file_data:
            return FileDocument(**file_data)
        return None
    
    async def get_all_file_metadata(self) -> list[FileDocument]:
        files_collection = self.mongo_db[self.config.mongo.files_collection]

        cursor = files_collection.find({})  
        file_documents = []

        async for file_data in cursor:
            file_documents.append(FileDocument(**file_data))

        return file_documents

    async def list_files(self, page: int = 1, page_size: int = 50) -> tuple[list[FileDocument], int]:
        files_collection = self.mongo_db[self.config.mongo.files_collection]

        total_count = await files_collection.count_documents({})
        skip = (page - 1) * page_size
        cursor = files_collection.find({}).sort('upload_timestamp', -1).skip(skip).limit(page_size)

        files = []
        async for file_data in cursor:
            files.append(FileDocument(**file_data))
        
        return files, total_count

    async def update_file_status(self, file_id: str, status: FileStatus, error_message: str | None = None, chunk_count: str | None = None) -> bool:

        files_collection = self.mongo_db[self.config.mongo.files_collection]

        update_data: dict[str, Any] = {'status': status}

        if status == FileStatus.PROCESSING:
            update_data["processing_timestamp"] = datetime.now(timezone.utc)
        elif status == FileStatus.COMPLETED:
            update_data["completion_timestamp"] = datetime.now(timezone.utc)
            if chunk_count is not None:
                update_data["chunk_count"] = chunk_count
        elif status == FileStatus.FAILED:
            update_data["error_message"] = error_message
        
        result = await files_collection.update_one(
            {"file_id": file_id},
            {"$set": update_data}
        )


        return result.modified_count > 0
    
    async def delete_file_metadata(self, file_id: str) -> FileDocument | None:
        files_collection = self.mongo_db[self.config.mongo.files_collection]

        file_data = await files_collection.find_one_and_delete({
            'file_id': file_id
        })

        if not file_data:
            return None
        
        await files_collection.delete_one({
            'file_id': file_id
        })

        return FileDocument(**file_data)
    


class VectorService(DatabaseService):
    
    async def store_embeddings(self, file_id: str, chunks_data: list[Document]) -> int:

        if not self.milvus_vector_store:
            raise RuntimeError("MilvusVectorStore is not initialized")

        for idx, doc in enumerate(chunks_data):
            doc.metadata.update({
                self.config.milvus.doc_id_field: file_id,
                "chunk_index": idx,
            })
            doc.id_ = f"{file_id}_{idx}"

        await self.milvus_vector_store.async_add(cast(list[BaseNode], chunks_data))
        
        logger.info(f"Stored {len(chunks_data)} embeddings for file_id: {file_id}")
        return len(chunks_data)
    

    async def search_similar(self, query_embedding: list[float], file_ids: list[str] | None = None, top_k: int = 10):
        if not self.milvus_vector_store:

            raise RuntimeError("MilvusVectorStore is not initialized")

        filters = MetadataFilters(filters=[])

        if file_ids:
            filters.filters.append(MetadataFilter(key=self.config.milvus.doc_id_field, value=file_ids, operator=FilterOperator.IN))
        
        vector_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            filters=filters,
            output_fields=[
                self.config.milvus.doc_id_field,
                "chunk_index",
                "text"
            ]
        )

        results = await self.milvus_vector_store.aquery(vector_query)

        search_results = []
        if not results.nodes:
            logger.info("No similar vectors found.")
            return []


        documents = cast(list[Document], results.nodes)
        for i, node in enumerate(documents):
            search_results.append({
                'file_id': node.metadata.get(self.config.milvus.doc_id_field),
                'text': node.text,
                'chunk_index': node.metadata.get('chunk_index'),
                'score': results.similarities[i] if results.similarities else None
            })
        
        logger.info(f"Found {len(search_results)} similar vectors.")

        return search_results


    async def delete_file_vectors(self, file_id: str) -> int:
        if not self.milvus_vector_store:

            raise RuntimeError("MilvusVectorStore is not initialized")
        
        filters = MetadataFilters(
            filters=[MetadataFilter(key = self.config.milvus.doc_id_field, value=file_id, operator=FilterOperator.EQ)]
        )

        try:
            await self.milvus_vector_store.adelete_nodes(filters=filters)
            logger.info(f"Successfully initiated deletion for vectors of file_id: {file_id}")
        
            return 1
        except Exception as e:
            logger.error(f"Failed to delete vectors for file_id: {file_id}: {e}")
            return 0
    




     
class ChatService(DatabaseService):
    """
    Service for chat operations
    """

    async def save_chat_history(
        self,
        session_id: str,
        query: str,
        response: str,
        file_ids: list[str],
        sources: list[dict[str, Any]]
    ):
        chat_collection = self.mongo_db[self.config.mongo.collection_name]

        chat_doc = ChatHistoryDocument(
            session_id=session_id,
            query=query,
            response=response,
            file_ids=file_ids,
            sources=sources
        )

        await chat_collection.insert_one(chat_doc)

    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> list[ChatHistoryDocument]:
        chat_collection = self.mongo_db[self.config.mongo.collection_name]

        cursor = chat_collection.find(
            {'session_id': session_id}
        ).sort('timestamp', -1).limit(limit)

        history = []
        async for chat in cursor:
            history.append(ChatHistoryDocument(**chat))
        
        return history



    



