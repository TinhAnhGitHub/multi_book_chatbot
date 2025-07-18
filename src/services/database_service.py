import sys
import os
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_DIR)

from datetime import datetime, timezone
from typing import Any, cast

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from llama_index.vector_stores.milvus import MilvusVectorStore, IndexManagement
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    MetadataFilter,
    FilterOperator,
    MetadataFilters,
)
from llama_index.core.schema import Document, BaseNode
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.storage.chat_store.mongo import MongoChatStore
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from pymilvus import DataType



from src.utils.logger_config import SimpleLogger
from src.config.settings import AppConfig
from src.api.api_models import FileDocument, FileStatus, ChatHistoryDocument, ChatResponse



logger = SimpleLogger(__name__)




class DatabaseService:
    """
    Thin wrapper client and Milvus vector store
    1. Owns the MongoClient and Milvus vector store
    2. exposes ready-made StorageContext Objects for file-level indices


    """
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
        
    def storage_context(self, file_id: str) -> StorageContext:
        """
        Return a StorageContext wired to mongo backend dockstore & index store with already initialized milvus vector store
        """
        return StorageContext.from_defaults(
            docstore=MongoDocumentStore.from_uri(
                uri=self.config.mongo.mongodb_uri,
                db_name=self.config.mongo.db_name,
                namespace=file_id
            ),
        )
    

    async def storage_context_delete(self, file_id: str):
        docstore =  MongoDocumentStore.from_uri(
                uri=self.config.mongo.mongodb_uri,
                db_name=self.config.mongo.db_name,
                namespace=file_id
            )
        
        all_nodes_dict = docstore.docs
        node_ids  = list(all_nodes_dict.keys())
        for node_id in node_ids:
            await docstore.adelete_document(node_id)

def get_nodes_from_context(self, file_id: str):
    docstore =  MongoDocumentStore.from_uri(
            uri=self.config.mongo.mongodb_uri,
            db_name=self.config.mongo.db_name,
            namespace=file_id
        )

    return docstore.docs        

        
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
    
    async def store_embeddings(self, file_id: str, chunks_data: list[BaseNode]) -> int:

        if not self.milvus_vector_store:
            raise RuntimeError("MilvusVectorStore is not initialized")

        for idx, doc in enumerate(chunks_data):
            doc.metadata.update({
                self.config.milvus.doc_id_field: file_id,
                "chunk_index": idx,
            })
            doc.id_ = f"{file_id}_{idx}"

        await self.milvus_vector_store.async_add(chunks_data)
        
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
    



def doc_to_message(chat_his_doc: ChatHistoryDocument) -> list[ChatMessage]:
    msgs = [
        ChatMessage(role=MessageRole.USER, content=chat_his_doc.query),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content=chat_his_doc.response,
            additional_kwargs={
                'file_ids': chat_his_doc.file_ids,
                'sources': chat_his_doc.sources,
                'timestamp': chat_his_doc.timestamp.isoformat(),
                'query': chat_his_doc.query,
                'response': chat_his_doc.response
            }
        ),
    ]
    return msgs
    
class ChatService: 
    def __init__(self, config: AppConfig):
        self.mongo_store = MongoChatStore(
            mongo_uri=config.mongo.mongodb_uri,
            db_name=config.mongo.db_name,
            collection_name=config.mongo.collection_name
        )

        self.redis_chat = RedisChatStore(redis_url=config.redis.redis_url(), ttl=6000)
    

    def memory(self, session_id: str) -> ChatMemoryBuffer:
        return ChatMemoryBuffer.from_defaults(
            token_limit=4000,
            chat_store=self.redis_chat,
            chat_store_key=session_id
        )
    async def load_session_into_redis(self, session_id: str, max_turns: int = 50 )-> None:
        msgs = await self.mongo_store.aget_messages(session_id)
        await self.redis_chat.aset_messages(session_id, msgs)
    
    async def save_chat_history(
        self,
        chat_his_doc: ChatHistoryDocument
    ) -> None:
        msgs = doc_to_message(chat_his_doc)

        for msg in msgs:
            await self.mongo_store.async_add_message(chat_his_doc.session_id, msg)
            await self.redis_chat.async_add_message(chat_his_doc.session_id, msg)

    async def get_chat_history(
        self, session_id: str
    ) -> list[ChatHistoryDocument]:
        raw_msgs  = await self.mongo_store.aget_messages(session_id)

        turns: list[ChatHistoryDocument] = []
        for i in range(0, len(raw_msgs) - 1, 2):
            user_msg = raw_msgs[i]
            assistant_msg = raw_msgs[i + 1]

            turns.append(
                ChatHistoryDocument(
                    session_id=session_id,
                    query=user_msg.content or "",
                    response=assistant_msg.content or "",
                    file_ids=assistant_msg.additional_kwargs.get("file_ids", []),
                    sources=assistant_msg.additional_kwargs.get("sources", []),
                    timestamp=datetime.fromisoformat(
                        assistant_msg.additional_kwargs.get("timestamp")
                    ),
                )
            )
        return turns[::-1]





