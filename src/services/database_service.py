import sys
import os
from pathlib import Path
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
from llama_index.core import DocumentSummaryIndex
from llama_index.core import load_index_from_storage


from pymilvus import DataType



from src.utils.logger_config import SimpleLogger
from src.config.settings import AppConfig
from src.api.api_models import FileDocument, FileStatus, ChatHistoryDocument, ChatResponse



logger = SimpleLogger(__name__)


class VectorStoreIndexInitialization:
    """
    This class will handling the intialization of the vector index
    1. Branch new vector index with raw storage context if the persist data is empty
    2. load the vector index from the storage context(if they are persist)
    """

    def __init__(self, config: AppConfig , nodes: list[BaseNode] | None = None):
        self.config = config

        self.vector_index = self._initialize_vector_store_index(nodes)
        
    
    def _initialize_vector_store_index(self, nodes: list[BaseNode] | None) -> VectorStoreIndex:
        persist_folder = self.config.storage_persist_dir

        if not os.path.exists(persist_folder):
            Path(persist_folder).mkdir(parents=True, exist_ok=True)

            vector_store = MilvusVectorStore(
                uri=f"{self.config.milvus.host}:{self.config.milvus.port}",
                token=self.config.milvus.token,
                collection_name=self.config.milvus.collection_name,
                overwrite=False,
                upsert_mode=True,
                doc_id_field=self.config.milvus.doc_id_field,
                text_key='text',
                scalar_field_names=['file_id', 'chunk_index'], 
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

            storage_context = StorageContext.from_defaults(
                docstore=MongoDocumentStore.from_uri(
                    uri=self.config.mongo.mongodb_uri,
                    db_name=self.config.mongo.db_name,
                    namespace=self.config.mongo.file_collection_namespace
                ),
                index_store=MongoIndexStore.from_uri(
                    uri=self.config.mongo.mongodb_uri,
                    db_name=self.config.mongo.db_name,
                    namespace=self.config.mongo.index_store_namespace
                ),
                vector_store=vector_store
            )


            
            vector_index = VectorStoreIndex(storage_context=storage_context, nodes=nodes)
            vector_index.storage_context.persist(persist_dir=persist_folder)
            return vector_index
        
        storage_context = StorageContext.from_defaults(persist_dir=persist_folder)
        vector_index: VectorStoreIndex = cast(VectorStoreIndex, load_index_from_storage(storage_context))

        return vector_index




class DocumentSummaryIndexInitialization:
    









