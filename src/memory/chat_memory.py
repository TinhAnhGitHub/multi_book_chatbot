from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.core.memory.types import BaseChatStoreMemory
from llama_index.storage.chat_store.mongo import MongoChatStore


from ..utils.logger_config import SimpleLogger
from ..config.settings import MongoConfig

logger = SimpleLogger(__name__)


def get_chat_memory(config: MongoConfig) -> BaseChatStore | BaseChatStoreMemory:
    if config.mongodb_uri:
        try:
            logger.info(f"Connecting to MongoDB for chat history at {config.db_name}...")
            return MongoChatStore(
                mongo_uri=config.mongodb_uri,
                db_name=config.db_name,
                collection_name=config.collection_name
            )
        except Exception as e:
            logger.warning(f"Warning: Failed to connect to MongoDB: {e}. Falling back to in-memory chat history.")
    
    logger.info("Using in-memory chat history.")
    return ChatMemoryBuffer.from_defaults()






