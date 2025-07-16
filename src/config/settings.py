from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
import os 


load_dotenv()
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)



class GeminiConfig(BaseSettings):
    api_key: str = Field(..., alias="GOOGLE_API_KEY")
    google_model_name: str = Field(..., alias="GOOGLE_MODEL_NAME")  
    google_embedding_model: str = Field(..., alias="GOOGLE_EMBEDDING_MODEL")




class CohereConfig(BaseSettings):
    api_key: str = Field(..., alias="COHERE_API_KEY")
    model: str = Field("rerank-english-v3.0", alias="COHERE_MODEL")
    top_n: int = Field(5, alias="COHERE_TOP_N")


class MongoConfig(BaseSettings):
    mongodb_uri: str = Field(..., alias="MONGODB_URI")
    db_name: str = Field("chatbot_memory_book", alias="MONGODB_DB_NAME")
    collection_name: str = Field("chat_history", alias="MONGODB_COLLECTION_NAME")
    files_collection : str  = Field("file_collection")



class RedisConfig(BaseSettings):
    host: str = Field("localhost", alias="REDIS_HOST")
    port: int = Field(6379, alias="REDIS_PORT")
    db: int = Field(0, alias="REDIS_DB")
    password: str = Field("", alias="REDIS_PASSWORD")
    session_timeout: int = Field(3600, alias="REDIS_SESSION_TIMEOUT")


class MinIOConfig(BaseSettings):
    endpoint: str = Field("localhost:9000", alias="MINIO_ENDPOINT")
    access_key: str = Field("minioadmin", alias="MINIO_ACCESS_KEY")
    secret_key: str = Field("minioadmin", alias="MINIO_SECRET_KEY")
    bucket_name: str = Field("chatbot-files", alias="MINIO_BUCKET_NAME")
    secure: bool = Field(False, alias="MINIO_SECURE")



    



class MilvusConfig(BaseSettings):
    host: str = Field(..., alias="MILVUS_HOST")
    port: str = Field(..., alias='MILVUS_PORT')
    token: str = Field("")
    doc_id_field: str = Field("file_id")
    vector_dim: int = Field(3072)
    grpc_port: str = Field(..., alias="MILVUS_GRPC_PORT")
    http_port: str = Field(..., alias="MILVUS_HTTP_PORT")
    collection_name: str = Field(..., alias="MILVUS_COLLECTION_NAME")
    index_type: str = Field(..., alias="MILVUS_INDEX_TYPE")
    metric_type: str = Field(..., alias="MILVUS_METRIC_TYPE")




class AgentConfig(BaseSettings):
    
    similarity_top_k: int = 10
    rerank_top_n: int = 5

    system_prompt: str = """
        You are an expert Q&A system for a library of books.
        You have access to a conversation history and a set of tools, each representing a specific book.

        Here's how you should answer:
        1. First, review the conversation history. If the user's query is a follow-up or can be answered based on the previous turns, answer it directly without using any tools.
        2. If the query requires new information from the books, use the provided tools to find the answer.
        3. You must always use the tools for new questions about the books. Do not rely on your prior knowledge.

        Always provide a clear and concise answer based on the retrieved information.
    """



class FileConfig(BaseSettings):
    max_file_size_mb: int = Field(50)
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(50)
    allowed_file_types: str = Field('pdf,docx,txt,md')




class ChatConfig(BaseSettings):
    token_limit: int = 40_000
    chat_history_token_ratio: float = 0.7
    token_flush_size: int = 3_000


class LoggingConfig(BaseSettings):
    log_level: str = Field("DEBUG", alias="LOG_LEVEL")
    log_file: str = Field("logs/app.log", alias="LOG_FILE")
    environment: str = Field("development", alias="ENVIRONMENT")



class AppConfig:
    def __init__(self):
        self.gemini = GeminiConfig()
        self.milvus = MilvusConfig()
        self.mongo = MongoConfig()
        self.redis = RedisConfig()
        self.minio = MinIOConfig()
        self.agent = AgentConfig()
        self.file_config = FileConfig()
        self.chat_config = ChatConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        
        self.data_path = os.path.join(ROOT_DIR, 'data')
        self.books_path = os.path.join(self.data_path, 'books')
        self.indexes_path = os.path.join(self.data_path, 'indexes')
        self.temp_path = os.path.join(self.data_path, 'temp')
        self.logs_path = os.path.join(ROOT_DIR, 'logs')
        
        self.max_file_size_mb = self.file_config.max_file_size_mb
        self.chunk_size = self.file_config.chunk_size
        self.chunk_overlap = self.file_config.chunk_overlap

def get_config() -> AppConfig:
    config = AppConfig()
    
    os.makedirs(config.books_path, exist_ok=True)
    os.makedirs(config.indexes_path, exist_ok=True)
    os.makedirs(config.temp_path, exist_ok=True)
    os.makedirs(config.logs_path, exist_ok=True)
    
    return config