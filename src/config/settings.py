from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
import os 


load_dotenv()
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)

class AzureOpenAIConfig(BaseSettings):
    api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    api_version: str = Field(..., alias="AZURE_OPENAI_API_VERSION")
    azure_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    deployment_name: str = Field(..., alias="AZURE_OPENAI_DEPLOYMENT")
    embedding_deployment_name: str = Field(..., alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
   
    embed_batch_size: int = Field(512, alias="AZURE_OPENAI_EMBED_BATCH_SIZE")

class CohereConfig(BaseSettings):
    api_key: str = Field(..., alias="COHERE_API_KEY")
    model: str = Field("rerank-english-v3.0", alias="COHERE_MODEL")
    top_n: int = Field(5, alias="COHERE_TOP_N")


class MongoConfig(BaseSettings):
    mongodb_uri: str = Field("mongodb://localhost:27017", alias="MONGODB_URI")
    db_name: str = Field("chatbot_memory_book", alias="MONGODB_DB_NAME")
    collection_name: str = Field("chat_history", alias="MONGODB_COLLECTION_NAME")


class SQLiteConfig(BaseSettings):
    sqlite_db_path: str = Field(os.path.join(ROOT_DIR, 'data', 'chatbot_memory_book.db'), alias="SQLITE_DB_PATH")
    table_name: str = Field("chat_history", alias="SQLITE_TABLE_NAME")

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




class ChatConfig(BaseSettings):
    token_limit: int = 40_000
    chat_history_token_ratio: float = 0.7
    token_flush_size: int = 3_000



class AppConfig:
    def __init__(self):
        self.azure_openai = AzureOpenAIConfig()
        self.cohere = CohereConfig()
        self.mongo = MongoConfig()
        self.sql_lite = SQLiteConfig()
        self.agent = AgentConfig()
        self.chat_config = ChatConfig()
        self.data_path = os.path.join(ROOT_DIR, 'data')
        self.books_path = os.path.join(self.data_path, 'books')
        self.indexes_path = os.path.join(self.data_path, 'indexes')
    

def get_config() -> AppConfig:
    config = AppConfig()
    os.makedirs(config.books_path, exist_ok=True)
    os.makedirs(config.indexes_path, exist_ok=True)
    return config


        

