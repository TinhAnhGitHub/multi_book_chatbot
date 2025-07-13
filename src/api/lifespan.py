from contextlib import asynccontextmanager
import os
from typing import Callable
from fastapi import FastAPI
from dataclasses import dataclass, field
import pickle

from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.base.llms.types import TextBlock
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core import Document
from llama_index.core.memory import Memory, StaticMemoryBlock, FactExtractionMemoryBlock

import os
import sys

ROOT_DIR =  os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)

sys.path.insert(0, ROOT_DIR)

from src.utils.logger_config import SimpleLogger
from src.config.settings import get_config, AppConfig
from src.utils.file_processor import load_books
from src.utils.index_builder import build_or_load_indexes, get_book_summary
from src.tools.query_tools import create_book_query_tools
from src.agents.book_agent import build_book_agent


@dataclass
class AppState:
    config: AppConfig | None = None
    llm: AzureOpenAI | None = None
    embed_model: AzureOpenAIEmbedding | None = None
    all_tools: list[FunctionTool] = field(default_factory=list)


app_state = AppState()
logger = SimpleLogger(__name__)




def get_memory(session_user_id: str, config: AppConfig) -> Memory:

    static_content = TextBlock(
        text="User is a vivid reader"
    )
    llm = app_state.llm

    blocks = [
         StaticMemoryBlock(
            name='user_profile',
            static_content=[static_content],
            priority=0
        ),
        FactExtractionMemoryBlock(
            name="facts",
                llm=llm,
                max_facts=50,
                priority=1,
        ),
    ]

    async_database_uri = f"sqlite+aiosqlite:///{config.sql_lite.sqlite_db_path}"


    memory = Memory.from_defaults(
        session_id=session_user_id,
        token_limit=config.chat_config.token_limit,
        chat_history_token_ratio=config.chat_config.chat_history_token_ratio,
        token_flush_size=config.chat_config.token_flush_size,
        memory_blocks=blocks,
        insert_method="system",
        async_database_uri=async_database_uri
        
    )
    return memory

def get_agent_tool_callable(agent: FunctionAgent) -> Callable:
    async def query_agent(query: str) -> str:
        response = await agent.run(query)
        return str(response)

    return query_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Loads models and builds agents on startup.
    """

    logger.info("--- Application StartUp ---")


    config = get_config()
    app_state.config = config
    
    llm = AzureOpenAI(
        model=config.azure_openai.deployment_name,
        deployment_name=config.azure_openai.deployment_name,
        api_key=config.azure_openai.api_key,
        azure_endpoint=config.azure_openai.azure_endpoint,
        api_version=config.azure_openai.api_version
    )

    app_state.llm = llm
    
    embed_model = AzureOpenAIEmbedding(
        model=config.azure_openai.embedding_deployment_name,
        deployment_name=config.azure_openai.embedding_deployment_name,
        api_key=config.azure_openai.api_key,
        azure_endpoint=config.azure_openai.azure_endpoint,
        api_version=config.azure_openai.api_version,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512   

    new_docs, new_bases, old_bases = load_books(
        config.books_path, config.indexes_path
    )


    if not new_docs and not old_bases:
        logger.warning("No books at all found on startup.")
    
    all_tools = []
    
    
    async def setup_book(base: str, doc: Document | None):

        if doc is None:
            vector_index, _ = build_or_load_indexes(doc, config, base)
            sumary_index_path = os.path.join(
                config.data_path, "indexes", f"{base.split('.')[0]}_summary.pkl"
            )
            with open(sumary_index_path, 'rb') as f:
                summary = pickle.load(f)    
           
            tools = create_book_query_tools(
                vector_index=vector_index,
                summary_index=None,
                file_base=base,
            )

            agent = build_book_agent(tools, base, llm)
            fn = get_agent_tool_callable(agent)
            sanitized_name = base.split('.')[0].replace(' ', '_').replace('-', '_')
            return FunctionTool.from_defaults(
                fn, name=f"tool_{sanitized_name}", description=summary
            )


        vector_index, summary_index = build_or_load_indexes(doc, config)
        tools = create_book_query_tools(
            vector_index=vector_index,
            summary_index=summary_index,
            file_base=base,
        )
        agent = build_book_agent(tools, doc.metadata.get("title", base), llm)

        summary_pkl = os.path.join(
            config.data_path, "indexes", f"{base}_summary.pkl"
        )
        summary = await get_book_summary(summary_index, summary_pkl, llm)

        fn = get_agent_tool_callable(agent)
        sanitized_name = base.split('.')[0].replace(' ', '_').replace('-', '_')
        return FunctionTool.from_defaults(
            fn, name=f"tool_{sanitized_name}", description=summary
        )

    for doc in new_docs:
        base = doc.metadata["file_name"].rsplit(".", 1)[0]
        all_tools.append(await setup_book(base, doc))

    for base in old_bases:
        all_tools.append(await setup_book(base, None))

    app_state.all_tools = all_tools
    logger.info(f"--- Startup Complete. {len(all_tools)} book tools ready. ---")
    yield 
    logger.info("--- Application Shutdown ---")