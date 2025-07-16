import os
import pickle
from pathlib import Path
from typing import List

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    SummaryIndex,
    load_index_from_storage
)

from llama_index.core.indices.base import BaseIndex 
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, BaseNode
from llama_index.llms.azure_openai import AzureOpenAI

from ..config.settings import AppConfig
from ..utils.logger_config import SimpleLogger

logger = SimpleLogger(__name__)


def build_or_load_indexes(
    doc: Document | None,
    config: AppConfig,
    filebase: str | None = None
) -> tuple[VectorStoreIndex, SummaryIndex | None]:

    if doc:
        file_base = doc.metadata['file_name'] 
        vi_out_path = os.path.join(config.indexes_path, file_base.split('.')[0])
        logger.debug(f"Processing: {file_base} | {vi_out_path}")

        node_parser = SentenceSplitter()
        nodes = node_parser.get_nodes_from_documents([doc])

        logger.info(f"Building vector index for {file_base}...")
        vector_index  =  VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=vi_out_path)

        summary_index = SummaryIndex(nodes)
        return vector_index, summary_index
    else:
        if not filebase:
            raise ValueError("filebase is required when doc is None.")
        vi_out_path = os.path.join(config.indexes_path, filebase.split('.')[0])
        logger.info(f"Loading vector index for {filebase} from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=vi_out_path)
        vector_index = load_index_from_storage(storage_context)

        return vector_index, None
        

    

    # file_base = doc.metadata['file_name'] if doc is not None else filebase
    # vi_out_path = os.path.join(config.indexes_path, file_base.split('.')[0])

    # logger.debug(f"Processing: {file_base} | {vi_out_path}")

    # node_parser = SentenceSplitter()
    # nodes = node_parser.get_nodes_from_documents([doc])

    # if not os.path.exists(vi_out_path):
    #     logger.info(f"Building vector index for {file_base}...")
    #     vector_index  =  VectorStoreIndex(nodes)
    #     vector_index.storage_context.persist(persist_dir=vi_out_path)

    # else:
    #     logger.info(f"Loading vector index for {file_base} from storage...")
    #     storage_context = StorageContext.from_defaults(persist_dir=vi_out_path)
    #     vector_index = load_index_from_storage(storage_context)
    

    # summary_index = SummaryIndex(nodes)

    # return vector_index, summary_index, nodes



async def get_book_summary(summary_index: SummaryIndex  | None, summary_path: str, llm: AzureOpenAI) -> str:


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





