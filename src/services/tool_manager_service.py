from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex, SummaryIndex

import sys
import os
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_DIR)


from src.utils.logger_config import SimpleLogger
from src.config.settings import AppConfig
from src.services.database_service import get_nodes_from_context



QA_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "provide a comprehensive and detailed answer to the question: {query_str}\n"
    "Include specific examples, numerical values, and technical details from the text. "
    "Explore the topic in depth and cover multiple perspectives if available."
)



class ToolManagerService:
    """
    Service for tool management
    1. Each file is a tool
    2. Update tools on deman
    """

    def __init__(self):
        self.fileid2tools: dict[str, list[QueryEngineTool]] = {}
    

    def update_file_ids(self, current_file_ids: list[str]):
        
        new_file_ids = [
            file_id for file_id in current_file_ids if file_id not in list(self.fileid2tools.keys())
        ]

        delete_file_ids = [
            file_id for file_id in list(self.fileid2tools.keys()) if file_id not in current_file_ids
        ]

        for del_file_id in delete_file_ids:
            del self.fileid2tools[del_file_id]
        
        return new_file_ids
    
    def add_new_tools(self, new_file_ids: list[str]) -> list[QueryEngineTool]:
        

        
        
