import sys
import os
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_DIR)

from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from src.utils.logger_config import SimpleLogger
from src.config.settings import AppConfig


logger = SimpleLogger(__name__)



class IndexInterface:
    def __init__(self, vector_index: VectorStoreIndex, summary_index: SummaryIndex):
        self.vector_index = vector_index
        self.summary_index = summary_index
    
    

class IndexService:
    """
    Service for Indexing
    """

    def __init__(self, config: AppConfig, embed_model: GoogleGenAIEmbedding):
        self.config = config
        self.embed_model = embed_model
        self.node_parser = SentenceSplitter(
            chunk_overlap=getattr(config, 'chunk_overlap', 50),
            chunk_size=getattr(config, 'chunk_size', 512)
        )
    
    async def create_vector_summary_index(self, doc: Document) -> IndexInterface:
        nodes = self.node_parser.get_nodes_from_documents([doc])

        vector_index = VectorStoreIndex(
            nodes=nodes,
            use_async=True,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        summary_index = SummaryIndex(nodes=nodes, show_progress=True)

        return IndexInterface(
            vector_index=vector_index,
            summary_index=summary_index
        )

