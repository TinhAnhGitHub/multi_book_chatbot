from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.objects import ObjectIndex, ObjectRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.tools import FunctionTool, BaseTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.objects import SimpleObjectNodeMapping
from llama_index.core.llms.llm import LLM
from ..config.settings import AppConfig
from ..utils.logger_config import SimpleLogger


logger = SimpleLogger(__name__)



class CustomObjectRetriever(ObjectRetriever):
    def __init__(
        self,
        retriever: BaseRetriever,
        object_node_mapping: SimpleObjectNodeMapping,
        llm: LLM,
        node_postprocessors=None,
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm 
        self._reranker = reranker
        self._node_postprocessors = node_postprocessors or []

    def retrieve(self, str_or_query_bundle: QueryBundle) -> list[BaseTool]:
        nodes = self._retriever.retrieve(str_or_query_bundle)
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(
                nodes, query_bundle=str_or_query_bundle
            )
        reranked_nodes = self._reranker.postprocess_nodes(
            nodes=nodes,
            query_bundle=str_or_query_bundle
        )

        tools = [self._object_node_mapping.from_node(n.node) for n in reranked_nodes]

        sub_agent = FunctionAgent(
            name="compare_tool",
            description=f"""\
            Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
            tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
            """,
            tools=tools,
            llm=self._llm,
            system_prompt="""You are an expert at comparing documents.
            Given a query, use the tools provided to compare the documents and return a summary of the results.
            """
        )

        async def query_sub_agent(query: str) -> str:
            response = await sub_agent.run(query)
            return str(response)

        sub_question_tool = FunctionTool.from_defaults(
            query_sub_agent,
            name=sub_agent.name,
            description=sub_agent.description
        )

        return tools + [sub_question_tool]
    


def create_tool_retriever(
    all_tools: list[BaseTool], config: AppConfig, llm: LLM
) -> CustomObjectRetriever:
    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex
    )

    vector_node_retriever = obj_index.as_node_retriever(
        similarity_top_k=config.agent.similarity_top_k,
    )

    # reranker = CohereRerank(
    #     api_key=config.cohere.api_key,
    #     top_n=config.agent.rerank_top_n,
    #     model=config.cohere.model
    # )

    # logger.debug(f"Initialized reranker: {reranker}")

    return CustomObjectRetriever(
        retriever=vector_node_retriever,
        object_node_mapping=obj_index.object_node_mapping,
        llm=llm,
        # reranker=reranker
    )





    
