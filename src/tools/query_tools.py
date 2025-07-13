from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate


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



def create_book_query_tools(
    vector_index: VectorStoreIndex,
    summary_index: SummaryIndex | None,
    file_base: str
):
    
    returned_tools = []
    vector_query_engine = vector_index.as_query_engine()

    vector_response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        verbose=True,
        use_async=True,
        text_qa_template=PromptTemplate(QA_TEMPLATE)
    )

    vector_query_engine = vector_index.as_query_engine(
        response_synthesizer=vector_response_synthesizer,
        similarity_top_k=8,  
        node_postprocessors=[],  
        llm_kwargs={"temperature": 0.5},
    )



    sanitized_name = file_base.split('.')[0].replace(' ', '_').replace('-', '_')
    returned_tools.append(
        QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            name=f"vector_tool_{sanitized_name}",
            description="Useful for questions that related to retrieve specified facts, details, information or snippets from the book"
        )
    )


    if summary_index:
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
        returned_tools.append(
            QueryEngineTool.from_defaults(
                query_engine=summary_query_engine,
                name=f"summary_tool_{sanitized_name}",
                description="Useful for high-level summarization questions about the entire book.",
            )
        )
    
    return returned_tools

