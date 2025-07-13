from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms  import LLM
from llama_index.core.tools import QueryEngineTool

def build_book_agent(tools: list[QueryEngineTool], book_title: str, llm: LLM) -> FunctionAgent:
    system_prompt = f"""
    You are a specialized agent designed to answer queries about the book '{book_title}'.
    You must always use at least one of the tools provided when answering the questions.
    Do NOT rely on any prior knowledge you have about this book.
    Base your answers only on the information retrieved from the tools.
    """

    return FunctionAgent(
        tools = tools,
        llm = llm,
        system_prompt=system_prompt,
        verbose=True
    )

