from typing import List
from llama_index.core.agent.workflow import ReActAgent, FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import Memory
from llama_index.core.memory import StaticMemoryBlock, FactExtractionMemoryBlock
from llama_index.core.storage.chat_store.base import BaseChatStore
from llama_index.core.tools import BaseTool
from ..config.settings import AppConfig
from ..tools.retrieval_tools import create_tool_retriever


def build_orchestration_agent(
    config: AppConfig,
    llm: LLM,
    all_tools: list[BaseTool],
) -> FunctionAgent:
    """
    Build the main orchestration using the ReACT agent and workflow API
    """
    tool_retriever = create_tool_retriever(all_tools, config, llm)

    agent = FunctionAgent(
        tool_retriever=tool_retriever,
        system_prompt=config.agent.system_prompt,
        llm=llm,
        verbose=True,
    )

    return agent 

