from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core.response import Response

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../..'
    )
)
sys.path.insert(0, ROOT_DIR)

from src.api.lifespan import lifespan, app_state, get_memory
from src.api.api_models import ChatRequest
from src.agents.orchestration_agent import build_orchestration_agent
from src.utils.logger_config import SimpleLogger

logger = SimpleLogger(__name__)

app = FastAPI(
    title="Multi-book Agent Chatbot",
    description="An API for chatting with an agent that has knowledge of multiple books",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  
    redoc_url=None,
)

@app.post('/chat', response_class=StreamingResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query
    user_id = request.user_id
    config = app_state.config
    llm = app_state.llm
    all_tools = app_state.all_tools
    logger.debug(f"{all_tools}")

    
    memory = get_memory(session_user_id=user_id, config=config)


    func_agent : FunctionAgent = build_orchestration_agent(
        config=config,
        llm=llm,
        all_tools=all_tools
    )

    response = await func_agent.run(query, memory=memory)

    formatted_response = Response(response=str(response))

    pprint_response(formatted_response, show_source=True)

    async def stream_response():
        yield str(response)
    

                
    return StreamingResponse(stream_response(), media_type="text/plain")

@app.get('/')
def root():
    return {
        'message': 'Multi-book Agent API is running. POST to /chat for interaction'
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8000)