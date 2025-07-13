import streamlit as st  
import requests 
import uuid

st.set_page_config(
    page_title="Multi-Book AI Chatbot",
    layout="wide"
)

FASTAPI_URL = "http://localhost:8000/chat"


with st.sidebar:
    st.markdown("## Session Controls")

    if st.button("New Chat"):
        st.session_state.user_id = str(uuid.uuid4())
        st.session_state.messages = [
            {
                "role": "assistant", "content": "Hello! I am an AI assistant with knowledge of several books. How can I help you today?"
            }
        ]
    
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am an AI assistant with knowledge of several books. How can I help you today?"}
        ]



if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am an AI assistant with knowledge of several books. How can I help you today?"}
    ]

st.title("📚 Multi-Book AI Chatbot")
st.caption("A Streamlit interface for the RAG agent powered by LlamaIndex and FastAPI")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def get_api_response_stream(query: str, user_id: str):
    try:
        response = requests.post(
             FASTAPI_URL,
            json={"query": query, "user_id": user_id},
            stream=True
        )
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            yield chunk

    except Exception as e:
        yield f"An error occur: {e}"



if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response_stream = get_api_response_stream(prompt, st.session_state.user_id)
        full_response = ""
        for chunk in response_stream:
            st.write(chunk)
            full_response += chunk
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})