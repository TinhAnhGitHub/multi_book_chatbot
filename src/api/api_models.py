from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        description="The user's query to the chatbot"
    )
    user_id: str = Field(
        default="default-user",
        description="A unique identifier for the user to maintain separate chat histories"
    )




