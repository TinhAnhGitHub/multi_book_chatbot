from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone
from enum import Enum
import uuid


class FileStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SupportedFileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1)
    file_ids: list[str] | None = Field(default=None, description="Specific files to query")
    session_id: list[str] | None = Field(default=None)



class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    file_size: int
    status: FileStatus
    upload_timestamp: datetime
    message: str


class FileMetadata(BaseModel):
    file_id: str
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: FileStatus
    upload_timestamp: datetime
    processing_timestamp: datetime | None = None
    completion_timestamp: datetime | None = None
    error_message: str | None = None
    chunk_count: int | None = None
    storage_path: str
    milvus_collection: str | None = None



class FileListResponse(BaseModel):
    files: list[FileMetadata]
    total_count: int
    page: int
    page_size: int

class FileDeleteResponse(BaseModel):
    file_id: str
    message: str
    deleted_chunks: int


class ChatResponse(BaseModel):
    response: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str
    timestamp: datetime
    
class ErrorResponse(BaseModel):
    error: str
    details: str | None = None
    error_code: str | None = None


class FileValidation(BaseModel):
    is_valid: bool
    file_type: str | None = None
    error_message: str | None = None
    file_size: int | None = None


class FileDocument(BaseModel):
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: FileStatus = FileStatus.UPLOADING
    upload_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_timestamp: datetime | None = None
    completion_timestamp: datetime | None = None
    error_message: str | None = None
    chunk_count: int | None = None # # Number of chunks the file is divided into
    storage_path: str   # # Path where the file is stored
    milvus_collection: str | None = None #  Name of the Milvus collection associated with the file
    metadata: dict[str, Any] = Field(default_factory=dict)

class ChatHistoryDocument(BaseModel):
    session_id: str
    query: str
    response: str
    file_ids: list[str] = Field(default_factory=list) # list of retrieved file ids
    timestamp: datetime= Field(default_factory=lambda: datetime.now(timezone.utc)) # 
    sources: list[dict[str, Any]] = Field(default_factory=list) # List of sources used to generate the response
    




