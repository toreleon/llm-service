from typing import List, Optional
from pydantic import BaseModel, validator

class BloomRequest(BaseModel):
    text: str
    temperature: Optional[float] = 0.7
    max_length: Optional[int] = 100
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 0
    num_return_sequences: Optional[int] = 1
    diversity_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0

@validator('text')
def check_text(cls, v):
    if v is None:
        raise ValueError('text must not be None')
    return v

class BloomResponse(BaseModel):
    generated_text: str

class BloomEmbeddingRequest(BaseModel):
    text: str

class BloomEmbeddingResponse(BaseModel):
    embedding: List[float]
