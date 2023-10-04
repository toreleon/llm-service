from typing import List, Optional
from pydantic import BaseModel, validator

class BloomRequest(BaseModel):
    text: str = None
    temperature: Optional[float] = 0.7
    max_length: Optional[int] = 100
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 0
    num_return_sequences: Optional[int] = 1
    diversity_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0

class BloomResponse(BaseModel):
    date: str
    generated_text: str
    prompt_tokens: int

class BloomErrorResponse(BaseModel):
    status: int
    error: str

@validator('text')
def check_text(cls, v):
    if v is None:
        raise ValueError('text must not be None')
    return v

@validator('temperature')
def check_temperature(cls, v):
    if v is None:
        raise ValueError('temperature must not be None')
    if v < 0.0 or v > 2.0:
        raise ValueError('temperature must be between 0.0 and 2.0')
    return v

@validator('max_length')
def check_max_length(cls, v):
    if v is None:
        raise ValueError('max_length must not be None')
    if v < 0:
        raise ValueError('max_length must be greater than 0')
    return v

@validator('top_p')
def check_top_p(cls, v):
    if v is None:
        raise ValueError('top_p must not be None')
    if v < 0.0 or v > 1.0:
        raise ValueError('top_p must be between 0.0 and 1.0')
    return v

@validator('top_k')
def check_top_k(cls, v):
    if v is None:
        return BloomErrorResponse(status=400, error="top_k must not be None")
    if v < 0:
        raise ValueError('top_k must be greater than 0')
    return v

@validator('num_return_sequences')
def check_num_return_sequences(cls, v):
    if v is None:
        raise ValueError('num_return_sequences must not be None')
    if v < 0:
        raise ValueError('num_return_sequences must be greater than 0')
    return v