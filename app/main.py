import os
import sys
import logging
from typing import Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from .schemas import BloomRequest, BloomResponse, BloomEmbeddingRequest, BloomEmbeddingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.on_event("startup")
async def startup() -> None:
    global tokenizer, model
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    if not torch.cuda.is_available():
        logger.error("CUDA is not available on this device, please enable it to use this service")
        sys.exit(1)
    logger.info("Starting to load the tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_CARD"))
    logger.info("Tokenizer loaded")
    logger.info("Starting to load the model")
    model = AutoModelForCausalLM.from_pretrained(os.getenv("MODEL_CARD"), load_in_4bit=True, device_map="auto")
    logger.info("Model loaded")

@app.post("/get/completion", response_model=BloomResponse)
async def get_completion(request: BloomRequest) -> Union[BloomResponse, str]:
    try:
        logger.info("Starting to tokenize the input")
        inputs = tokenizer(request.text, return_tensors="pt")
        logger.info("Tokenization done")
        total_tokens = inputs["input_ids"].shape[1] + request.max_length
        if total_tokens > model.config.hidden_size:
            return f"Total tokens {total_tokens} exceeds the maximum length of the model {model.config.max_length}"
        logger.info("Starting to generate the text")
        outputs = model.generate(**inputs, max_length=request.max_length, temperature=request.temperature, top_p=request.top_p, top_k=request.top_k, num_return_sequences=request.num_return_sequences, diversity_penalty=request.diversity_penalty, repetition_penalty=request.repetition_penalty)
        logger.info("Text generated")
        logger.info("Starting to decode the output")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Decoding done")
        logger.info("Returning the response")
        return BloomResponse(generated_text=generated_text, prompt_tokens=inputs["input_ids"].shape[1], completion_tokens=outputs.shape[1])
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        return f"An error occurred while processing the request: {e}"