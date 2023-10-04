# 4 Bit Bloom 7B API Service

## Overview
This repository contain the source code for a FastAPI service to provide a REST API for hosting a Bloom language model (7B parameters) with 4 bit quantization for inference.

## Requirements
- Python >= 3.10
- Minimum 16GB RAM
- GPU (VRAM >= 8GB) with CUDA 11.1
- [Poetry](https://python-poetry.org/)

## Installation
### Install using Poetry
Install the poetry package manager:
```bash
pip install poetry
```

Install the dependencies using poetry:
```bash
poetry install
```

## Usage
Run the service using poetry:
```bash
poetry run ./run.sh
```

## TODO List

- [x] Add FastAPI service
- [x] Add Dockerfile
- [x] Add multiple conccurent requests for batch inference
- [] Add ONNX runtime for inference
