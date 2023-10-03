#!/bin/bash
gunicorn -w ${WORKERS:=4} \
  -b :8000 \
  -t ${TIMEOUT:=180} \
  -k uvicorn.workers.UvicornWorker \
  app.main:app \