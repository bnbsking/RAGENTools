# /bin/bash

projectPath="/home/james/Desktop/code/RAGENTools"

docker run -it --rm \
  -v "${projectPath}:/app" \
  -w /app \
  --name lapp \
  python:3.12-slim \
  /bin/bash
