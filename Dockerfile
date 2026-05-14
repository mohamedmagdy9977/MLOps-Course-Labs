FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app/
COPY main.py ./
COPY data ./data/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Bind on 0.0.0.0 so the port is reachable from outside the container
CMD ["litestar", "--app", "main:app", "run", "--host", "0.0.0.0", "--port", "8000"]
