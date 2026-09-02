FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Production ASGI server; module-level app so no factory args needed
CMD ["uvicorn", "doc_intelligence.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
