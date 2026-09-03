FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# run as an unprivileged user rather than root
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

# Production ASGI server; module-level app so no factory args needed
CMD ["uvicorn", "doc_intelligence.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
