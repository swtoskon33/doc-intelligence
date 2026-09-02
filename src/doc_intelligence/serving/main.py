"""Production ASGI entrypoint: a module-level app for uvicorn / the Docker image.

Run: uvicorn doc_intelligence.serving.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from doc_intelligence.serving.app import create_app

app = create_app()
