"""Local dev server: run the extraction API on port 8000."""
from __future__ import annotations

import uvicorn

from doc_intelligence.serving.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
