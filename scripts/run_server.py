"""Launch the tiered embedding cache server with uvicorn.

Usage:
    python scripts/run_server.py
    HOT_CAPACITY=512 python scripts/run_server.py   # override via env

Then hit e.g. http://127.0.0.1:8000/embedding/hello and /stats.
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    # Import string form so uvicorn can manage the app lifecycle.
    uvicorn.run("src.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
