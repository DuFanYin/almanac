"""python -m src.web: the page on 127.0.0.1:WEB_PORT (8083 by default): open http://127.0.0.1:8083."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.web.app:app", host="127.0.0.1", port=int(os.environ.get("WEB_PORT", 8083)), proxy_headers=False)
