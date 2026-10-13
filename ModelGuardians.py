"""Compatibility entrypoint for the model monitoring application."""

import os

from app import app, create_app

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
    )
