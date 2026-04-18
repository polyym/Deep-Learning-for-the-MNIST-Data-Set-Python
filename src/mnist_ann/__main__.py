"""Run the Flask development server: ``python -m mnist_ann``.

For production, use Gunicorn directly:
    gunicorn "mnist_ann.app:create_app()" --bind 0.0.0.0:5000 --workers 2 --threads 4
"""

from __future__ import annotations

import logging

from .app import create_app
from .config import DATA_DIR, DEBUG, HOST, PORT


def main() -> None:
    app = create_app()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MNIST Neural Network server on {HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Data directory: {DATA_DIR}")
    if DEBUG:
        logger.warning("Running in DEBUG mode - do not use in production!")
    app.run(debug=DEBUG, host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    main()
