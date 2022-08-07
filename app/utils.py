import logging

from rich.logging import RichHandler


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    for name in {"uvicorn", "uvicorn.access", "uvicorn.error"}:
        logger = logging.getLogger(name)
        logger.handlers = [RichHandler(rich_tracebacks=True)]
        logger.propagate = False
