import logging
from functools import wraps

logger = logging.getLogger("nr_openai_observability")


def handle_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.error(f"An error occurred in {func.__name__}: {err}")

    return wrapper
