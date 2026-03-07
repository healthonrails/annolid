import logging
import functools
from typing import Any, Callable, Dict, List, Optional
from annolid.interfaces.memory.registry import get_memory_service
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory, MemorySource

logger = logging.getLogger(__name__)


class AutoCaptureService:
    """Provides utilities and decorators to selectively capture insights from operations."""

    @staticmethod
    def capture_insight(
        text: str,
        scope: str = MemoryScope.GLOBAL,
        category: str = MemoryCategory.FACT,
        source: str = MemorySource.SYSTEM,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Manually trigger an asynchronous or background capture of a useful insight."""
        service = get_memory_service()
        if not service:
            return None

        try:
            return service.store_memory(
                text=text,
                scope=scope,
                category=category,
                source=source,
                importance=importance,
                tags=tags,
                dedupe=True,
            )
        except Exception as e:
            logger.warning(f"Auto-capture failed silently: {e}")
            return None

    @staticmethod
    def capture_on_success(
        success_message_extractor: Callable[
            [Any, Any], str
        ],  # (result, kwargs) -> text
        scope_generator: Callable[[Dict[str, Any]], str] = lambda _: MemoryScope.GLOBAL,
        category: str = MemoryCategory.WORKFLOW_RECIPE,
        importance: float = 0.7,
    ):
        """
        Decorator that auto-captures a memory if the wrapped function executes successfully.
        """

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                try:
                    # Only capture if it didn't raise
                    text = success_message_extractor(result, kwargs)
                    if text:
                        scope = scope_generator(kwargs)
                        AutoCaptureService.capture_insight(
                            text=text,
                            scope=scope,
                            category=category,
                            importance=importance,
                            source=MemorySource.WORKFLOW,
                        )
                except Exception as e:
                    logger.debug(f"Auto-capture on success failed: {e}")
                return result

            return wrapper

        return decorator

    @staticmethod
    def capture_on_error(
        scope_generator: Callable[[Dict[str, Any]], str] = lambda _: MemoryScope.GLOBAL,
        tags_generator: Callable[[Exception], List[str]] = lambda _: ["auto_error"],
    ):
        """
        Decorator that auto-captures troubleshooting context if the wrapped function fails.
        """

        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    try:
                        scope = scope_generator(kwargs)
                        tags = tags_generator(e)
                        func_name = getattr(func, "__name__", "unknown_function")

                        error_msg = f"Task '{func_name}' failed with {type(e).__name__}: {str(e)}"

                        AutoCaptureService.capture_insight(
                            text=error_msg,
                            scope=scope,
                            category=MemoryCategory.TROUBLESHOOTING,
                            source=MemorySource.SYSTEM,
                            importance=0.8,
                            tags=tags,
                        )
                    except Exception as logging_err:
                        logger.debug(
                            f"Auto-capture on error failed to log: {logging_err}"
                        )

                    # Re-raise the original exception
                    raise e

            return wrapper

        return decorator
