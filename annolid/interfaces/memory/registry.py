import os
import logging
from typing import Optional
from annolid.domain.memory.protocols import MemoryBackend
from annolid.services.memory.memory_service import MemoryService
from annolid.services.memory.retrieval_service import RetrievalService
from annolid.services.memory.context_service import ContextService
from annolid.services.memory.persistence_service import PersistenceService

logger = logging.getLogger(__name__)

_memory_backend: Optional[MemoryBackend] = None
_memory_service: Optional[MemoryService] = None
_retrieval_service: Optional[RetrievalService] = None
_context_service: Optional[ContextService] = None
_persistence_service: Optional[PersistenceService] = None


def _initialize_subsystem():
    global \
        _memory_backend, \
        _memory_service, \
        _retrieval_service, \
        _context_service, \
        _persistence_service
    _memory_backend = None
    _memory_service = None
    _retrieval_service = None
    _context_service = None
    _persistence_service = None

    backend_type = os.getenv("ANNOLID_MEMORY_BACKEND", "lancedb").lower()

    if backend_type == "none":
        return

    if backend_type == "lancedb":
        try:
            from annolid.infrastructure.memory.lancedb.config import LanceDBConfig
            from annolid.infrastructure.memory.lancedb.backend import (
                LanceDBMemoryBackend,
            )

            config = LanceDBConfig.from_env()
            _memory_backend = LanceDBMemoryBackend(config)
        except ImportError as e:
            logger.warning(f"Could not load LanceDB backend: {e}")
            _memory_backend = None
    else:
        logger.warning("Unsupported memory backend: %s", backend_type)

    if _memory_backend:
        _memory_service = MemoryService(_memory_backend)
        _retrieval_service = RetrievalService(_memory_backend)
        _context_service = ContextService(_retrieval_service)
        _persistence_service = PersistenceService(_memory_service)


def get_memory_backend() -> Optional[MemoryBackend]:
    if _memory_backend is None:
        _initialize_subsystem()
    return _memory_backend


def get_memory_service() -> Optional[MemoryService]:
    if _memory_service is None:
        _initialize_subsystem()
    return _memory_service


def get_retrieval_service() -> Optional[RetrievalService]:
    if _retrieval_service is None:
        _initialize_subsystem()
    return _retrieval_service


def get_context_service() -> Optional[ContextService]:
    if _context_service is None:
        _initialize_subsystem()
    return _context_service


def get_persistence_service() -> Optional[PersistenceService]:
    if _persistence_service is None:
        _initialize_subsystem()
    return _persistence_service
