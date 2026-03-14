from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import threading
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")
K = TypeVar("K")


@dataclass(frozen=True)
class TileRenderPlan(Generic[K]):
    visible_keys: tuple[K, ...] = ()
    prefetch_keys: tuple[K, ...] = ()
    stale_keys: tuple[K, ...] = ()


@dataclass
class TileSchedulerStats:
    visible_requests: int = 0
    prefetch_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    loads_started: int = 0
    loads_completed: int = 0
    stale_skips: int = 0
    outstanding_requests: int = 0


class TileRequestScheduler:
    def __init__(
        self,
        *,
        cache_get: Callable[[K], T | None],
        cache_put: Callable[[K, T], None],
        load_tile: Callable[[K], T],
        async_load: bool = False,
        max_workers: int = 1,
    ) -> None:
        self._cache_get = cache_get
        self._cache_put = cache_put
        self._load_tile = load_tile
        self._async_load = bool(async_load)
        self._generation = 0
        self._pending: dict[K, tuple[int, Future[T]]] = {}
        self._completed: dict[K, tuple[int, T]] = {}
        self._lock = threading.Lock()
        self._stats = TileSchedulerStats()
        self._executor = (
            ThreadPoolExecutor(
                max_workers=max(1, int(max_workers)),
                thread_name_prefix="annolid_tile",
            )
            if self._async_load
            else None
        )

    def reset(self) -> None:
        self._generation += 1
        with self._lock:
            for _key, (_generation, future) in list(self._pending.items()):
                future.cancel()
            self._pending.clear()
            self._completed.clear()
        self._stats = TileSchedulerStats()

    def stats(self) -> TileSchedulerStats:
        stats = TileSchedulerStats(**self._stats.__dict__)
        stats.outstanding_requests = len(self._pending)
        stats.cache_misses = max(
            0,
            int(stats.visible_requests)
            + int(stats.prefetch_requests)
            - int(stats.cache_hits),
        )
        return stats

    def begin_generation(self) -> int:
        self._generation += 1
        return self._generation

    def schedule(
        self,
        visible_keys: Iterable[K],
        *,
        prefetch_keys: Iterable[K] = (),
        prime_keys: Iterable[K] = (),
    ) -> dict[K, T]:
        generation = self.begin_generation()
        visible_keys = tuple(visible_keys)
        prefetch_keys = tuple(prefetch_keys)
        prime_set = set(prime_keys)
        self._cancel_stale_pending(set(visible_keys) | set(prefetch_keys))
        ready: dict[K, T] = {}
        for key in visible_keys:
            result = self._resolve_key(
                key,
                generation=generation,
                prefetch=False,
                force_sync=key in prime_set,
            )
            if result is not None:
                ready[key] = result
        for key in prefetch_keys:
            self._resolve_key(
                key,
                generation=generation,
                prefetch=True,
                force_sync=False,
            )
        return ready

    def take_completed(self) -> dict[K, T]:
        ready: dict[K, T] = {}
        current_generation = self._generation
        with self._lock:
            items = list(self._completed.items())
            self._completed.clear()
        for key, (generation, tile) in items:
            self._cache_put(key, tile)
            if generation == current_generation:
                ready[key] = tile
            else:
                self._stats.stale_skips += 1
        return ready

    def shutdown(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _cancel_stale_pending(self, active_keys: set[K]) -> None:
        with self._lock:
            for key, (_generation, future) in list(self._pending.items()):
                if key in active_keys:
                    continue
                if future.cancel():
                    self._pending.pop(key, None)
                    self._stats.stale_skips += 1

    def _resolve_key(
        self, key: K, *, generation: int, prefetch: bool, force_sync: bool
    ) -> T | None:
        cached = self._cache_get(key)
        if prefetch:
            self._stats.prefetch_requests += 1
        else:
            self._stats.visible_requests += 1
        if cached is not None:
            self._stats.cache_hits += 1
            return cached
        with self._lock:
            if key in self._pending:
                return None
        if not self._async_load or force_sync or self._executor is None:
            self._stats.loads_started += 1
            tile = self._load_tile(key)
            self._cache_put(key, tile)
            self._stats.loads_completed += 1
            if generation != self._generation:
                self._stats.stale_skips += 1
                return None
            return tile
        self._stats.loads_started += 1
        future = self._executor.submit(self._load_tile, key)
        with self._lock:
            self._pending[key] = (generation, future)

        def _on_done(
            done: Future[T], *, tile_key: K = key, tile_generation: int = generation
        ) -> None:
            with self._lock:
                self._pending.pop(tile_key, None)
            if done.cancelled():
                return
            try:
                tile = done.result()
            except Exception:
                return
            self._stats.loads_completed += 1
            with self._lock:
                self._completed[tile_key] = (tile_generation, tile)

        future.add_done_callback(_on_done)
        return None
