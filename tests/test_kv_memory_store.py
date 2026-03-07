from __future__ import annotations

import torch

from annolid.segmentation.cutie_vos.inference.kv_memory_store import KeyValueMemoryStore


def _seed_store_with_tokens(token_count: int = 6) -> tuple[KeyValueMemoryStore, int]:
    store = KeyValueMemoryStore(save_selection=False, save_usage=False)
    key = torch.randn(1, 2, token_count)
    shrinkage = torch.randn(1, 1, token_count)
    values = {1: torch.randn(1, 3, token_count)}
    store.add(key, values, shrinkage, selection=None)
    bucket_id = next(iter(store.buckets.keys()))
    return store, bucket_id


def test_remove_old_memory_accepts_float_max_len() -> None:
    store, bucket_id = _seed_store_with_tokens(6)
    store.remove_old_memory(bucket_id, 4.0)
    assert store.k[bucket_id].shape[-1] == 4
    assert store.v[1].shape[-1] == 4


def test_sieve_by_range_accepts_float_like_bounds() -> None:
    store, bucket_id = _seed_store_with_tokens(8)
    store.sieve_by_range(bucket_id, 0.0, -4.0, 0.0)
    assert store.k[bucket_id].shape[-1] == 4
    assert store.s[bucket_id].shape[-1] == 4
    assert store.v[1].shape[-1] == 4
