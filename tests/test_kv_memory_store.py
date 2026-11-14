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


def test_permanent_insertion_preserves_perm_temp_boundary_and_trimming() -> None:
    # 1. Purging a bucket that only ever received permanent tokens must not KeyError
    # on uninitialized temporary selection/usage dictionaries.
    perm_only = KeyValueMemoryStore(save_selection=True, save_usage=True)
    perm_only.add(
        torch.ones(1, 2, 4),
        {1: torch.ones(1, 3, 4)},
        torch.ones(1, 1, 4),
        selection=torch.ones(1, 2, 4),
        as_permanent="first",
    )
    perm_only.purge_except([])
    assert not perm_only.engaged()

    # 2. Adding `as_permanent='all'` after temporary memory exists must insert at
    # `perm_end_pt` rather than after temporary tokens, keeping `e` and `use_cnt`
    # aligned with temporary tokens and protecting the new permanent correction.
    store = KeyValueMemoryStore(save_selection=True, save_usage=True)
    k_p0 = torch.full((1, 2, 3), 1.0)
    k_t1 = torch.full((1, 2, 4), 2.0)
    k_p2 = torch.full((1, 2, 3), 3.0)
    s_3 = torch.ones(1, 1, 3)
    s_4 = torch.ones(1, 1, 4)
    e_t1 = torch.full((1, 2, 4), 0.5)

    store.add(
        k_p0,
        {10: torch.full((1, 3, 3), 10.0), 20: torch.full((1, 3, 3), 20.0)},
        s_3,
        selection=torch.zeros(1, 2, 3),
        as_permanent="first",
    )
    store.add(
        k_t1,
        {10: torch.full((1, 3, 4), 11.0), 20: torch.full((1, 3, 4), 21.0)},
        s_4,
        selection=e_t1,
        as_permanent="first",
    )
    # Stream the permanent mid-run correction across 2 object chunks
    store.add_streaming(
        k_p2,
        [
            {10: torch.full((1, 3, 3), 12.0)},
            {20: torch.full((1, 3, 3), 22.0)},
        ],
        [10, 20],
        s_3,
        selection=torch.zeros(1, 2, 3),
        as_permanent="all",
    )

    assert store.perm_size(0) == 6
    assert store.non_perm_size(0) == 4
    assert torch.equal(store.k[0][:, :, :3], k_p0)
    assert torch.equal(store.k[0][:, :, 3:6], k_p2)
    assert torch.equal(store.k[0][:, :, 6:], k_t1)
    assert torch.all(store.v[10][:, :, :3] == 10.0)
    assert torch.all(store.v[10][:, :, 3:6] == 12.0)
    assert torch.all(store.v[10][:, :, 6:] == 11.0)
    assert torch.equal(store.e[0], e_t1)

    # Trimming permanent tokens retains initial anchor + latest permanent tokens
    # while leaving temporary tokens and usage/selection metadata untouched.
    store.trim_permanent_memory(0, max_perm_tokens=4)
    assert store.perm_size(0) == 4
    assert store.non_perm_size(0) == 4
    assert torch.all(store.k[0][:, :, :2] == 1.0)
    assert torch.all(store.k[0][:, :, 2:4] == 3.0)
    assert torch.equal(store.k[0][:, :, 4:], k_t1)
    assert torch.equal(store.e[0], e_t1)
    assert store.use_cnt[0].shape == (1, 4)

    store.clear_non_permanent_memory()
    assert store.size(0) == 4
    assert store.non_perm_size(0) == 0
    assert torch.all(store.v[20][:, :, :2] == 20.0)
    assert torch.all(store.v[20][:, :, 2:4] == 22.0)
