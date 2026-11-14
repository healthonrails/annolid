from typing import Dict, Iterable, List, Optional, Literal
from collections import defaultdict
import torch


def _coerce_index(value, *, name: str) -> int:
    """Convert slice-related values to a safe integer index."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except Exception as exc:
        raise TypeError(f"{name} must be an integer-like value, got {type(value)!r}") from exc


def _add_last_dim(
    dictionary,
    key,
    new_value,
    prepend: bool = False,
    *,
    insert_at: Optional[int] = None,
):
    # append/insert a new value along the last dimension of a tensor in a dictionary
    # if the key does not exist, put the new value in
    if key not in dictionary:
        dictionary[key] = new_value
        return
    existing = dictionary[key]
    if insert_at is not None:
        if insert_at <= 0:
            dictionary[key] = torch.cat([new_value, existing], -1)
        elif insert_at >= existing.shape[-1]:
            dictionary[key] = torch.cat([existing, new_value], -1)
        else:
            dictionary[key] = torch.cat(
                [existing[..., :insert_at], new_value, existing[..., insert_at:]], -1
            )
    elif prepend:
        dictionary[key] = torch.cat([new_value, existing], -1)
    else:
        dictionary[key] = torch.cat([existing, new_value], -1)


class KeyValueMemoryStore:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """
    def __init__(self, save_selection: bool = False, save_usage: bool = False):
        """
        We store keys and values of objects that first appear in the same frame in a bucket.
        Each bucket contains a set of object ids.
        Each bucket is associated with a single key tensor
            and a dictionary of value tensors indexed by object id.

        The keys and values are stored as the concatenation of a permanent part and a temporary part.
        """
        self.save_selection = save_selection
        self.save_usage = save_usage

        self.global_bucket_id = 0  # does not reduce even if buckets are removed
        self.buckets: Dict[int, List[int]] = {}  # indexed by bucket id
        self.k: Dict[int, torch.Tensor] = {}  # indexed by bucket id
        self.v: Dict[int, torch.Tensor] = {}  # indexed by object id

        # indexed by bucket id; the end point of permanent memory
        self.perm_end_pt: Dict[int, int] = defaultdict(int)

        # shrinkage and selection are just like the keys
        self.s = {}
        if self.save_selection:
            self.e = {}  # does not contain the permanent memory part

        # usage
        if self.save_usage:
            self.use_cnt = {}  # indexed by bucket id, does not contain the permanent memory part
            self.life_cnt = {}  # indexed by bucket id, does not contain the permanent memory part

    def add(self,
            key: torch.Tensor,
            values: Dict[int, torch.Tensor],
            shrinkage: torch.Tensor,
            selection: torch.Tensor,
            supposed_bucket_id: int = -1,
            as_permanent: Literal['no', 'first', 'all'] = 'no') -> None:
        """
        key: (1/2)*C*N
        values: dict of values ((1/2)*C*N), object ids are used as keys
        shrinkage: (1/2)*1*N
        selection: (1/2)*C*N

        supposed_bucket_id: used to sync the bucket id between working and long-term memory
        if provided, the input should all be in a single bucket indexed by this id
        as_permanent: whether to store the input as permanent memory
            'no': don't
            'first': only store it as permanent memory if the bucket is empty
            'all': always store it as permanent memory
        """
        self.add_streaming(
            key,
            [values],
            list(values.keys()),
            shrinkage,
            selection,
            supposed_bucket_id=supposed_bucket_id,
            as_permanent=as_permanent,
        )

    def add_streaming(
        self,
        key: torch.Tensor,
        values_chunks: Iterable[Dict[int, torch.Tensor]],
        all_objects: List[int],
        shrinkage: torch.Tensor,
        selection: Optional[torch.Tensor],
        supposed_bucket_id: int = -1,
        as_permanent: Literal['no', 'first', 'all'] = 'no',
    ) -> None:
        """Insert a single memory frame across one or more object-value chunks."""
        bs = key.shape[0]
        ne = key.shape[-1]
        assert len(key.shape) == 3
        assert len(shrinkage.shape) == 3
        assert not self.save_selection or selection is None or len(selection.shape) == 3
        assert as_permanent in ['no', 'first', 'all']

        old_obj_to_bucket = {
            obj_id: b_id
            for b_id, object_ids in self.buckets.items()
            for obj_id in object_ids
        }
        old_perm_end = {b_id: int(self.perm_end_pt[b_id]) for b_id in self.buckets}

        new_bucket_id: Optional[int] = None
        new_bucket_objs: List[int] = []
        planned_obj_bucket: Dict[int, int] = {}
        if supposed_bucket_id >= 0:
            enabled_buckets = [supposed_bucket_id]
            bucket_exist = supposed_bucket_id in self.buckets
            if bucket_exist:
                for obj in all_objects:
                    assert obj in self.v
                    assert old_obj_to_bucket.get(obj) == supposed_bucket_id
            else:
                for obj in all_objects:
                    assert obj not in self.v
                old_perm_end[supposed_bucket_id] = int(
                    self.perm_end_pt[supposed_bucket_id]
                )
            for obj in all_objects:
                planned_obj_bucket[obj] = supposed_bucket_id
        else:
            enabled_buckets = []
            enabled_set = set()
            for obj in all_objects:
                if obj in self.v:
                    assert obj in old_obj_to_bucket
                    b_id = old_obj_to_bucket[obj]
                    planned_obj_bucket[obj] = b_id
                    if b_id not in enabled_set:
                        enabled_set.add(b_id)
                        enabled_buckets.append(b_id)
                else:
                    if new_bucket_id is None:
                        new_bucket_id = self.global_bucket_id
                        old_perm_end[new_bucket_id] = 0
                        enabled_set.add(new_bucket_id)
                        enabled_buckets.append(new_bucket_id)
                    planned_obj_bucket[obj] = new_bucket_id
                    new_bucket_objs.append(obj)

        add_as_permanent: Dict[int, bool] = {}
        for bucket_id in enabled_buckets:
            add_as_permanent[bucket_id] = (as_permanent == 'all') or (
                as_permanent == 'first' and old_perm_end[bucket_id] == 0
            )

        seen_objs = set()
        for chunk_values in values_chunks:
            for obj, value in chunk_values.items():
                assert len(value.shape) == 3
                assert value.shape[-1] == ne
                b_id = planned_obj_bucket[obj]
                insert_at = old_perm_end[b_id] if add_as_permanent[b_id] else None
                _add_last_dim(self.v, obj, value, insert_at=insert_at)
                seen_objs.add(obj)
        assert len(seen_objs) == len(all_objects)

        if supposed_bucket_id >= 0:
            self.buckets[supposed_bucket_id] = list(all_objects)
        elif new_bucket_id is not None:
            self.global_bucket_id = new_bucket_id + 1
            self.buckets[new_bucket_id] = new_bucket_objs

        for bucket_id in enabled_buckets:
            if add_as_permanent[bucket_id]:
                self.perm_end_pt[bucket_id] = old_perm_end[bucket_id] + ne

        if self.save_usage and any(not flag for flag in add_as_permanent.values()):
            new_count = torch.zeros((bs, ne), device=key.device, dtype=torch.float32)
            new_life = torch.zeros((bs, ne), device=key.device, dtype=torch.float32) + 1e-7

        for bucket_id in self.buckets:
            if bucket_id not in add_as_permanent:
                continue

            insert_at = (
                old_perm_end[bucket_id] if add_as_permanent[bucket_id] else None
            )
            _add_last_dim(self.k, bucket_id, key, insert_at=insert_at)
            _add_last_dim(self.s, bucket_id, shrinkage, insert_at=insert_at)
            if not add_as_permanent[bucket_id]:
                if self.save_selection:
                    assert selection is not None and len(selection.shape) == 3
                    _add_last_dim(self.e, bucket_id, selection)
                if self.save_usage:
                    _add_last_dim(self.use_cnt, bucket_id, new_count)
                    _add_last_dim(self.life_cnt, bucket_id, new_life)

    def update_bucket_usage(self, bucket_id: int, usage: torch.Tensor) -> None:
        # increase all life count by 1
        # increase use of indexed elements
        if not self.save_usage:
            return

        usage = usage[:, self.perm_end_pt[bucket_id]:]
        if usage.shape[-1] == 0:
            # if there is no temporary memory, we don't need to update
            return
        self.use_cnt[bucket_id] += usage.view_as(self.use_cnt[bucket_id])
        self.life_cnt[bucket_id] += 1

    def sieve_by_range(self, bucket_id: int, start: int, end: int, min_size: int) -> None:
        # keep only the temporary elements *outside* of this range (with some boundary conditions)
        # the permanent elements are ignored in this computation
        # i.e., concat (a[:start], a[end:])
        # bucket with size <= min_size are not modified

        start = _coerce_index(start, name="start")
        end = _coerce_index(end, name="end")
        min_size = _coerce_index(min_size, name="min_size")
        assert start >= 0
        assert end <= 0

        object_ids = self.buckets[bucket_id]
        bucket_num_elements = self.k[bucket_id].shape[-1] - self.perm_end_pt[bucket_id]
        if bucket_num_elements <= min_size:
            return

        if end == 0:
            # negative 0 would not work as the end index!
            # effectively make the second part an empty slice
            end = self.k[bucket_id].shape[-1] + 1

        p_size = self.perm_end_pt[bucket_id]
        start = start + p_size

        k = self.k[bucket_id]
        s = self.s[bucket_id]
        if self.save_selection:
            e = self.e[bucket_id]
        if self.save_usage:
            use_cnt = self.use_cnt[bucket_id]
            life_cnt = self.life_cnt[bucket_id]

        self.k[bucket_id] = torch.cat([k[:, :, :start], k[:, :, end:]], -1)
        self.s[bucket_id] = torch.cat([s[:, :, :start], s[:, :, end:]], -1)
        if self.save_selection:
            self.e[bucket_id] = torch.cat([e[:, :, :start - p_size], e[:, :, end:]], -1)
        if self.save_usage:
            self.use_cnt[bucket_id] = torch.cat([use_cnt[:, :start - p_size], use_cnt[:, end:]], -1)
            self.life_cnt[bucket_id] = torch.cat([life_cnt[:, :start - p_size], life_cnt[:, end:]],
                                                 -1)
        for obj_id in object_ids:
            v = self.v[obj_id]
            self.v[obj_id] = torch.cat([v[:, :, :start], v[:, :, end:]], -1)

    def remove_old_memory(self, bucket_id: int, max_len: int) -> None:
        max_len = max(0, _coerce_index(max_len, name="max_len"))
        self.sieve_by_range(bucket_id, 0, -max_len, max_len)

    def remove_obsolete_features(self, bucket_id: int, max_size: int) -> None:
        # for long-term memory only
        object_ids = self.buckets[bucket_id]

        assert self.perm_end_pt[bucket_id] == 0  # permanent memory should be empty in LT memory

        # normalize with life duration
        usage = self.get_usage(bucket_id)
        bs = usage.shape[0]

        survivals = []

        for bi in range(bs):
            _, survived = torch.topk(usage[bi], k=max_size)
            survivals.append(survived.flatten())
            assert survived.shape[-1] == survivals[0].shape[-1]

        self.k[bucket_id] = torch.stack(
            [self.k[bucket_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)
        self.s[bucket_id] = torch.stack(
            [self.s[bucket_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)

        if self.save_selection:
            # Long-term memory does not store selection so this should not be needed
            self.e[bucket_id] = torch.stack(
                [self.e[bucket_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)
        for obj_id in object_ids:
            self.v[obj_id] = torch.stack(
                [self.v[obj_id][bi, :, survived] for bi, survived in enumerate(survivals)], 0)

        self.use_cnt[bucket_id] = torch.stack(
            [self.use_cnt[bucket_id][bi, survived] for bi, survived in enumerate(survivals)], 0)
        self.life_cnt[bucket_id] = torch.stack(
            [self.life_cnt[bucket_id][bi, survived] for bi, survived in enumerate(survivals)], 0)

    def get_usage(self, bucket_id: int) -> torch.Tensor:
        # return normalized usage
        if not self.save_usage:
            raise RuntimeError('I did not count usage!')
        else:
            usage = self.use_cnt[bucket_id] / self.life_cnt[bucket_id]
            return usage

    def get_all_sliced(
        self, bucket_id: int, start: int, end: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], torch.Tensor):
        # return k, sk, ek, value, normalized usage in order, sliced by start and end
        # this only queries the temporary memory

        assert start >= 0
        assert end <= 0

        p_size = self.perm_end_pt[bucket_id]
        start = start + p_size

        if end == 0:
            # negative 0 would not work as the end index!
            k = self.k[bucket_id][:, :, start:]
            sk = self.s[bucket_id][:, :, start:]
            ek = self.e[bucket_id][:, :, start - p_size:] if self.save_selection else None
            value = {obj_id: self.v[obj_id][:, :, start:] for obj_id in self.buckets[bucket_id]}
            usage = self.get_usage(bucket_id)[:, start - p_size:] if self.save_usage else None
        else:
            k = self.k[bucket_id][:, :, start:end]
            sk = self.s[bucket_id][:, :, start:end]
            ek = self.e[bucket_id][:, :, start - p_size:end] if self.save_selection else None
            value = {obj_id: self.v[obj_id][:, :, start:end] for obj_id in self.buckets[bucket_id]}
            usage = self.get_usage(bucket_id)[:, start - p_size:end] if self.save_usage else None

        return k, sk, ek, value, usage

    def purge_except(self, obj_keep_idx: List[int]):
        # purge certain objects from the memory except the one listed
        obj_keep_idx = set(obj_keep_idx)

        # remove objects that are not in the keep list from the buckets
        buckets_to_remove = []
        for bucket_id, object_ids in self.buckets.items():
            self.buckets[bucket_id] = [obj_id for obj_id in object_ids if obj_id in obj_keep_idx]
            if len(self.buckets[bucket_id]) == 0:
                buckets_to_remove.append(bucket_id)

        # remove object values that are not in the keep list
        self.v = {k: v for k, v in self.v.items() if k in obj_keep_idx}

        # remove buckets that are empty
        for bucket_id in buckets_to_remove:
            del self.buckets[bucket_id]
            self.k.pop(bucket_id, None)
            self.s.pop(bucket_id, None)
            self.perm_end_pt.pop(bucket_id, None)
            if self.save_selection:
                self.e.pop(bucket_id, None)
            if self.save_usage:
                self.use_cnt.pop(bucket_id, None)
                self.life_cnt.pop(bucket_id, None)

    def trim_permanent_memory(self, bucket_id: int, max_perm_tokens: int) -> None:
        """Bound permanent tokens in `bucket_id`, retaining initial anchor + newest permanent tokens."""
        max_perm_tokens = _coerce_index(max_perm_tokens, name="max_perm_tokens")
        if bucket_id not in self.buckets or max_perm_tokens <= 0:
            return
        p_size = int(self.perm_end_pt.get(bucket_id, 0))
        if p_size <= max_perm_tokens:
            return

        keep_head = max(1, max_perm_tokens // 2)
        keep_tail = max_perm_tokens - keep_head

        def _slice_perm(tensor: torch.Tensor) -> torch.Tensor:
            if keep_tail > 0:
                return torch.cat(
                    [tensor[..., :keep_head], tensor[..., p_size - keep_tail:]],
                    dim=-1,
                )
            return torch.cat(
                [tensor[..., :keep_head], tensor[..., p_size:]],
                dim=-1,
            )

        self.k[bucket_id] = _slice_perm(self.k[bucket_id])
        self.s[bucket_id] = _slice_perm(self.s[bucket_id])
        for obj_id in self.buckets[bucket_id]:
            self.v[obj_id] = _slice_perm(self.v[obj_id])
        self.perm_end_pt[bucket_id] = max_perm_tokens

    def clear_non_permanent_memory(self):
        # clear all non-permanent memory
        for bucket_id in self.buckets:
            self.sieve_by_range(bucket_id, 0, 0, 0)

    def cast_values_dtype(self, dtype: torch.dtype) -> None:
        """Convert stored value tensors in-place to `dtype` with float16 range safety."""
        for obj_id, val in list(self.v.items()):
            if val.dtype == dtype:
                continue
            if dtype == torch.float16 and val.dtype == torch.float32:
                self.v[obj_id] = val.clamp(-65504.0, 65504.0).to(dtype=dtype)
            else:
                self.v[obj_id] = val.to(dtype=dtype)

    def value_bytes(self, *, non_permanent_only: bool = False) -> int:
        """Return total byte footprint of stored `v` tensors across all objects."""
        if not self.v:
            return 0
        if not non_permanent_only:
            return sum(int(v.numel() * v.element_size()) for v in self.v.values())
        total = 0
        for bucket_id, object_ids in self.buckets.items():
            p_size = int(self.perm_end_pt.get(bucket_id, 0))
            for obj_id in object_ids:
                val = self.v.get(obj_id)
                if val is None:
                    continue
                non_perm_tokens = max(0, int(val.shape[-1]) - p_size)
                elems = int(val[..., :1].numel()) * non_perm_tokens
                total += elems * int(val.element_size())
        return total

    def get_v_size(self, obj_id: int) -> int:
        return self.v[obj_id].shape[-1]

    def size(self, bucket_id: int) -> int:
        if bucket_id not in self.k:
            return 0
        else:
            return self.k[bucket_id].shape[-1]

    def perm_size(self, bucket_id: int) -> int:
        return self.perm_end_pt[bucket_id]

    def non_perm_size(self, bucket_id: int) -> int:
        return self.size(bucket_id) - self.perm_size(bucket_id)

    def engaged(self, bucket_id: Optional[int] = None) -> bool:
        if bucket_id is None:
            return len(self.buckets) > 0
        else:
            return bucket_id in self.buckets

    @property
    def num_objects(self) -> int:
        return len(self.v)

    @property
    def key(self) -> Dict[int, torch.Tensor]:
        return self.k

    @property
    def value(self) -> Dict[int, torch.Tensor]:
        return self.v

    @property
    def shrinkage(self) -> Dict[int, torch.Tensor]:
        return self.s

    @property
    def selection(self) -> Dict[int, torch.Tensor]:
        return self.e

    def __contains__(self, key):
        return key in self.v
