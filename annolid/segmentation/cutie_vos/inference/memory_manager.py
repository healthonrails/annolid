import itertools
import logging
import math
from omegaconf import DictConfig
from typing import Iterable, List, Dict, Literal, Optional
import torch
import torch.nn.functional as F

from annolid.segmentation.cutie_vos.inference.object_manager import ObjectManager
from annolid.segmentation.cutie_vos.inference.kv_memory_store import KeyValueMemoryStore
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.model.utils.memory_utils import *

log = logging.getLogger()


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    DEFAULT_AUTO_COMPACT_KV_OBJECTS = 16
    DEFAULT_MAX_KV_MEMORY_MB = 2048

    @staticmethod
    def _cfg_get(cfg, key: str, default):
        if isinstance(cfg, dict):
            val = cfg.get(key, default)
        else:
            try:
                val = cfg.get(key, default)
            except AttributeError:
                val = getattr(cfg, key, default)
        return default if val is None else val

    @staticmethod
    def _normalize_kv_storage_dtype(value) -> str:
        text = str(value or "auto").strip().lower()
        if text in {"auto", "float16", "fp16", "half"}:
            return "auto" if text == "auto" else "float16"
        if text in {"bfloat16", "bf16"}:
            return "bfloat16"
        if text in {"float32", "fp32", "float"}:
            return "float32"
        return "auto"

    @classmethod
    def _normalize_max_kv_memory_mb(cls, value) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(cls.DEFAULT_MAX_KV_MEMORY_MB)
        return parsed

    def __init__(self, cfg: DictConfig, object_manager: ObjectManager):
        self.object_manager = object_manager
        self.sensory_dim = cfg.model.sensory_dim
        self.top_k = cfg.top_k
        self.chunk_size = cfg.chunk_size
        self.kv_storage_dtype = self._normalize_kv_storage_dtype(
            self._cfg_get(cfg, "kv_storage_dtype", "auto")
        )
        self.max_kv_memory_mb = self._normalize_max_kv_memory_mb(
            self._cfg_get(cfg, "max_kv_memory_mb", self.DEFAULT_MAX_KV_MEMORY_MB)
        )
        self._logged_kv_budget_keys = set()

        self.save_aux = cfg.save_aux

        self.use_long_term = cfg.use_long_term
        self.count_long_term_usage = cfg.long_term.count_usage
        # subtract 1 because the first-frame is now counted as "permanent memory"
        # and is not counted towards max_mem_frames
        # but we want to keep the hyperparameters consistent as before for the same behavior
        if self.use_long_term:
            self.max_mem_frames = cfg.long_term.max_mem_frames - 1
            self.min_mem_frames = cfg.long_term.min_mem_frames - 1
            self.num_prototypes = cfg.long_term.num_prototypes
            self.max_long_tokens = cfg.long_term.max_num_tokens
            self.buffer_tokens = cfg.long_term.buffer_tokens
        else:
            self.max_mem_frames = cfg.max_mem_frames - 1

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The sensory memory is stored as a dictionary indexed by object ids
        # each of shape bs * C^h * H * W
        self.sensory = {}

        # a dictionary indexed by object ids, each of shape bs * T * Q * C
        self.obj_v = {}

        self.work_mem = KeyValueMemoryStore(save_selection=self.use_long_term,
                                            save_usage=self.use_long_term)
        if self.use_long_term:
            self.long_mem = KeyValueMemoryStore(
                save_usage=self.count_long_term_usage)

        self.config_stale = True
        self.engaged = False

    def update_config(self, cfg: DictConfig) -> None:
        self.config_stale = True
        self.top_k = cfg['top_k']
        self.kv_storage_dtype = self._normalize_kv_storage_dtype(
            self._cfg_get(cfg, "kv_storage_dtype", getattr(self, "kv_storage_dtype", "auto"))
        )
        self.max_kv_memory_mb = self._normalize_max_kv_memory_mb(
            self._cfg_get(
                cfg,
                "max_kv_memory_mb",
                getattr(self, "max_kv_memory_mb", self.DEFAULT_MAX_KV_MEMORY_MB),
            )
        )

        assert self.use_long_term == cfg.use_long_term, 'cannot update this'
        assert self.count_long_term_usage == cfg.long_term.count_usage, 'cannot update this'

        self.use_long_term = cfg.use_long_term
        self.count_long_term_usage = cfg.long_term.count_usage
        if self.use_long_term:
            self.max_mem_frames = cfg.long_term.max_mem_frames - 1
            self.min_mem_frames = cfg.long_term.min_mem_frames - 1
            self.num_prototypes = cfg.long_term.num_prototypes
            self.max_long_tokens = cfg.long_term.max_num_tokens
            self.buffer_tokens = cfg.long_term.buffer_tokens
        else:
            self.max_mem_frames = cfg.max_mem_frames - 1

    def _resolve_target_value_dtype(
        self, total_objects: int, incoming_dtype: torch.dtype
    ) -> torch.dtype:
        if not incoming_dtype.is_floating_point:
            return incoming_dtype
        mode = getattr(self, "kv_storage_dtype", "auto")
        if mode == "float16":
            return torch.float16
        if mode == "bfloat16":
            return torch.bfloat16
        if mode == "float32":
            return torch.float32
        if int(total_objects) > self.DEFAULT_AUTO_COMPACT_KV_OBJECTS:
            return torch.float16
        existing_sample = next(iter(self.work_mem.v.values()), None)
        if existing_sample is not None and existing_sample.dtype == torch.float16:
            return torch.float16
        return incoming_dtype

    @staticmethod
    def _cast_value_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if tensor.dtype == dtype:
            return tensor
        if dtype == torch.float16 and tensor.dtype == torch.float32:
            return tensor.clamp(-65504.0, 65504.0).to(dtype=dtype)
        return tensor.to(dtype=dtype)

    @staticmethod
    def _compute_pooled_hw(h: int, w: int, max_tokens: int) -> tuple[int, int]:
        raw_tokens = int(h) * int(w)
        if max_tokens <= 0 or raw_tokens <= max_tokens:
            return (int(h), int(w))
        if max_tokens == 1:
            return (1, 1)
        scale = math.sqrt(float(max_tokens) / float(raw_tokens))
        ph = max(1, min(int(h), int(math.floor(h * scale))))
        pw = max(1, min(int(w), int(math.floor(w * scale))))
        while ph * pw > max_tokens:
            if pw >= ph and pw > 1:
                pw -= 1
            elif ph > 1:
                ph -= 1
            else:
                break
        return (ph, pw)

    @staticmethod
    def _pool_spatial_2d(
        tensor: Optional[torch.Tensor],
        target_hw: tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tuple(tensor.shape[-2:]) == tuple(target_hw):
            return tensor
        if tensor.ndim == 4:
            return F.adaptive_avg_pool2d(tensor, target_hw)
        if tensor.ndim == 5:
            b, n, c, h, w = tensor.shape
            pooled = F.adaptive_avg_pool2d(
                tensor.reshape(b * n, c, h, w),
                target_hw,
            )
            return pooled.view(b, n, c, target_hw[0], target_hw[1])
        raise ValueError(
            f"Expected 4D or 5D spatial tensor for KV pooling; got shape {tuple(tensor.shape)}."
        )

    def _budget_tokens_per_object(
        self,
        total_objects: int,
        bs: int,
        value_dtype: torch.dtype,
    ) -> Optional[int]:
        max_kv_mb = float(getattr(self, "max_kv_memory_mb", 0.0))
        if max_kv_mb <= 0.0 or total_objects <= 0 or not self.CV:
            return None
        min_frames = 4 if self.use_long_term else 2
        bytes_per_elem = torch.empty((), dtype=value_dtype).element_size()
        bytes_per_token_per_obj = max(1, int(bs) * int(self.CV) * int(bytes_per_elem))
        budget_bytes = int(max_kv_mb * 1024.0 * 1024.0)
        return max(
            min_frames,
            budget_bytes // max(1, int(total_objects) * bytes_per_token_per_obj),
        )

    def _resolve_frame_pool_hw(
        self,
        raw_h: int,
        raw_w: int,
        total_objects: int,
        bs: int,
        value_dtype: torch.dtype,
    ) -> tuple[int, int]:
        budget_tokens_per_obj = self._budget_tokens_per_object(
            total_objects, bs, value_dtype
        )
        if budget_tokens_per_obj is None:
            return (int(raw_h), int(raw_w))
        min_frames = 4 if self.use_long_term else 2
        max_frame_tokens = max(1, budget_tokens_per_obj // min_frames)
        return self._compute_pooled_hw(raw_h, raw_w, max_frame_tokens)

    def _max_perm_tokens_for_budget(
        self,
        total_objects: int,
        bs: int,
        value_dtype: torch.dtype,
        frame_tokens: int,
    ) -> Optional[int]:
        budget_tokens_per_obj = self._budget_tokens_per_object(
            total_objects, bs, value_dtype
        )
        if budget_tokens_per_obj is None:
            return None
        resolved_frame = max(1, int(frame_tokens))
        if not self.use_long_term:
            nominal_work = max(1, self.max_mem_frames * resolved_frame)
            if budget_tokens_per_obj >= nominal_work + resolved_frame:
                return max(resolved_frame, budget_tokens_per_obj - nominal_work)
            return max(resolved_frame, budget_tokens_per_obj // 2)

        nominal_non_perm = (
            max(1, self.max_mem_frames * resolved_frame) + self.max_long_tokens
        )
        if budget_tokens_per_obj >= nominal_non_perm + resolved_frame:
            return max(resolved_frame, budget_tokens_per_obj - nominal_non_perm)
        return max(resolved_frame, budget_tokens_per_obj // 4)

    def _effective_token_limits(
        self,
        total_objects: int,
        bs: int,
        value_dtype: torch.dtype,
        *,
        frame_tokens: Optional[int] = None,
        perm_tokens: int = 0,
    ) -> tuple[int, int, int, int, int, bool]:
        nominal_min_work = self.min_work_tokens if self.use_long_term else 0
        nominal_max_long = self.max_long_tokens if self.use_long_term else 0
        nominal_prototypes = self.num_prototypes if self.use_long_term else 0
        nominal_buffer = self.buffer_tokens if self.use_long_term else 0
        if not self.HW or not self.CV:
            return (
                self.max_work_tokens,
                nominal_min_work,
                nominal_max_long,
                nominal_prototypes,
                nominal_buffer,
                False,
            )

        budget_tokens_per_obj = self._budget_tokens_per_object(
            total_objects, bs, value_dtype
        )
        if budget_tokens_per_obj is None:
            return (
                self.max_work_tokens,
                nominal_min_work,
                nominal_max_long,
                nominal_prototypes,
                nominal_buffer,
                False,
            )

        resolved_frame = max(
            1, int(self.HW if frame_tokens is None else frame_tokens)
        )
        pooled_frame = resolved_frame < self.HW
        scaled_max_work = (
            max(1, self.max_mem_frames * resolved_frame)
            if pooled_frame
            else self.max_work_tokens
        )
        rem_tokens = max(1, budget_tokens_per_obj - max(0, int(perm_tokens)))

        if not self.use_long_term:
            if not pooled_frame and rem_tokens >= self.max_work_tokens:
                return (self.max_work_tokens, 0, 0, 0, 0, False)
            eff_max_work = max(resolved_frame, min(scaled_max_work, rem_tokens))
            throttled = eff_max_work < self.max_work_tokens or pooled_frame
            return (eff_max_work, 0, 0, 0, 0, throttled)

        scaled_min_work = (
            max(1, self.min_mem_frames * resolved_frame)
            if pooled_frame
            else self.min_work_tokens
        )
        scaled_max_work = max(scaled_min_work + resolved_frame, scaled_max_work)
        nominal_total = scaled_max_work + self.max_long_tokens
        if not pooled_frame and rem_tokens >= nominal_total:
            return (
                self.max_work_tokens,
                self.min_work_tokens,
                self.max_long_tokens,
                self.num_prototypes,
                self.buffer_tokens,
                False,
            )

        rem_tokens = max(3, rem_tokens)
        eff_min_work = max(
            resolved_frame,
            min(scaled_min_work, max(resolved_frame, rem_tokens // 3)),
        )
        eff_max_work = max(
            eff_min_work + resolved_frame,
            min(
                scaled_max_work,
                max(eff_min_work + resolved_frame, (rem_tokens * 2) // 3),
            ),
        )
        max_retained_work = max(eff_min_work, eff_max_work - 1)
        rem_long = max(2, rem_tokens - max_retained_work)
        eff_max_long = max(2, min(self.max_long_tokens, rem_long))
        eff_prototypes = max(
            1,
            min(
                self.num_prototypes,
                resolved_frame,
                max(1, eff_max_long // 3),
            ),
        )
        eff_buffer = max(
            1,
            min(
                self.buffer_tokens,
                max(1, (eff_max_long - eff_prototypes) // 2),
            ),
        )
        throttled = (
            pooled_frame
            or eff_max_work < self.max_work_tokens
            or eff_max_long < self.max_long_tokens
        )
        return (
            eff_max_work,
            eff_min_work,
            eff_max_long,
            eff_prototypes,
            eff_buffer,
            throttled,
        )

    def _readout(self, affinity, v) -> torch.Tensor:
        # affinity: bs*N*HW
        # v: bs*C*N or bs*num_objects*C*N
        # returns bs*C*HW or bs*num_objects*C*HW
        if v.dtype != affinity.dtype:
            v = v.to(dtype=affinity.dtype)
        if len(v.shape) == 3:
            # single object
            return v @ affinity
        else:
            bs, num_objects, C, N = v.shape
            v = v.view(bs, num_objects * C, N)
            out = v @ affinity
            return out.view(bs, num_objects, C, -1)

    def _get_mask_by_ids(self, mask: torch.Tensor, obj_ids: List[int]) -> torch.Tensor:
        # -1 because the mask does not contain the background channel
        return mask[:, [self.object_manager.find_tmp_by_id(obj) - 1 for obj in obj_ids]]

    def _get_sensory_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.sensory[obj] for obj in obj_ids], dim=1)

    def _get_object_mem_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        return torch.stack([self.obj_v[obj] for obj in obj_ids], dim=1)

    def _get_visual_values_by_ids(self, obj_ids: List[int]) -> torch.Tensor:
        # All the values that the object ids refer to should have the same shape
        value = torch.stack([self.work_mem.value[obj]
                            for obj in obj_ids], dim=1)
        if self.use_long_term and obj_ids[0] in self.long_mem.value:
            lt_value = torch.stack([self.long_mem.value[obj]
                                   for obj in obj_ids], dim=1)
            value = torch.cat([lt_value, value], dim=-1)

        return value

    def read(self, pix_feat: torch.Tensor, query_key: torch.Tensor, selection: torch.Tensor,
             last_mask: torch.Tensor, network: CUTIE, *,
             chunk_size: Optional[int] = None) -> Dict[int, torch.Tensor]:
        """
        Read from all memory stores and returns a single memory readout tensor for each object

        pix_feat: (1/2) x C x H x W
        query_key: (1/2) x C^k x H x W
        selection:  (1/2) x C^k x H x W
        last_mask: (1/2) x num_objects x H x W (at stride 16)
        return a dict of memory readouts, indexed by object indices. Each readout is C*H*W
        """
        h, w = pix_feat.shape[-2:]
        bs = pix_feat.shape[0]
        assert query_key.shape[0] == bs
        assert selection.shape[0] == bs
        assert last_mask.shape[0] == bs

        effective_chunk_size = self.chunk_size if chunk_size is None else chunk_size

        query_key = query_key.flatten(start_dim=2)  # bs*C^k*HW
        selection = selection.flatten(start_dim=2)  # bs*C^k*HW

        full_last_mask = F.interpolate(last_mask, size=(h, w), mode="area")
        total_last_mask = (
            None
            if getattr(network, "single_object", False)
            else full_last_mask.sum(dim=1, keepdim=True)
        )
        """
        Compute affinity and perform readout
        """
        all_readout_mem = {}
        buckets = self.work_mem.buckets
        for bucket_id, bucket in buckets.items():
            if self.use_long_term and self.long_mem.engaged(bucket_id):
                # Use long-term memory
                long_mem_size = self.long_mem.size(bucket_id)
                memory_key = torch.cat([self.long_mem.key[bucket_id], self.work_mem.key[bucket_id]],
                                       -1)
                shrinkage = torch.cat(
                    [self.long_mem.shrinkage[bucket_id], self.work_mem.shrinkage[bucket_id]], -1)

                similarity = get_similarity(
                    memory_key, shrinkage, query_key, selection)
                affinity, usage = do_softmax(similarity,
                                             top_k=self.top_k,
                                             inplace=True,
                                             return_usage=True)
                """
                Record memory usage for working and long-term memory
                """
                # ignore the index return for long-term memory
                work_usage = usage[:, long_mem_size:]
                self.work_mem.update_bucket_usage(bucket_id, work_usage)

                if self.count_long_term_usage:
                    # ignore the index return for working memory
                    long_usage = usage[:, :long_mem_size]
                    self.long_mem.update_bucket_usage(bucket_id, long_usage)
            else:
                # no long-term memory
                memory_key = self.work_mem.key[bucket_id]
                shrinkage = self.work_mem.shrinkage[bucket_id]
                similarity = get_similarity(
                    memory_key, shrinkage, query_key, selection)

                if self.use_long_term:
                    affinity, usage = do_softmax(similarity,
                                                 top_k=self.top_k,
                                                 inplace=True,
                                                 return_usage=True)
                    self.work_mem.update_bucket_usage(bucket_id, usage)
                else:
                    affinity = do_softmax(
                        similarity, top_k=self.top_k, inplace=True)

            if effective_chunk_size < 1 or effective_chunk_size >= len(bucket):
                object_chunks = [bucket]
            else:
                object_chunks = [
                    bucket[i:i + effective_chunk_size]
                    for i in range(0, len(bucket), effective_chunk_size)
                ]

            bucket_pixel_readout: Optional[torch.Tensor] = None
            write_offset = 0
            for objects in object_chunks:
                this_sensory = self._get_sensory_by_ids(objects)
                this_last_mask = self._get_mask_by_ids(full_last_mask, objects)
                this_last_others = (
                    None
                    if total_last_mask is None
                    else (total_last_mask - this_last_mask).clamp(0, 1)
                )
                this_msk_value = self._get_visual_values_by_ids(
                    objects)  # (1/2)*num_objects*C*N
                visual_readout = self._readout(affinity,
                                               this_msk_value).view(bs, len(objects), self.CV, h, w)
                fused_chunk = network.pixel_fusion(
                    pix_feat,
                    visual_readout,
                    this_sensory,
                    this_last_mask,
                    chunk_size=effective_chunk_size,
                    last_others=this_last_others,
                )
                if len(object_chunks) == 1:
                    bucket_pixel_readout = fused_chunk
                else:
                    if bucket_pixel_readout is None:
                        bucket_pixel_readout = fused_chunk.new_empty(
                            (bs, len(bucket), fused_chunk.shape[2], h, w)
                        )
                    bucket_pixel_readout[
                        :, write_offset : write_offset + len(objects)
                    ] = fused_chunk
                    write_offset += len(objects)
            bucket_obj_mem = self._get_object_mem_by_ids(bucket).unsqueeze(2)
            readout_memory, aux_features = network.readout_query(
                bucket_pixel_readout,
                bucket_obj_mem,
                need_weights=self.save_aux,
                chunk_size=effective_chunk_size,
            )
            for i, obj in enumerate(bucket):
                all_readout_mem[obj] = readout_memory[:, i]

            if self.save_aux:
                bucket_sensory = self._get_sensory_by_ids(bucket)
                aux_output = {
                    'sensory': bucket_sensory,
                    'pixel_readout': bucket_pixel_readout,
                    'q_logits': aux_features['logits'] if aux_features else None,
                    'q_weights': aux_features['q_weights'] if aux_features else None,
                    'p_weights': aux_features['p_weights'] if aux_features else None,
                    'attn_mask': aux_features['attn_mask'].float() if (aux_features and 'attn_mask' in aux_features and aux_features['attn_mask'] is not None) else None,
                }
                self.aux = aux_output

        return all_readout_mem

    @staticmethod
    def _normalize_as_permanent(
        as_permanent: Literal['no', 'first', 'all'] | bool,
    ) -> Literal['no', 'first', 'all']:
        if isinstance(as_permanent, bool):
            return 'all' if as_permanent else 'no'
        return as_permanent

    def _accumulate_object_summaries(
        self,
        objects: List[int],
        obj_value: torch.Tensor,
    ) -> None:
        for obj_id, obj in enumerate(objects):
            if obj in self.obj_v:
                last_acc = self.obj_v[obj][:, :, -1]
                new_acc = last_acc + obj_value[:, obj_id, :, -1]
                self.obj_v[obj][:, :, :-1] = (
                    self.obj_v[obj][:, :, :-1] + obj_value[:, obj_id, :, :-1]
                )
                self.obj_v[obj][:, :, -1] = new_acc
            else:
                self.obj_v[obj] = obj_value[:, obj_id]

    def add_memory(self,
                   key: torch.Tensor,
                   shrinkage: torch.Tensor,
                   msk_value: torch.Tensor,
                   obj_value: torch.Tensor,
                   objects: List[int],
                   selection: torch.Tensor = None,
                   *,
                   as_permanent: Literal['no', 'first', 'all'] | bool = False) -> None:
        # key: (1/2)*C*H*W
        # msk_value: (1/2)*num_objects*C*H*W
        # obj_value: (1/2)*num_objects*Q*C
        # objects contains a list of object ids corresponding to the objects in msk_value/obj_value
        bs = key.shape[0]
        assert shrinkage.shape[0] == bs
        assert msk_value.shape[0] == bs
        assert obj_value.shape[0] == bs
        if not objects:
            return
        self.add_memory_chunks(
            key,
            shrinkage,
            [(objects, msk_value, obj_value)],
            objects,
            selection=selection,
            as_permanent=as_permanent,
        )

    def add_memory_chunks(
        self,
        key: torch.Tensor,
        shrinkage: torch.Tensor,
        encoded_chunks: Iterable[tuple[List[int], torch.Tensor, torch.Tensor]],
        objects: List[int],
        selection: Optional[torch.Tensor] = None,
        *,
        as_permanent: Literal['no', 'first', 'all'] | bool = 'first',
    ) -> None:
        """Insert a memory frame from `(chunk_objects, msk_value_chunk, obj_value_chunk)` chunks."""
        bs = key.shape[0]
        assert shrinkage.shape[0] == bs
        if not objects:
            return

        chunk_iter = iter(encoded_chunks)
        first_chunk = next(chunk_iter, None)
        if first_chunk is None:
            return
        first_objs, first_msk_value, first_obj_value = first_chunk
        assert first_msk_value.shape[0] == bs
        assert first_obj_value.shape[0] == bs

        raw_h, raw_w = first_msk_value.shape[-2:]
        self.engaged = True
        if self.H is None or self.config_stale:
            self.config_stale = False
            self.H, self.W = raw_h, raw_w
            self.HW = self.H * self.W
            self.max_work_tokens = self.max_mem_frames * self.HW
            if self.use_long_term:
                self.min_work_tokens = self.min_mem_frames * self.HW

        self.CK = key.shape[1]
        self.CV = first_msk_value.shape[2]

        total_objects = len(set(self.work_mem.v.keys()).union(objects))
        target_v_dtype = self._resolve_target_value_dtype(
            total_objects, first_msk_value.dtype
        )
        self.work_mem.cast_values_dtype(target_v_dtype)
        if self.use_long_term:
            self.long_mem.cast_values_dtype(target_v_dtype)

        pool_hw = self._resolve_frame_pool_hw(
            raw_h, raw_w, total_objects, bs, target_v_dtype
        )
        frame_tokens = pool_hw[0] * pool_hw[1]

        key_flat = self._pool_spatial_2d(key, pool_hw).flatten(start_dim=2)
        shrinkage_flat = self._pool_spatial_2d(shrinkage, pool_hw).flatten(start_dim=2)
        selection_flat = (
            self._pool_spatial_2d(selection, pool_hw).flatten(start_dim=2)
            if selection is not None
            else None
        )

        def _value_dict_stream():
            for chunk_objs, chunk_msk_val, chunk_obj_val in itertools.chain(
                [(first_objs, first_msk_value, first_obj_value)],
                chunk_iter,
            ):
                assert chunk_msk_val.shape[0] == bs
                assert chunk_obj_val.shape[0] == bs
                self._accumulate_object_summaries(chunk_objs, chunk_obj_val)
                pooled_val = self._pool_spatial_2d(chunk_msk_val, pool_hw).flatten(
                    start_dim=3
                )
                stored_val = self._cast_value_tensor(pooled_val, target_v_dtype)
                yield {
                    obj: stored_val[:, j]
                    for j, obj in enumerate(chunk_objs)
                }

        resolved_perm = self._normalize_as_permanent(as_permanent)
        self.work_mem.add_streaming(
            key_flat,
            _value_dict_stream(),
            objects,
            shrinkage_flat,
            selection=selection_flat,
            as_permanent=resolved_perm,
        )

        num_stored_objects = len(self.work_mem.v)
        max_perm_tokens = self._max_perm_tokens_for_budget(
            num_stored_objects, bs, target_v_dtype, frame_tokens
        )

        for bucket_id in list(self.work_mem.buckets.keys()):
            if max_perm_tokens is not None:
                self.work_mem.trim_permanent_memory(bucket_id, max_perm_tokens)
            actual_perm = self.work_mem.perm_size(bucket_id)
            (
                eff_max_work_tokens,
                eff_min_work_tokens,
                eff_max_long_tokens,
                eff_num_prototypes,
                eff_buffer_tokens,
                throttled,
            ) = self._effective_token_limits(
                num_stored_objects,
                bs,
                target_v_dtype,
                frame_tokens=frame_tokens,
                perm_tokens=actual_perm,
            )
            if throttled:
                logged = getattr(self, "_logged_kv_budget_keys", None)
                if logged is None:
                    logged = set()
                    self._logged_kv_budget_keys = logged
                budget_key = (
                    num_stored_objects,
                    frame_tokens,
                    eff_max_work_tokens,
                    eff_max_long_tokens,
                )
                if budget_key not in logged:
                    logged.add(budget_key)
                    log.info(
                        "CUTIE adaptive KV budget active (objects=%s, dtype=%s, "
                        "budget=%.1f MB, frame_tokens=%s/%s): "
                        "max_work_tokens=%s, max_long_tokens=%s.",
                        num_stored_objects,
                        target_v_dtype,
                        float(getattr(self, "max_kv_memory_mb", 0.0)),
                        frame_tokens,
                        self.HW,
                        eff_max_work_tokens,
                        eff_max_long_tokens,
                    )

            # long-term memory cleanup
            if self.use_long_term:
                if (
                    self.long_mem.engaged(bucket_id)
                    and self.long_mem.non_perm_size(bucket_id) > eff_max_long_tokens
                ):
                    keep_lt = max(1, eff_max_long_tokens - eff_buffer_tokens)
                    if self.long_mem.non_perm_size(bucket_id) > keep_lt:
                        self.long_mem.remove_obsolete_features(bucket_id, keep_lt)
                # Do memory compressed if needed
                if self.work_mem.non_perm_size(bucket_id) >= eff_max_work_tokens:
                    # Remove obsolete features if needed
                    if self.long_mem.engaged(bucket_id) and (
                        self.long_mem.non_perm_size(bucket_id)
                        >= (eff_max_long_tokens - eff_num_prototypes)
                    ):
                        keep_lt = max(
                            1,
                            eff_max_long_tokens
                            - eff_num_prototypes
                            - eff_buffer_tokens,
                        )
                        if self.long_mem.non_perm_size(bucket_id) > keep_lt:
                            self.long_mem.remove_obsolete_features(
                                bucket_id,
                                keep_lt,
                            )

                    self.compress_features(
                        bucket_id,
                        min_work_tokens=eff_min_work_tokens,
                        num_prototypes=eff_num_prototypes,
                        value_dtype=target_v_dtype,
                    )
            else:
                # FIFO
                self.work_mem.remove_old_memory(
                    bucket_id, eff_max_work_tokens)

    def purge_except(self, obj_keep_idx: List[int]) -> None:
        # purge certain objects from the memory except the one listed
        self.work_mem.purge_except(obj_keep_idx)
        if self.use_long_term and self.long_mem.engaged():
            self.long_mem.purge_except(obj_keep_idx)
        self.sensory = {k: v for k, v in self.sensory.items()
                        if k in obj_keep_idx}
        self.obj_v = {k: v for k, v in self.obj_v.items()
                      if k in obj_keep_idx}

        if not self.work_mem.engaged():
            # everything is removed!
            self.engaged = False

    def compress_features(
        self,
        bucket_id: int,
        *,
        min_work_tokens: Optional[int] = None,
        num_prototypes: Optional[int] = None,
        value_dtype: Optional[torch.dtype] = None,
    ) -> None:
        resolved_min_work = (
            self.min_work_tokens if min_work_tokens is None else int(min_work_tokens)
        )
        # perform memory consolidation
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(bucket_id, 0, -resolved_min_work),
            num_prototypes=num_prototypes,
            value_dtype=value_dtype,
        )

        # remove consolidated working memory
        self.work_mem.sieve_by_range(bucket_id,
                                     0,
                                     -resolved_min_work,
                                     min_size=resolved_min_work)

        # add to long-term memory
        self.long_mem.add(prototype_key,
                          prototype_value,
                          prototype_shrinkage,
                          selection=None,
                          supposed_bucket_id=bucket_id)

    def consolidation(self, candidate_key: torch.Tensor, candidate_shrinkage: torch.Tensor,
                      candidate_selection: torch.Tensor, candidate_value: Dict[int, torch.Tensor],
                      usage: torch.Tensor, *,
                      num_prototypes: Optional[int] = None,
                      value_dtype: Optional[torch.dtype] = None) -> (torch.Tensor, Dict[int, torch.Tensor], torch.Tensor):
        # find the indices with max usage
        bs = candidate_key.shape[0]
        assert bs in [1, 2]
        resolved_prototypes = max(
            1,
            min(
                int(self.num_prototypes if num_prototypes is None else num_prototypes),
                int(usage.shape[-1]),
            ),
        )

        prototype_key = []
        prototype_selection = []
        for bi in range(bs):
            _, max_usage_indices = torch.topk(
                usage[bi], k=resolved_prototypes, dim=-1, sorted=True)
            prototype_indices = max_usage_indices.flatten()
            prototype_key.append(candidate_key[bi, :, prototype_indices])
            prototype_selection.append(
                candidate_selection[bi, :, prototype_indices])
        prototype_key = torch.stack(prototype_key, dim=0)
        prototype_selection = torch.stack(prototype_selection, dim=0)
        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, candidate_shrinkage, prototype_key,
                                    prototype_selection)
        affinity = do_softmax(similarity)

        # readout the values
        prototype_value = {
            k: (
                self._cast_value_tensor(self._readout(affinity, v), value_dtype)
                if value_dtype is not None
                else self._readout(affinity, v)
            )
            for k, v in candidate_value.items()
        }

        # readout the shrinkage term
        prototype_shrinkage = self._readout(affinity, candidate_shrinkage)

        return prototype_key, prototype_value, prototype_shrinkage

    def initialize_sensory_if_needed(self, sample_key: torch.Tensor, ids: List[int]):
        for obj in ids:
            if obj not in self.sensory:
                # also initializes the sensory memory
                bs, _, h, w = sample_key.shape
                self.sensory[obj] = torch.zeros((bs, self.sensory_dim, h, w),
                                                device=sample_key.device)

    def update_sensory(self, sensory: torch.Tensor, ids: List[int]):
        # sensory: 1*num_objects*C*H*W
        for obj_id, obj in enumerate(ids):
            self.sensory[obj] = sensory[:, obj_id]

    def get_sensory(self, ids: List[int]):
        # returns (1/2)*num_objects*C*H*W
        return self._get_sensory_by_ids(ids)

    def clear_non_permanent_memory(self):
        self.work_mem.clear_non_permanent_memory()
        if self.use_long_term:
            self.long_mem.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.sensory = {}
