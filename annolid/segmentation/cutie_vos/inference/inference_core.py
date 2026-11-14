from typing import List, Optional, Iterable, Dict
import logging
from omegaconf import DictConfig

import numpy as np
import torch
import torch.nn.functional as F

from annolid.segmentation.cutie_vos.inference.memory_manager import MemoryManager
from annolid.segmentation.cutie_vos.inference.object_manager import ObjectManager
from annolid.segmentation.cutie_vos.inference.image_feature_store import ImageFeatureStore
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.utils.tensor_utils import pad_divide_by, unpad, aggregate

log = logging.getLogger()


class InferenceCore:
    DEFAULT_AUTO_CHUNK_SIZE = 16

    def __init__(self,
                 network: CUTIE,
                 cfg: DictConfig,
                 *,
                 image_feature_store: ImageFeatureStore = None):
        self.network = network
        self.cfg = cfg
        self.mem_every = cfg.mem_every
        stagger_updates = cfg.stagger_updates
        self.chunk_size = cfg.chunk_size
        self.save_aux = cfg.save_aux
        self.max_internal_size = cfg.max_internal_size
        self.flip_aug = cfg.flip_aug

        self.curr_ti = -1
        self.last_mem_ti = 0
        # at which time indices should we update the sensory memory
        if stagger_updates >= self.mem_every:
            self.stagger_ti = set(range(1, self.mem_every + 1))
        else:
            self.stagger_ti = set(
                np.round(np.linspace(1, self.mem_every, stagger_updates)).astype(int))
        self.object_manager = ObjectManager()
        self.memory = MemoryManager(
            cfg=cfg, object_manager=self.object_manager)

        if image_feature_store is None:
            self.image_feature_store = ImageFeatureStore(self.network)
        else:
            self.image_feature_store = image_feature_store

        self.last_mask = None

    def _effective_chunk_size(self, num_objects: int) -> int:
        """Return explicit `chunk_size`, or auto-chunk large object batches (`> 16`)."""
        chunk_size = getattr(self, "chunk_size", -1)
        if chunk_size is not None and int(chunk_size) > 0:
            return int(chunk_size)
        if int(num_objects) > self.DEFAULT_AUTO_CHUNK_SIZE:
            return self.DEFAULT_AUTO_CHUNK_SIZE
        return -1

    @staticmethod
    def _downsample_last_mask_to_stride16(last_mask: torch.Tensor) -> torch.Tensor:
        """Store `last_mask` at stride 16 (`MemoryManager.read` resolution) between steps."""
        h, w = last_mask.shape[-2:]
        if h >= 16 and w >= 16 and (h % 16 == 0) and (w % 16 == 0):
            target = (h // 16, w // 16)
            if (h, w) != target:
                return F.interpolate(last_mask, size=target, mode="area")
        return last_mask

    @staticmethod
    def _normalize_mask_objects(
        mask: torch.Tensor,
        objects: Optional[List[int]],
        *,
        idx_mask: bool,
    ) -> List[int]:
        """Validate mask metadata and return normalized positive object ids."""
        if not isinstance(mask, torch.Tensor):
            raise TypeError(
                f"CUTIE masks must be torch tensors; received {type(mask)!r}."
            )
        expected_ndim = 2 if idx_mask else 3
        if mask.ndim != expected_ndim:
            mode = "indexed" if idx_mask else "per-object"
            raise ValueError(
                f"CUTIE {mode} masks must have {expected_ndim} dimensions; "
                f"received shape {tuple(mask.shape)}."
            )

        indexed_values: List[int] = []
        if idx_mask:
            if mask.is_complex():
                raise ValueError("CUTIE indexed masks cannot contain complex values.")
            inferred = torch.unique(mask.detach()).tolist()
            for value in inferred:
                numeric_value = float(value)
                object_id = int(numeric_value)
                if numeric_value != object_id:
                    raise ValueError(
                        "CUTIE indexed masks must contain integer object ids; "
                        f"received {value!r}."
                    )
                if object_id < 0:
                    raise ValueError(
                        "CUTIE indexed masks cannot contain negative object ids; "
                        f"received {object_id}."
                    )
                if object_id != 0:
                    indexed_values.append(object_id)

        if objects is None:
            if idx_mask:
                objects = indexed_values
            else:
                objects = list(range(1, mask.shape[0] + 1))

        normalized: List[int] = []
        for obj in objects:
            if isinstance(obj, bool):
                raise ValueError("CUTIE object ids must be positive integers, not booleans.")
            try:
                object_id = int(obj)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CUTIE object ids must be positive integers; received {obj!r}."
                ) from exc
            if isinstance(obj, (float, np.floating)) and not float(obj).is_integer():
                raise ValueError(
                    f"CUTIE object ids must be integers; received {obj!r}."
                )
            if object_id <= 0:
                raise ValueError(
                    f"CUTIE object ids must be positive; received {object_id}."
                )
            normalized.append(object_id)

        if len(set(normalized)) != len(normalized):
            raise ValueError("CUTIE object ids must not contain duplicates.")
        if not normalized:
            raise ValueError("CUTIE masks must contain at least one foreground object.")
        unregistered_values = set(indexed_values) - set(normalized)
        if unregistered_values:
            raise ValueError(
                "CUTIE indexed mask contains object ids missing from the objects list: "
                f"{sorted(unregistered_values)}."
            )
        if not idx_mask and mask.shape[0] != len(normalized):
            raise ValueError(
                "CUTIE per-object mask channels must match the objects list: "
                f"received {mask.shape[0]} channel(s) for {len(normalized)} object(s)."
            )
        return normalized

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryManager(
            cfg=self.cfg, object_manager=self.object_manager)

    def clear_non_permanent_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_non_permanent_memory()

    def clear_sensory_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory.clear_sensory_memory()

    def update_config(self, cfg):
        self.mem_every = cfg['mem_every']
        self.memory.update_config(cfg)

    @staticmethod
    def _resize_index_mask(
        mask: torch.Tensor,
        size: tuple[int, int],
        *,
        dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=size,
            mode='nearest-exact',
        )[0, 0]
        return resized.round().to(dtype=dtype)

    @classmethod
    def _resize_input_mask(
        cls,
        mask: torch.Tensor,
        size: tuple[int, int],
        *,
        idx_mask: bool,
    ) -> torch.Tensor:
        if tuple(mask.shape[-2:]) == tuple(size):
            return mask
        if idx_mask:
            return cls._resize_index_mask(mask, size, dtype=mask.dtype)
        return F.interpolate(
            mask.unsqueeze(0),
            size=size,
            mode='bilinear',
            align_corners=False,
        )[0]

    @classmethod
    def _finalize_output(
        cls,
        output_prob: torch.Tensor,
        *,
        output_size: Optional[tuple[int, int]] = None,
        return_index_mask: bool = False,
    ) -> torch.Tensor:
        if return_index_mask:
            output_mask = torch.argmax(output_prob, dim=0)
            if (
                output_size is not None
                and tuple(output_mask.shape[-2:]) != tuple(output_size)
            ):
                return cls._resize_index_mask(
                    output_mask,
                    output_size,
                    dtype=torch.int32,
                )
            return output_mask.to(dtype=torch.int32)

        if (
            output_size is not None
            and tuple(output_prob.shape[-2:]) != tuple(output_size)
        ):
            output_prob = F.interpolate(
                output_prob.unsqueeze(0),
                size=output_size,
                mode='bilinear',
                align_corners=False,
            )[0]
        return output_prob

    def _can_stream_memory_chunks(self, num_objects: int) -> bool:
        """Return True when chunk-streaming `encode_mask_chunk` + `add_memory_chunks` is safe."""
        return (
            isinstance(getattr(self, "network", None), CUTIE)
            and not self.save_aux
            and self._effective_chunk_size(num_objects) > 0
            and getattr(getattr(self.network, "encode_mask", None), "__func__", None)
            is CUTIE.encode_mask
            and getattr(getattr(self.memory, "add_memory", None), "__func__", None)
            is MemoryManager.add_memory
            and hasattr(self.memory, "add_memory_chunks")
        )

    def _add_memory(self,
                    image: torch.Tensor,
                    pix_feat: torch.Tensor,
                    prob: torch.Tensor,
                    key: torch.Tensor,
                    shrinkage: torch.Tensor,
                    selection: torch.Tensor,
                    *,
                    is_deep_update: bool = True,
                    force_permanent: bool = False) -> None:
        """
        Memorize the given segmentation in all memory stores.

        The batch dimension is 1 if flip augmentation is not used.
        image: RGB image, (1/2)*3*H*W
        pix_feat: from the key encoder, (1/2)*_*H*W
        prob: (1/2)*num_objects*H*W, in [0, 1]
        key/shrinkage/selection: for anisotropic l2, (1/2)*_*H*W
        selection can be None if not using long-term memory
        is_deep_update: whether to use deep update (e.g. with the mask encoder)
        force_permanent: whether to force the memory to be permanent
        """
        num_objects = prob.shape[1]
        if num_objects == 0:
            # nothing to add
            log.warn('Trying to add an empty object mask to memory!')
            return

        if force_permanent:
            as_permanent = 'all'
        else:
            as_permanent = 'first'

        effective_chunk_size = self._effective_chunk_size(num_objects)
        all_obj_ids = self.object_manager.all_obj_ids
        self.memory.initialize_sensory_if_needed(key, all_obj_ids)

        if (
            self._can_stream_memory_chunks(num_objects)
            and num_objects > effective_chunk_size
        ):
            norm_image = self.network.normalize_image(image)
            total_masks = (
                None
                if self.network.single_object
                else prob.sum(dim=1, keepdim=True)
            )

            def _encoded_chunks():
                for start in range(0, num_objects, effective_chunk_size):
                    end = min(num_objects, start + effective_chunk_size)
                    chunk_ids = all_obj_ids[start:end]
                    prob_chunk = prob[:, start:end]
                    others_chunk = (
                        None
                        if total_masks is None
                        else (total_masks - prob_chunk).clamp(0, 1)
                    )
                    sensory_chunk = self.memory.get_sensory(chunk_ids)
                    (
                        msk_val_chunk,
                        new_sensory_chunk,
                        obj_val_chunk,
                        _,
                    ) = self.network.encode_mask_chunk(
                        norm_image,
                        pix_feat,
                        sensory_chunk,
                        prob_chunk,
                        others_chunk,
                        deep_update=is_deep_update,
                        need_weights=False,
                    )
                    if is_deep_update:
                        self.memory.update_sensory(new_sensory_chunk, chunk_ids)
                    yield chunk_ids, msk_val_chunk, obj_val_chunk

            self.memory.add_memory_chunks(
                key,
                shrinkage,
                _encoded_chunks(),
                all_obj_ids,
                selection=selection,
                as_permanent=as_permanent,
            )
            self.obj_logits = None
            self.last_mem_ti = self.curr_ti
            return

        msk_value, sensory, obj_value, self.obj_logits = self.network.encode_mask(
            image,
            pix_feat,
            self.memory.get_sensory(all_obj_ids),
            prob,
            deep_update=is_deep_update,
            chunk_size=effective_chunk_size,
            need_weights=self.save_aux)
        self.memory.add_memory(key,
                               shrinkage,
                               msk_value,
                               obj_value,
                               all_obj_ids,
                               selection=selection,
                               as_permanent=as_permanent)
        self.last_mem_ti = self.curr_ti
        if is_deep_update:
            self.memory.update_sensory(
                sensory, all_obj_ids)

    def _segment(self,
                 key: torch.Tensor,
                 selection: torch.Tensor,
                 pix_feat: torch.Tensor,
                 ms_features: Iterable[torch.Tensor],
                 update_sensory: bool = True) -> torch.Tensor:
        """
        Produce a segmentation using the given features and the memory

        The batch dimension is 1 if flip augmentation is not used.
        key/selection: for anisotropic l2: (1/2) * _ * H * W
        pix_feat: from the key encoder, (1/2) * _ * H * W
        ms_features: an iterable of multiscale features from the encoder, each is (1/2)*_*H*W
                      with strides 16, 8, and 4 respectively
        update_sensory: whether to update the sensory memory

        Returns: (num_objects+1)*H*W normalized probability; the first channel is the background
        """
        bs = key.shape[0]
        if self.flip_aug:
            assert bs == 2
        else:
            assert bs == 1

        if not self.memory.engaged:
            log.warn('Trying to segment without any memory!')
            return torch.zeros((1, key.shape[-2] * 16, key.shape[-1] * 16),
                               device=key.device,
                               dtype=key.dtype)

        effective_chunk_size = self._effective_chunk_size(
            self.object_manager.num_obj
        )
        memory_readout = self.memory.read(
            pix_feat,
            key,
            selection,
            self.last_mask,
            self.network,
            chunk_size=effective_chunk_size,
        )
        memory_readout = self.object_manager.realize_dict(memory_readout)
        segment_kwargs = dict(
            chunk_size=effective_chunk_size,
            update_sensory=update_sensory,
        )
        if isinstance(self.network, CUTIE):
            segment_kwargs["return_logits"] = False
        sensory, _, pred_prob_with_bg = self.network.segment(
            ms_features,
            memory_readout,
            self.memory.get_sensory(self.object_manager.all_obj_ids),
            **segment_kwargs,
        )
        # remove batch dim
        if self.flip_aug:
            # average predictions of the non-flipped and flipped version
            pred_prob_with_bg = (pred_prob_with_bg[0] +
                                 torch.flip(pred_prob_with_bg[1], dims=[-1])) / 2
        else:
            pred_prob_with_bg = pred_prob_with_bg[0]
        if update_sensory:
            self.memory.update_sensory(
                sensory, self.object_manager.all_obj_ids)
        return pred_prob_with_bg

    def _segment_compact(
        self,
        key: torch.Tensor,
        selection: torch.Tensor,
        pix_feat: torch.Tensor,
        ms_features: Iterable[torch.Tensor],
        update_sensory: bool = True,
        *,
        memorize: bool = False,
        image: Optional[torch.Tensor] = None,
        shrinkage: Optional[torch.Tensor] = None,
        force_permanent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stream temporary-id argmax + stride-16 `last_mask` (and optional memory commit) in `O(chunk_size * H * W)`."""
        if not self.memory.engaged:
            log.warn('Trying to segment without any memory!')
            h16, w16 = key.shape[-2], key.shape[-1]
            empty_idx = torch.zeros(
                (h16 * 16, w16 * 16), device=key.device, dtype=torch.int32
            )
            empty_last = torch.zeros(
                (1, max(1, self.object_manager.num_obj), h16, w16),
                device=key.device,
                dtype=key.dtype,
            )
            return empty_idx, empty_last

        all_obj_ids = self.object_manager.all_obj_ids
        num_objects = len(all_obj_ids)
        effective_chunk_size = self._effective_chunk_size(num_objects)
        memory_readout = self.memory.read(
            pix_feat,
            key,
            selection,
            self.last_mask,
            self.network,
            chunk_size=effective_chunk_size,
        )
        memory_readout = self.object_manager.realize_dict(memory_readout)

        if not memorize or num_objects == 0:
            sensory, best_idx, last_mask_s16 = self.network.segment_compact(
                ms_features,
                memory_readout,
                self.memory.get_sensory(all_obj_ids),
                chunk_size=effective_chunk_size,
                update_sensory=update_sensory,
            )
            if update_sensory:
                self.memory.update_sensory(sensory, all_obj_ids)
            return best_idx[0], last_mask_s16

        assert image is not None and shrinkage is not None
        sensory, logits_s4, m, d, best_idx = self.network._decode_compact_pass1(
            ms_features,
            memory_readout,
            self.memory.get_sensory(all_obj_ids),
            chunk_size=effective_chunk_size,
            update_sensory=update_sensory,
        )
        if update_sensory:
            self.memory.update_sensory(sensory, all_obj_ids)

        bs, _, h4, w4 = logits_s4.shape
        h1, w1 = h4 * 4, w4 * 4
        h16, w16 = max(1, h1 // 16), max(1, w1 // 16)
        step = max(
            1,
            int(effective_chunk_size)
            if effective_chunk_size and int(effective_chunk_size) > 0
            else num_objects,
        )
        last_mask_s16 = logits_s4.new_empty((bs, num_objects, h16, w16))

        self.memory.initialize_sensory_if_needed(key, all_obj_ids)
        norm_image = self.network.normalize_image(image)
        if self.network.single_object:
            total_masks = None
        else:
            bg_logit = F.interpolate(
                logits_s4[:, 0:1],
                scale_factor=4,
                mode='bilinear',
                align_corners=False,
            )
            total_masks = (1.0 - torch.exp(bg_logit - m) / d).clamp(0.0, 1.0)

        def _memorize_chunks():
            for fg0 in range(0, num_objects, step):
                fg1 = min(num_objects, fg0 + step)
                chunk_ids = all_obj_ids[fg0:fg1]
                chunk = F.interpolate(
                    logits_s4[:, 1 + fg0 : 1 + fg1],
                    scale_factor=4,
                    mode='bilinear',
                    align_corners=False,
                )
                prob_chunk = torch.exp(chunk - m) / d
                last_mask_s16[:, fg0:fg1] = F.interpolate(
                    prob_chunk, size=(h16, w16), mode='area'
                )
                others_chunk = (
                    None
                    if total_masks is None
                    else (total_masks - prob_chunk).clamp(0, 1)
                )
                sensory_chunk = self.memory.get_sensory(chunk_ids)
                (
                    msk_val_chunk,
                    new_sensory_chunk,
                    obj_val_chunk,
                    _,
                ) = self.network.encode_mask_chunk(
                    norm_image,
                    pix_feat,
                    sensory_chunk,
                    prob_chunk,
                    others_chunk,
                    deep_update=True,
                    need_weights=False,
                )
                self.memory.update_sensory(new_sensory_chunk, chunk_ids)
                yield chunk_ids, msk_val_chunk, obj_val_chunk

        as_permanent = 'all' if force_permanent else 'first'
        self.memory.add_memory_chunks(
            key,
            shrinkage,
            _memorize_chunks(),
            all_obj_ids,
            selection=selection,
            as_permanent=as_permanent,
        )
        self.obj_logits = None
        self.last_mem_ti = self.curr_ti
        return best_idx[0], last_mask_s16

    def _step_indexed_seed_compact(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        objects: List[int],
        pix_feat: torch.Tensor,
        key: torch.Tensor,
        shrinkage: torch.Tensor,
        selection: Optional[torch.Tensor],
        *,
        memorize: bool,
        force_permanent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stream an indexed seed frame without materializing `(N, H, W)` one-hot/prob buffers."""
        self.object_manager.add_new_objects(objects)
        padded_mask, _ = pad_divide_by(mask, 16)
        tmp_mask = self.object_manager.obj_to_tmp_cls(padded_mask)

        all_obj_ids = self.object_manager.all_obj_ids
        num_objects = len(all_obj_ids)
        step = max(1, self._effective_chunk_size(num_objects))

        # Calibrate exact `aggregate` + `softmax` values for binary mutually-exclusive
        # masks on a 2-pixel reference tensor `(num_objects, 2)` in `O(num_objects)`.
        ref_one_hot = torch.zeros(
            (num_objects, 2), dtype=image.dtype, device=image.device
        )
        ref_one_hot[0, 1] = 1.0
        ref_prob = torch.softmax(aggregate(ref_one_hot, dim=0), dim=0)
        p_fg_at_bg = ref_prob[1, 0]
        p_owner_at_fg = ref_prob[1, 1]
        p_other_at_fg = ref_prob[2, 1] if num_objects >= 2 else ref_prob[1, 0]

        has_fg = tmp_mask > 0
        base_non_owner = (
            torch.where(has_fg, p_other_at_fg, p_fg_at_bg)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        h1, w1 = tmp_mask.shape[-2:]
        downsample_s16 = h1 >= 16 and w1 >= 16 and (h1 % 16 == 0) and (w1 % 16 == 0)
        h16, w16 = (h1 // 16, w1 // 16) if downsample_s16 else (h1, w1)
        last_mask_s16 = torch.empty(
            (1, num_objects, h16, w16), dtype=ref_prob.dtype, device=image.device
        )

        def _make_prob_chunk(fg0: int, fg1: int) -> torch.Tensor:
            chunk_tmp_ids = torch.arange(
                fg0 + 1,
                fg1 + 1,
                device=tmp_mask.device,
                dtype=tmp_mask.dtype,
            ).view(1, -1, 1, 1)
            is_owner = tmp_mask.unsqueeze(0).unsqueeze(0) == chunk_tmp_ids
            prob_chunk = torch.where(is_owner, p_owner_at_fg, base_non_owner)
            last_mask_s16[:, fg0:fg1] = (
                F.interpolate(prob_chunk, size=(h16, w16), mode='area')
                if downsample_s16
                else prob_chunk
            )
            return prob_chunk

        if memorize:
            self.memory.initialize_sensory_if_needed(key, all_obj_ids)
            norm_image = self.network.normalize_image(image)
            if self.network.single_object:
                total_masks = None
            else:
                total_fg_at_bg = ref_prob[1:, 0].sum()
                total_fg_at_fg = ref_prob[1:, 1].sum()
                total_masks = (
                    torch.where(has_fg, total_fg_at_fg, total_fg_at_bg)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

            def _seed_chunks():
                for fg0 in range(0, num_objects, step):
                    fg1 = min(num_objects, fg0 + step)
                    chunk_ids = all_obj_ids[fg0:fg1]
                    prob_chunk = _make_prob_chunk(fg0, fg1)
                    others_chunk = (
                        None
                        if total_masks is None
                        else (total_masks - prob_chunk).clamp(0, 1)
                    )
                    sensory_chunk = self.memory.get_sensory(chunk_ids)
                    (
                        msk_val_chunk,
                        new_sensory_chunk,
                        obj_val_chunk,
                        _,
                    ) = self.network.encode_mask_chunk(
                        norm_image,
                        pix_feat,
                        sensory_chunk,
                        prob_chunk,
                        others_chunk,
                        deep_update=True,
                        need_weights=False,
                    )
                    self.memory.update_sensory(new_sensory_chunk, chunk_ids)
                    yield chunk_ids, msk_val_chunk, obj_val_chunk

            as_permanent = 'all' if force_permanent else 'first'
            self.memory.add_memory_chunks(
                key,
                shrinkage,
                _seed_chunks(),
                all_obj_ids,
                selection=selection,
                as_permanent=as_permanent,
            )
            self.obj_logits = None
            self.last_mem_ti = self.curr_ti
        else:
            for fg0 in range(0, num_objects, step):
                _make_prob_chunk(fg0, min(num_objects, fg0 + step))

        return tmp_mask, last_mask_s16

    def step(self,
             image: torch.Tensor,
             mask: Optional[torch.Tensor] = None,
             objects: Optional[List[int]] = None,
             *,
             idx_mask: bool = True,
             end: bool = False,
             delete_buffer: bool = True,
             force_permanent: bool = False,
             return_index_mask: bool = False,
             resize_output: bool = True,
             output_size: Optional[tuple[int, int]] = None) -> torch.Tensor:
        """
        Take a step with a new incoming image.
        If there is an incoming mask with new objects, we will memorize them.
        If there is no incoming mask, we will segment the image using the memory.
        In both cases, we will update the memory and return a segmentation.

        image: 3*H*W
        mask: H*W (if idx mask) or len(objects)*H*W or None
        objects: list of object ids that are valid in the mask Tensor.
                The ids themselves do not need to be consecutive/in order, but they need to be
                in the same position in the list as the corresponding mask
                in the tensor in non-idx-mask mode.
                objects is ignored if the mask is None.
                If idx_mask is False and objects is None, we sequentially infer the object ids.
        idx_mask: if True, mask is expected to contain an object id at every pixel.
                  If False, mask should have multiple channels with each channel representing one object.
        end: if we are at the end of the sequence, we do not need to update memory
            if unsure just set it to False
        delete_buffer: whether to delete the image feature buffer after this step
        force_permanent: the memory recorded this frame will be added to the permanent memory
        return_index_mask: return one integer temporary-id mask instead of
            per-object probabilities. This avoids restoring every probability
            channel to the original image size when callers only need the winner.
        resize_output: restore the returned output to the input image size after
            internal inference resizing. Disable when the caller discards the output.
        output_size: optional explicit output height and width when the caller
            resized the input before tensor conversion.
        """
        if mask is not None:
            objects = self._normalize_mask_objects(
                mask,
                objects,
                idx_mask=idx_mask,
            )

        # resize input if needed -- currently only used for the GUI
        resize_needed = False
        if self.max_internal_size > 0:
            h, w = image.shape[-2:]
            min_side = min(h, w)
            if min_side > self.max_internal_size:
                resize_needed = True
                new_h = int(h / min_side * self.max_internal_size)
                new_w = int(w / min_side * self.max_internal_size)
                image = F.interpolate(image.unsqueeze(0),
                                      size=(new_h, new_w),
                                      mode='bilinear',
                                      align_corners=False)[0]
                if mask is not None:
                    mask = self._resize_input_mask(
                        mask,
                        (new_h, new_w),
                        idx_mask=idx_mask,
                    )
        if mask is not None:
            mask = self._resize_input_mask(
                mask,
                tuple(image.shape[-2:]),
                idx_mask=idx_mask,
            )
            if mask.device != image.device:
                mask = mask.to(image.device, non_blocking=True)

        self.curr_ti += 1

        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension
        if self.flip_aug:
            image = torch.cat([image, torch.flip(image, dims=[-1])], dim=0)

        # whether to update the working memory
        is_mem_frame = ((self.curr_ti - self.last_mem_ti >= self.mem_every) or
                        (mask is not None)) and (not end)
        # Segment when there is no input mask or when an input correction does
        # not cover every object that was already active. New objects alone do
        # not replace the predictions for existing objects.
        active_object_ids = set(self.object_manager.all_obj_ids)
        provided_object_ids = set(objects or [])
        need_segment = mask is None or bool(active_object_ids - provided_object_ids)
        update_sensory = ((self.curr_ti - self.last_mem_ti)
                          in self.stagger_ti) and (not end)

        # encoding the image
        want_memorize = bool(is_mem_frame or force_permanent)
        default_add_memory = (
            getattr(getattr(self, "_add_memory", None), "__func__", None)
            is InferenceCore._add_memory
        )
        use_compact_segment = (
            need_segment
            and mask is None
            and return_index_mask
            and not self.flip_aug
            and self._effective_chunk_size(self.object_manager.num_obj) > 0
            and getattr(getattr(self, "_segment", None), "__func__", None)
            is InferenceCore._segment
            and hasattr(getattr(self, "network", None), "segment_compact")
            and (
                not want_memorize
                or (
                    default_add_memory
                    and self._can_stream_memory_chunks(self.object_manager.num_obj)
                    and hasattr(self.network, "_decode_compact_pass1")
                )
            )
        )
        projected_seed_objects = len(active_object_ids | provided_object_ids)
        seed_chunk_size = self._effective_chunk_size(projected_seed_objects)
        use_compact_indexed_seed = (
            mask is not None
            and idx_mask
            and not need_segment
            and return_index_mask
            and not self.flip_aug
            and seed_chunk_size > 0
            and projected_seed_objects > seed_chunk_size
            and (
                not want_memorize
                or (
                    default_add_memory
                    and self._can_stream_memory_chunks(projected_seed_objects)
                )
            )
            and isinstance(getattr(self, "network", None), CUTIE)
        )
        compact_index_mask: Optional[torch.Tensor] = None
        try:
            ms_feat, pix_feat = self.image_feature_store.get_features(
                self.curr_ti, image)
            key, shrinkage, selection = self.image_feature_store.get_key(
                self.curr_ti, image)

            if use_compact_segment:
                compact_index_mask, self.last_mask = self._segment_compact(
                    key,
                    selection,
                    pix_feat,
                    ms_feat,
                    update_sensory=update_sensory,
                    memorize=want_memorize,
                    image=image,
                    shrinkage=shrinkage,
                    force_permanent=force_permanent,
                )
            elif use_compact_indexed_seed:
                compact_index_mask, self.last_mask = self._step_indexed_seed_compact(
                    image,
                    mask,
                    objects,
                    pix_feat,
                    key,
                    shrinkage,
                    selection,
                    memorize=want_memorize,
                    force_permanent=force_permanent,
                )
            else:
                # segmentation from memory if needed
                if need_segment:
                    pred_prob_with_bg = self._segment(key,
                                                      selection,
                                                      pix_feat,
                                                      ms_feat,
                                                      update_sensory=update_sensory)

                # use the input mask if provided
                if mask is not None:
                    # Temporary ids identify tensor channels and can differ from
                    # both stable object ids and the order supplied by the caller.
                    corresponding_tmp_ids, _ = self.object_manager.add_new_objects(
                        objects)

                    mask, _ = pad_divide_by(mask, 16)
                    if need_segment:
                        # Merge predicted masks with the supplied correction while
                        # retaining channels for every existing and new object.
                        pred_prob_no_bg = pred_prob_with_bg[1:].clone()
                        if idx_mask:
                            occupied = mask > 0
                        else:
                            occupied = mask.max(0).values > 0.5
                        pred_prob_no_bg[:, occupied] = 0

                        merged_mask = pred_prob_no_bg.new_zeros(
                            (
                                self.object_manager.num_obj,
                                *pred_prob_no_bg.shape[-2:],
                            )
                        )
                        merged_mask[:pred_prob_no_bg.shape[0]] = pred_prob_no_bg
                        if (
                            idx_mask
                            and objects
                            and not mask.is_floating_point()
                            and not mask.is_complex()
                            and max(objects) <= ObjectManager._MAX_DIRECT_LUT_ID
                        ):
                            mask_min = int(mask.min().item()) if mask.numel() > 0 else 0
                            mask_max = int(mask.max().item()) if mask.numel() > 0 else 0
                            if mask_min >= 0 and mask_max <= ObjectManager._MAX_DIRECT_LUT_ID:
                                for tmp_id in corresponding_tmp_ids:
                                    merged_mask[tmp_id - 1].zero_()
                                lut = torch.zeros(
                                    max(max(objects), mask_max) + 1,
                                    dtype=torch.int64,
                                )
                                for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                                    lut[objects[mask_id]] = tmp_id
                                if mask.device.type != "cpu":
                                    lut = lut.to(mask.device)
                                mapped = lut[mask.long()]
                                fg_pixels = mapped > 0
                                if torch.any(fg_pixels):
                                    ys, xs = torch.where(fg_pixels)
                                    merged_mask[mapped[ys, xs] - 1, ys, xs] = 1.0
                            else:
                                for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                                    merged_mask[tmp_id - 1] = (
                                        mask == objects[mask_id]
                                    ).type_as(pred_prob_no_bg)
                        else:
                            for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                                if idx_mask:
                                    this_mask = (mask == objects[mask_id]).type_as(
                                        pred_prob_no_bg)
                                else:
                                    this_mask = mask[mask_id].type_as(pred_prob_no_bg)
                                merged_mask[tmp_id - 1] = this_mask
                        mask = merged_mask
                    elif idx_mask:
                        mask = self.object_manager.make_one_hot(mask)
                    else:
                        ordered_mask = mask.new_zeros(
                            (self.object_manager.num_obj, *mask.shape[-2:])
                        )
                        for mask_id, tmp_id in enumerate(corresponding_tmp_ids):
                            ordered_mask[tmp_id - 1] = mask[mask_id]
                        mask = ordered_mask
                    pred_prob_with_bg = aggregate(mask, dim=0)
                    pred_prob_with_bg = torch.softmax(pred_prob_with_bg, dim=0)

                self.last_mask = pred_prob_with_bg[1:].unsqueeze(0)
                if self.flip_aug:
                    self.last_mask = torch.cat(
                        [self.last_mask, torch.flip(self.last_mask, dims=[-1])], dim=0)

                # save as memory if needed
                if is_mem_frame or force_permanent:
                    self._add_memory(image,
                                     pix_feat,
                                     self.last_mask,
                                     key,
                                     shrinkage,
                                     selection,
                                     force_permanent=force_permanent)
                self.last_mask = self._downsample_last_mask_to_stride16(
                    self.last_mask
                )
        except BaseException:
            if delete_buffer:
                self.image_feature_store.delete(self.curr_ti)
            raise

        if delete_buffer:
            self.image_feature_store.delete(self.curr_ti)

        resolved_output_size = output_size
        if resolved_output_size is None and resize_needed:
            resolved_output_size = (h, w)
        if not resize_output:
            resolved_output_size = None

        if compact_index_mask is not None:
            output_mask = unpad(compact_index_mask, self.pad)
            if (
                resolved_output_size is not None
                and tuple(output_mask.shape[-2:]) != tuple(resolved_output_size)
            ):
                return self._resize_index_mask(
                    output_mask,
                    resolved_output_size,
                    dtype=torch.int32,
                )
            return output_mask.to(dtype=torch.int32)

        output_prob = unpad(pred_prob_with_bg, self.pad)
        return self._finalize_output(
            output_prob,
            output_size=resolved_output_size,
            return_index_mask=return_index_mask,
        )

    def delete_objects(self, objects: List[int]) -> None:
        """
        Delete the given objects from the memory.
        """
        removed = {int(obj) for obj in objects}
        kept_tmp_ids = [
            tmp_id
            for tmp_id, obj in self.object_manager.tmp_id_to_obj.items()
            if obj.id not in removed
        ]
        if self.last_mask is not None:
            self.last_mask = self.last_mask[
                :, [tmp_id - 1 for tmp_id in kept_tmp_ids]
            ]
        self.object_manager.delete_objects(objects)
        self.memory.purge_except(self.object_manager.all_obj_ids)

    def output_prob_to_mask(self, output: torch.Tensor) -> torch.Tensor:
        """Convert probabilities or temporary ids to stable CUTIE object ids."""
        if output.ndim == 3:
            temporary_mask = torch.argmax(output, dim=0)
        elif output.ndim == 2:
            temporary_mask = output
        else:
            raise ValueError(
                "CUTIE output must be a 2D index mask or 3D probability tensor; "
                f"received shape {tuple(output.shape)}."
            )
        return self.object_manager.tmp_to_obj_cls(temporary_mask)

    def get_aux_outputs(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        image, pads = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension
        _, pix_feat = self.image_feature_store.get_features(
            self.curr_ti, image)

        aux_inputs = self.memory.aux
        aux_outputs = self.network.compute_aux(
            pix_feat, aux_inputs, selector=None)
        aux_outputs['q_weights'] = aux_inputs['q_weights']
        aux_outputs['p_weights'] = aux_inputs['p_weights']

        for k, v in aux_outputs.items():
            if len(v.shape) == 5:
                aux_outputs[k] = F.interpolate(v[0],
                                               size=image.shape[-2:],
                                               mode='bilinear',
                                               align_corners=False)
            elif 'weights' in k:
                b, num_objects, num_heads, num_queries, h, w = v.shape
                v = v.view(num_objects * num_heads, num_queries, h, w)
                v = F.interpolate(
                    v, size=image.shape[-2:], mode='bilinear', align_corners=False)
                aux_outputs[k] = v.view(
                    num_objects, num_heads, num_queries, *image.shape[-2:])
            else:
                aux_outputs[k] = F.interpolate(v,
                                               size=image.shape[-2:],
                                               mode='bilinear',
                                               align_corners=False)[0]
            aux_outputs[k] = unpad(aux_outputs[k], pads)
            if 'weights' in k:
                weights = aux_outputs[k]
                weights = weights / (weights.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] +
                                     1e-8)
                aux_outputs[k] = (weights * 255).cpu().numpy()
            else:
                aux_outputs[k] = (aux_outputs[k].softmax(
                    dim=0) * 255).cpu().numpy()

        self.image_feature_store.delete(self.curr_ti)
        return aux_outputs

    def get_aux_object_weights(self, image: torch.Tensor) -> np.ndarray:
        image, pads = pad_divide_by(image, 16)
        # B*num_objects*H*W*num_queries -> num_objects*num_queries*H*W
        # weights = F.softmax(self.obj_logits, dim=-1)[0]
        weights = F.sigmoid(self.obj_logits)[0]
        weights = weights.permute(0, 3, 1, 2).contiguous()
        weights = F.interpolate(weights,
                                size=image.shape[-2:],
                                mode='bilinear',
                                align_corners=False)
        # weights = weights / (weights.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0])
        weights = unpad(weights, pads)
        weights = (weights * 255).cpu().numpy()
        return weights
