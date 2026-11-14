from typing import List, Dict, Optional
import logging
from omegaconf import DictConfig
import torch
import torch.nn as nn

from annolid.segmentation.cutie_vos.model.modules import *
from annolid.segmentation.cutie_vos.model.big_modules import *
from annolid.segmentation.cutie_vos.model.aux_modules import AuxComputer
from annolid.segmentation.cutie_vos.model.utils.memory_utils import *
from annolid.segmentation.cutie_vos.model.transformer.object_transformer import QueryTransformer
from annolid.segmentation.cutie_vos.model.transformer.object_summarizer import ObjectSummarizer
from annolid.segmentation.cutie_vos.utils.tensor_utils import aggregate

log = logging.getLogger()


class CUTIE(nn.Module):
    def __init__(self, cfg: DictConfig, *, single_object=False):
        super().__init__()
        model_cfg = cfg.model
        self.ms_dims = model_cfg.pixel_encoder.ms_dims
        self.key_dim = model_cfg.key_dim
        self.value_dim = model_cfg.value_dim
        self.sensory_dim = model_cfg.sensory_dim
        self.pixel_dim = model_cfg.pixel_dim
        self.embed_dim = model_cfg.embed_dim
        self.single_object = single_object

        log.info(f'Single object: {self.single_object}')

        self.pixel_encoder = PixelEncoder(model_cfg)
        self.pix_feat_proj = nn.Conv2d(
            self.ms_dims[0], self.pixel_dim, kernel_size=1)
        self.key_proj = KeyProjection(model_cfg)
        self.mask_encoder = MaskEncoder(model_cfg, single_object=single_object)
        self.mask_decoder = MaskDecoder(model_cfg)
        self.pixel_fuser = PixelFeatureFuser(
            model_cfg, single_object=single_object)
        self.object_transformer = QueryTransformer(model_cfg)
        self.object_summarizer = ObjectSummarizer(model_cfg)
        self.aux_computer = AuxComputer(cfg)

        self.register_buffer("pixel_mean", torch.Tensor(
            model_cfg.pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(
            model_cfg.pixel_std).view(-1, 1, 1), False)

    def _get_others(self, masks: torch.Tensor) -> torch.Tensor:
        # for each object, return the sum of masks of all other objects
        if self.single_object:
            return None

        num_objects = masks.shape[1]
        if num_objects >= 1:
            others = (masks.sum(dim=1, keepdim=True) - masks).clamp(0, 1)
        else:
            others = torch.zeros_like(masks)
        return others

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.pixel_mean) / self.pixel_std

    def encode_image(self, image: torch.Tensor) -> (Iterable[torch.Tensor], torch.Tensor):
        image = self.normalize_image(image)
        ms_image_feat = self.pixel_encoder(image)
        return ms_image_feat, self.pix_feat_proj(ms_image_feat[0])

    def encode_mask_chunk(
        self,
        normalized_image: torch.Tensor,
        pix_feat: torch.Tensor,
        sensory: torch.Tensor,
        masks: torch.Tensor,
        others: Optional[torch.Tensor] = None,
        *,
        deep_update: bool = True,
        need_weights: bool = False,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]):
        """Encode one object chunk (`<= chunk_size`) using a pre-normalized image."""
        if others is None and not self.single_object:
            others = self._get_others(masks)
        mask_value, new_sensory = self.mask_encoder(
            normalized_image,
            pix_feat,
            sensory,
            masks,
            others,
            deep_update=deep_update,
            chunk_size=-1,
        )
        object_summaries, object_logits = self.object_summarizer(
            masks, mask_value, need_weights, chunk_size=-1
        )
        return mask_value, new_sensory, object_summaries, object_logits

    def encode_mask(
            self,
            image: torch.Tensor,
            ms_features: List[torch.Tensor],
            sensory: torch.Tensor,
            masks: torch.Tensor,
            *,
            deep_update: bool = True,
            chunk_size: int = -1,
            need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        image = self.normalize_image(image)
        others = (
            None
            if (not self.single_object and chunk_size >= 1 and masks.shape[1] > chunk_size)
            else self._get_others(masks)
        )
        mask_value, new_sensory = self.mask_encoder(image,
                                                    ms_features,
                                                    sensory,
                                                    masks,
                                                    others,
                                                    deep_update=deep_update,
                                                    chunk_size=chunk_size)
        object_summaries, object_logits = self.object_summarizer(
            masks, mask_value, need_weights, chunk_size=chunk_size)
        return mask_value, new_sensory, object_summaries, object_logits

    def transform_key(self,
                      final_pix_feat: torch.Tensor,
                      *,
                      need_sk: bool = True,
                      need_ek: bool = True) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        key, shrinkage, selection = self.key_proj(
            final_pix_feat, need_s=need_sk, need_e=need_ek)
        return key, shrinkage, selection

    # Used in training only.
    # This step is replaced by MemoryManager in test time
    def read_memory(self, query_key: torch.Tensor, query_selection: torch.Tensor,
                    memory_key: torch.Tensor, memory_shrinkage: torch.Tensor,
                    msk_value: torch.Tensor, obj_memory: torch.Tensor, pix_feat: torch.Tensor,
                    sensory: torch.Tensor, last_mask: torch.Tensor,
                    selector: torch.Tensor) -> (torch.Tensor, Dict[str, torch.Tensor]):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        msk_value       : B * num_objects * CV * T * H * W
        obj_memory      : B * num_objects * T * num_summaries * C
        pixel_feature   : B * C * H * W
        """
        batch_size, num_objects = msk_value.shape[:2]

        # read using visual attention
        with torch.cuda.amp.autocast(enabled=False):
            affinity = get_affinity(memory_key.float(), memory_shrinkage.float(), query_key.float(),
                                    query_selection.float())

            msk_value = msk_value.flatten(start_dim=1, end_dim=2).float()

            # B * (num_objects*CV) * H * W
            pixel_readout = readout(affinity, msk_value)
            pixel_readout = pixel_readout.view(batch_size, num_objects, self.value_dim,
                                               *pixel_readout.shape[-2:])
        pixel_readout = self.pixel_fusion(
            pix_feat, pixel_readout, sensory, last_mask)

        # read from query transformer
        mem_readout, aux_features = self.readout_query(
            pixel_readout, obj_memory, selector=selector)

        aux_output = {
            'sensory': sensory,
            'q_logits': aux_features['logits'] if aux_features else None,
            'attn_mask': aux_features['attn_mask'] if aux_features else None,
        }

        return mem_readout, aux_output

    def pixel_fusion(self,
                     pix_feat: torch.Tensor,
                     pixel: torch.Tensor,
                     sensory: torch.Tensor,
                     last_mask: torch.Tensor,
                     *,
                     chunk_size: int = -1,
                     last_others: Optional[torch.Tensor] = None) -> torch.Tensor:
        last_mask = F.interpolate(
            last_mask, size=sensory.shape[-2:], mode='area')
        if last_others is None:
            last_others = self._get_others(last_mask)
        else:
            last_others = F.interpolate(
                last_others, size=sensory.shape[-2:], mode='area')
        fused = self.pixel_fuser(pix_feat,
                                 pixel,
                                 sensory,
                                 last_mask,
                                 last_others,
                                 chunk_size=chunk_size)
        return fused

    def readout_query(self,
                      pixel_readout,
                      obj_memory,
                      *,
                      selector=None,
                      need_weights=False,
                      chunk_size: int = -1) -> (torch.Tensor, Dict[str, torch.Tensor]):
        return self.object_transformer(pixel_readout,
                                       obj_memory,
                                       selector=selector,
                                       need_weights=need_weights,
                                       chunk_size=chunk_size)

    @staticmethod
    def _upsample_and_softmax_logits(
        logits_s4: torch.Tensor,
        *,
        chunk_size: int = -1,
        return_logits: bool = True,
    ) -> (Optional[torch.Tensor], torch.Tensor):
        num_channels = logits_s4.shape[1]
        if return_logits or chunk_size < 1 or num_channels <= chunk_size + 1:
            logits = F.interpolate(
                logits_s4, scale_factor=4, mode='bilinear', align_corners=False
            )
            prob = F.softmax(logits, dim=1)
            return logits, prob

        bs, _, h4, w4 = logits_s4.shape
        h1, w1 = h4 * 4, w4 * 4
        step = max(1, int(chunk_size))
        out = logits_s4.new_empty((bs, num_channels, h1, w1))
        max_val: Optional[torch.Tensor] = None
        for c0 in range(0, num_channels, step):
            c1 = min(num_channels, c0 + step)
            chunk = F.interpolate(
                logits_s4[:, c0:c1],
                scale_factor=4,
                mode='bilinear',
                align_corners=False,
            )
            out[:, c0:c1] = chunk
            chunk_max = chunk.max(dim=1, keepdim=True).values
            max_val = chunk_max if max_val is None else torch.maximum(max_val, chunk_max)

        sum_exp = logits_s4.new_zeros((bs, 1, h1, w1))
        for c0 in range(0, num_channels, step):
            c1 = min(num_channels, c0 + step)
            view = out[:, c0:c1]
            view.sub_(max_val).exp_()
            sum_exp.add_(view.sum(dim=1, keepdim=True))

        for c0 in range(0, num_channels, step):
            c1 = min(num_channels, c0 + step)
            out[:, c0:c1].div_(sum_exp)

        return None, out

    def segment(self,
                ms_image_feat: List[torch.Tensor],
                memory_readout: torch.Tensor,
                sensory: torch.Tensor,
                *,
                selector: bool = None,
                chunk_size: int = -1,
                update_sensory: bool = True,
                return_logits: bool = True) -> (torch.Tensor, Optional[torch.Tensor], torch.Tensor):
        """
        multi_scale_features is from the key encoder for skip-connection
        memory_readout is from working/long-term memory
        sensory is the sensory memory
        last_mask is the mask from the last frame, supplementing sensory memory
        selector is 1 if an object exists, and 0 otherwise. We use it to filter padded objects
            during training.
        """
        sensory, logits = self.mask_decoder(ms_image_feat,
                                            memory_readout,
                                            sensory,
                                            chunk_size=chunk_size,
                                            update_sensory=update_sensory)

        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector

        # Softmax over all objects[]
        logits = aggregate(prob, dim=1)
        logits, prob = self._upsample_and_softmax_logits(
            logits,
            chunk_size=chunk_size,
            return_logits=return_logits,
        )

        return sensory, logits, prob

    def _decode_compact_pass1(
        self,
        ms_image_feat: List[torch.Tensor],
        memory_readout: torch.Tensor,
        sensory: torch.Tensor,
        *,
        selector: bool = None,
        chunk_size: int = 16,
        update_sensory: bool = True,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """Run MaskDecoder + `aggregate` + streaming full-res softmax `(m, d, best_idx)`."""
        sensory, logits = self.mask_decoder(
            ms_image_feat,
            memory_readout,
            sensory,
            chunk_size=chunk_size,
            update_sensory=update_sensory,
        )

        prob = torch.sigmoid(logits)
        if selector is not None:
            prob = prob * selector

        logits_s4 = aggregate(prob, dim=1)
        num_channels = logits_s4.shape[1]
        step = max(
            1,
            int(chunk_size) if chunk_size and int(chunk_size) > 0 else num_channels,
        )

        m: Optional[torch.Tensor] = None
        d: Optional[torch.Tensor] = None
        best_idx: Optional[torch.Tensor] = None
        for c0 in range(0, num_channels, step):
            c1 = min(num_channels, c0 + step)
            chunk = F.interpolate(
                logits_s4[:, c0:c1],
                scale_factor=4,
                mode='bilinear',
                align_corners=False,
            )
            chunk_max, chunk_arg = chunk.max(dim=1, keepdim=True)
            if m is None:
                m = chunk_max
                best_idx = chunk_arg.squeeze(1).to(dtype=torch.int32)
                d = torch.exp(chunk - m).sum(dim=1, keepdim=True)
            else:
                better = (chunk_max > m).squeeze(1)
                best_idx = torch.where(
                    better,
                    chunk_arg.squeeze(1).to(dtype=torch.int32) + c0,
                    best_idx,
                )
                m_new = torch.maximum(m, chunk_max)
                d = d * torch.exp(m - m_new) + torch.exp(chunk - m_new).sum(
                    dim=1, keepdim=True
                )
                m = m_new

        return sensory, logits_s4, m, d, best_idx

    def segment_compact(
        self,
        ms_image_feat: List[torch.Tensor],
        memory_readout: torch.Tensor,
        sensory: torch.Tensor,
        *,
        selector: bool = None,
        chunk_size: int = 16,
        update_sensory: bool = True,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Stream full-res argmax + stride-16 `last_mask` in `O(chunk_size * H * W)` memory."""
        sensory, logits_s4, m, d, best_idx = self._decode_compact_pass1(
            ms_image_feat,
            memory_readout,
            sensory,
            selector=selector,
            chunk_size=chunk_size,
            update_sensory=update_sensory,
        )
        bs, num_channels, h4, w4 = logits_s4.shape
        num_objects = num_channels - 1
        h1, w1 = h4 * 4, w4 * 4
        h16, w16 = max(1, h1 // 16), max(1, w1 // 16)
        step = max(
            1,
            int(chunk_size) if chunk_size and int(chunk_size) > 0 else num_channels,
        )

        last_mask_s16 = logits_s4.new_empty((bs, num_objects, h16, w16))
        for fg0 in range(0, num_objects, step):
            fg1 = min(num_objects, fg0 + step)
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

        return sensory, best_idx, last_mask_s16

    def compute_aux(self, pix_feat: torch.Tensor, aux_inputs: Dict[str, torch.Tensor],
                    selector: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.aux_computer(pix_feat, aux_inputs, selector)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, src_dict, init_as_zero_if_needed=False) -> None:
        if not self.single_object:
            # Map single-object weight to multi-object weight (4->5 out channels in conv1)
            for k in list(src_dict.keys()):
                if k == 'mask_encoder.conv1.weight':
                    if src_dict[k].shape[1] == 4:
                        log.info(
                            f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros(
                            (64, 1, 7, 7), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
                elif k == 'pixel_fuser.sensory_compress.weight':
                    if src_dict[k].shape[1] == self.sensory_dim + 1:
                        log.info(
                            f'Converting {k} from single object to multiple objects.')
                        pads = torch.zeros(
                            (self.value_dim, 1, 1, 1), device=src_dict[k].device)
                        if not init_as_zero_if_needed:
                            nn.init.orthogonal_(pads)
                            log.info(f'Randomly initialized padding for {k}.')
                        else:
                            log.info(f'Zero-initialized padding for {k}.')
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)
        elif self.single_object:
            """
            If the model is multiple-object and we are training in single-object,
            we strip the last channel of conv1.
            This is not supposed to happen in standard training except when users are trying to
            finetune a trained model with single object datasets.
            """
            if src_dict['mask_encoder.conv1.weight'].shape[1] == 5:
                log.warning(f'Converting {k} from multiple objects to single object.'
                            'This is not supposed to happen in standard training.')
                src_dict[k] = src_dict[k][:, :-1]

        for k in src_dict:
            if k not in self.state_dict():
                log.info(
                    f'Key {k} found in src_dict but not in self.state_dict()!!!')
        for k in self.state_dict():
            if k not in src_dict:
                log.info(
                    f'Key {k} found in self.state_dict() but not in src_dict!!!')

        self.load_state_dict(src_dict, strict=False)

    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device
