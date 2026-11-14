from typing import Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
from annolid.segmentation.cutie_vos.model.group_modules import GConv2d
from annolid.segmentation.cutie_vos.utils.tensor_utils import aggregate
from annolid.segmentation.cutie_vos.model.transformer.positional_encoding import PositionalEncoding
from annolid.segmentation.cutie_vos.model.transformer.transformer_layers import *


class QueryTransformerBlock(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries
        self.ff_dim = this_cfg.ff_dim

        self.read_from_pixel = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_pixel.add_pe_to_qkv)
        self.self_attn = SelfAttention(self.embed_dim,
                                       self.num_heads,
                                       add_pe_to_qkv=this_cfg.query_self_attention.add_pe_to_qkv)
        self.ffn = FFN(self.embed_dim, self.ff_dim)
        self.read_from_query = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_query.add_pe_to_qkv,
                                              norm=this_cfg.read_from_query.output_norm)
        self.pixel_ffn = PixelFFN(self.embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            pixel: torch.Tensor,
            query_pe: torch.Tensor,
            pixel_pe: torch.Tensor,
            attn_mask: torch.Tensor,
            need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # x: (bs*num_objects)*num_queries*embed_dim
        # pixel: bs*num_objects*C*H*W
        # query_pe: (bs*num_objects)*num_queries*embed_dim
        # pixel_pe: (bs*num_objects)*(H*W)*C
        # attn_mask: (bs*num_objects*num_heads)*num_queries*(H*W)

        # bs*num_objects*C*H*W -> (bs*num_objects)*(H*W)*C
        pixel_flat = pixel.flatten(3, 4).flatten(
            0, 1).transpose(1, 2).contiguous()
        x, q_weights = self.read_from_pixel(x,
                                            pixel_flat,
                                            query_pe,
                                            pixel_pe,
                                            attn_mask=attn_mask,
                                            need_weights=need_weights)
        x = self.self_attn(x, query_pe)
        x = self.ffn(x)

        pixel_flat, p_weights = self.read_from_query(pixel_flat,
                                                     x,
                                                     pixel_pe,
                                                     query_pe,
                                                     need_weights=need_weights)
        pixel = self.pixel_ffn(pixel, pixel_flat)

        if need_weights:
            bs, num_objects, _, h, w = pixel.shape
            q_weights = q_weights.view(
                bs, num_objects, self.num_heads, self.num_queries, h, w)
            p_weights = p_weights.transpose(2, 3).view(bs, num_objects, self.num_heads,
                                                       self.num_queries, h, w)

        return x, pixel, q_weights, p_weights


class QueryTransformer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries

        # query initialization and embedding
        self.query_init = nn.Embedding(self.num_queries, self.embed_dim)
        self.query_emb = nn.Embedding(self.num_queries, self.embed_dim)

        # projection from object summaries to query initialization and embedding
        self.summary_to_query_init = nn.Linear(self.embed_dim, self.embed_dim)
        self.summary_to_query_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature
        self.pixel_init_proj = GConv2d(
            self.embed_dim, self.embed_dim, kernel_size=1)
        self.pixel_emb_proj = GConv2d(
            self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_pe = PositionalEncoding(self.embed_dim,
                                             scale=self.pixel_pe_scale,
                                             temperature=self.pixel_pe_temperature,
                                             channel_last=False,
                                             transpose_output=True)

        # transformer blocks
        self.num_blocks = this_cfg.num_blocks
        self.blocks = nn.ModuleList(
            QueryTransformerBlock(model_cfg) for _ in range(self.num_blocks))
        self.mask_pred = nn.ModuleList(
            nn.Sequential(nn.ReLU(), GConv2d(self.embed_dim, 1, kernel_size=1))
            for _ in range(self.num_blocks + 1))

        self.act = nn.ReLU(inplace=True)

    def forward(self,
                pixel: torch.Tensor,
                obj_summaries: torch.Tensor,
                selector: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                *,
                chunk_size: int = -1) -> (torch.Tensor, Dict[str, torch.Tensor]):
        # pixel: B*num_objects*embed_dim*H*W
        # obj_summaries: B*num_objects*T*num_queries*embed_dim
        T = obj_summaries.shape[2]
        bs, num_objects, _, H, W = pixel.shape

        if 0 < chunk_size < num_objects:
            return self._forward_chunked(
                pixel,
                obj_summaries,
                selector=selector,
                need_weights=need_weights,
                chunk_size=chunk_size,
            )

        # normalize object values
        # the last channel is the cumulative area of the object
        obj_summaries = obj_summaries.view(bs * num_objects, T, self.num_queries,
                                           self.embed_dim + 1)
        # sum over time
        # during inference, T=1 as we already did streaming average in memory_manager
        obj_sums = obj_summaries[:, :, :, :-1].sum(dim=1)
        obj_area = obj_summaries[:, :, :, -1:].sum(dim=1)
        obj_values = obj_sums / (obj_area + 1e-4)
        obj_init = self.summary_to_query_init(obj_values)
        obj_emb = self.summary_to_query_emb(obj_values)

        # positional embeddings for object queries
        query = self.query_init.weight.unsqueeze(
            0).expand(bs * num_objects, -1, -1) + obj_init
        query_emb = self.query_emb.weight.unsqueeze(
            0).expand(bs * num_objects, -1, -1) + obj_emb

        # positional embeddings for pixel features (5D call avoids caching bs*N copies)
        pixel_init = self.pixel_init_proj(pixel)
        pixel_emb = self.pixel_emb_proj(pixel)
        base_spatial_pe = self.spatial_pe(pixel).flatten(2, 3)
        pixel_emb = pixel_emb.flatten(3, 4).transpose(2, 3)
        pixel_pe = (base_spatial_pe + pixel_emb).flatten(0, 1).contiguous()

        pixel = pixel_init

        # run the transformer
        aux_features = {'logits': []}

        # first aux output
        aux_logits = self.mask_pred[0](pixel).squeeze(2)
        attn_mask = self._get_aux_mask(aux_logits, selector)
        aux_features['logits'].append(aux_logits)
        for i in range(self.num_blocks):
            query, pixel, q_weights, p_weights = self.blocks[i](query,
                                                                pixel,
                                                                query_emb,
                                                                pixel_pe,
                                                                attn_mask,
                                                                need_weights=need_weights)

            if self.training or i <= self.num_blocks - 1 or need_weights:
                aux_logits = self.mask_pred[i + 1](pixel).squeeze(2)
                attn_mask = self._get_aux_mask(aux_logits, selector)
                aux_features['logits'].append(aux_logits)

        aux_features['q_weights'] = q_weights  # last layer only
        aux_features['p_weights'] = p_weights  # last layer only

        if self.training:
            # no need to save all heads
            aux_features['attn_mask'] = attn_mask.view(bs, num_objects, self.num_heads,
                                                       self.num_queries, H, W)[:, :, 0]

        return pixel, aux_features

    def _forward_chunked(
        self,
        pixel: torch.Tensor,
        obj_summaries: torch.Tensor,
        *,
        selector: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        chunk_size: int,
    ) -> (torch.Tensor, Dict[str, torch.Tensor]):
        """Run `QueryTransformer` in `O(chunk_size)` working memory with exact cross-object parity."""
        T = obj_summaries.shape[2]
        bs, num_objects, _, H, W = pixel.shape
        ranges = [
            (start, min(start + chunk_size, num_objects))
            for start in range(0, num_objects, chunk_size)
        ]

        obj_summaries_5d = obj_summaries.view(
            bs, num_objects, T, self.num_queries, self.embed_dim + 1
        )
        obj_sums = obj_summaries_5d[:, :, :, :, :-1].sum(dim=2)
        obj_area = obj_summaries_5d[:, :, :, :, -1:].sum(dim=2)
        obj_values = obj_sums / (obj_area + 1e-4)
        query = (
            self.query_init.weight.view(1, 1, self.num_queries, self.embed_dim)
            + self.summary_to_query_init(obj_values)
        )
        query_emb = (
            self.query_emb.weight.view(1, 1, self.num_queries, self.embed_dim)
            + self.summary_to_query_emb(obj_values)
        )

        base_spatial_pe_flat = self.spatial_pe(pixel).flatten(2, 3)
        pixel_chunks = [
            self.pixel_init_proj(pixel[:, start:end]) for start, end in ranges
        ]

        aux_logits = torch.cat(
            [self.mask_pred[0](pc).squeeze(2) for pc in pixel_chunks],
            dim=1,
        )
        fg_mask = self._compute_foreground_mask(aux_logits, selector)
        aux_features: Dict[str, Optional[torch.Tensor]] = {'logits': [aux_logits]}

        q_weights = None
        p_weights = None
        for i in range(self.num_blocks):
            next_query_chunks = []
            next_pixel_chunks = []
            next_logit_chunks = []
            q_weight_chunks = [] if (need_weights and i == self.num_blocks - 1) else None
            p_weight_chunks = [] if (need_weights and i == self.num_blocks - 1) else None

            for chunk_idx, (start, end) in enumerate(ranges):
                c_n = end - start
                q_c = query[:, start:end].reshape(
                    bs * c_n, self.num_queries, self.embed_dim
                )
                q_emb_c = query_emb[:, start:end].reshape(
                    bs * c_n, self.num_queries, self.embed_dim
                )
                p_emb_c = (
                    self.pixel_emb_proj(pixel[:, start:end])
                    .flatten(3, 4)
                    .transpose(2, 3)
                )
                p_pe_c = (base_spatial_pe_flat + p_emb_c).flatten(0, 1).contiguous()
                attn_mask_c = self._expand_aux_mask(fg_mask[:, start:end])

                q_c, p_c, qw_c, pw_c = self.blocks[i](
                    q_c,
                    pixel_chunks[chunk_idx],
                    q_emb_c,
                    p_pe_c,
                    attn_mask_c,
                    need_weights=need_weights,
                )
                next_query_chunks.append(
                    q_c.view(bs, c_n, self.num_queries, self.embed_dim)
                )
                next_pixel_chunks.append(p_c)
                if q_weight_chunks is not None:
                    q_weight_chunks.append(qw_c)
                    p_weight_chunks.append(pw_c)
                if self.training or i <= self.num_blocks - 1 or need_weights:
                    next_logit_chunks.append(self.mask_pred[i + 1](p_c).squeeze(2))

            query = torch.cat(next_query_chunks, dim=1)
            pixel_chunks = next_pixel_chunks
            if q_weight_chunks is not None:
                q_weights = torch.cat(q_weight_chunks, dim=1)
                p_weights = torch.cat(p_weight_chunks, dim=1)
            if next_logit_chunks:
                aux_logits = torch.cat(next_logit_chunks, dim=1)
                fg_mask = self._compute_foreground_mask(aux_logits, selector)
                aux_features['logits'].append(aux_logits)

        aux_features['q_weights'] = q_weights
        aux_features['p_weights'] = p_weights

        if self.training:
            attn_head0_chunks = [
                self._expand_aux_mask(fg_mask[:, start:end]).view(
                    bs, end - start, self.num_heads, self.num_queries, H, W
                )[:, :, 0]
                for start, end in ranges
            ]
            aux_features['attn_mask'] = torch.cat(attn_head0_chunks, dim=1)

        return torch.cat(pixel_chunks, dim=1), aux_features

    def _compute_foreground_mask(
        self,
        logits: torch.Tensor,
        selector: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        agg_logits = aggregate(prob, dim=1)
        is_foreground = agg_logits[:, 1:] >= agg_logits.max(dim=1, keepdim=True)[0]
        return is_foreground.bool().flatten(start_dim=2)

    def _expand_aux_mask(self, foreground_mask: torch.Tensor) -> torch.Tensor:
        inv_foreground_mask = ~foreground_mask
        inv_background_mask = foreground_mask

        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)

        aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False

        return aux_mask

    def _get_aux_mask(self, logits: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # logits: batch_size*num_objects*H*W
        # selector: batch_size*num_objects*1*1
        # returns a mask of shape (batch_size*num_objects*num_heads)*num_queries*(H*W)
        # where True means the attention is blocked
        return self._expand_aux_mask(self._compute_foreground_mask(logits, selector))
