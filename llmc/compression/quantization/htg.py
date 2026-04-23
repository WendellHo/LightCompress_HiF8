import gc
from typing import Dict, List

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_


@ALGO_REGISTRY
class HTG(BaseBlockwiseQuantization):
    """Hierarchical Timestep Grouping (Phase-1).

    Phase-1 goal is to make HTG a formal, runnable method under
    `quant.video_gen.method` for Wan2.1-T2V W8A8 pipelines.
    This implementation focuses on:
    1) grouped timestep-aware channel shift statistics,
    2) EMA-based channel scaling,
    3) static re-parameterization compatible with current runtime.
    """

    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        special_config = self.quant_config.get('special', {})
        self.alpha = float(special_config.get('alpha', 0.99))
        self.group_num = special_config.get('group_num', None)
        self.group_ratio = int(special_config.get('group_ratio', 10))
        self.min_group_num = int(special_config.get('min_group_num', 1))
        self.save_shift = bool(special_config.get('save_shift', True))
        self.enable_dynamic_runtime = bool(special_config.get('enable_dynamic_runtime', True))

        # Kept for optional export in later phases.
        self.htg_meta = {}
        if self.save_shift and not hasattr(self, 'act_shifts'):
            self.act_shifts = {}

    @torch.no_grad()
    def filter_subset(self, prev_op):
        return isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_))

    @torch.no_grad()
    def _get_weight_scale(self, layers):
        weights = self.collect_layers_weights(layers)
        scale = torch.cat(
            [fc.abs().max(dim=0, keepdim=True)[0] for fc in weights],
            dim=0,
        )
        scale = scale.max(dim=0)[0].clamp(min=1e-5)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        return scale

    @torch.no_grad()
    def _channel_shift(self, x):
        x = x.cuda()
        x = x.view(-1, x.shape[-1])
        if not self.stream_stats or self.stream_chunk_size <= 0 or x.shape[0] <= self.stream_chunk_size:
            xmax = x.max(dim=0)[0]
            xmin = x.min(dim=0)[0]
            return (xmax + xmin) * 0.5

        xmax, xmin = None, None
        for start in range(0, x.shape[0], self.stream_chunk_size):
            end = min(start + self.stream_chunk_size, x.shape[0])
            cur_x = x[start:end]
            cur_max = cur_x.max(dim=0)[0]
            cur_min = cur_x.min(dim=0)[0]
            xmax = cur_max if xmax is None else torch.maximum(xmax, cur_max)
            xmin = cur_min if xmin is None else torch.minimum(xmin, cur_min)
        return (xmax + xmin) * 0.5

    @torch.no_grad()
    def _channel_abs_max(self, x):
        x = x.cuda()
        x = x.view(-1, x.shape[-1])
        if not self.stream_stats or self.stream_chunk_size <= 0 or x.shape[0] <= self.stream_chunk_size:
            return x.abs().max(dim=0)[0]

        max_values = None
        for start in range(0, x.shape[0], self.stream_chunk_size):
            end = min(start + self.stream_chunk_size, x.shape[0])
            current = x[start:end].abs().max(dim=0)[0]
            max_values = current if max_values is None else torch.maximum(max_values, current)
        return max_values

    def _resolve_num_steps(self, tensors):
        configured_steps = getattr(self.model, 'sample_steps', None)
        if configured_steps is not None and configured_steps > 0:
            return min(int(configured_steps), len(tensors))
        return len(tensors)

    def _resolve_num_steps_stream(self):
        configured_steps = getattr(self.model, 'sample_steps', None)
        if configured_steps is None or configured_steps <= 0:
            raise ValueError(
                'HTG stream_stats requires model.sample_steps to resolve timestep buckets.'
            )
        return int(configured_steps)

    def _split_by_timestep(self, tensors, num_steps):
        step_buckets = [[] for _ in range(num_steps)]
        for idx, tensor in enumerate(tensors):
            step_buckets[idx % num_steps].append(tensor)
        return step_buckets

    @torch.no_grad()
    def _compute_step_shifts(self, step_buckets):
        z_t = []
        for bucket in step_buckets:
            bucket_shift = None
            for x in bucket:
                cur_shift = self._channel_shift(x).float()
                bucket_shift = cur_shift if bucket_shift is None else (bucket_shift + cur_shift)
            bucket_shift = bucket_shift / max(len(bucket), 1)
            z_t.append(bucket_shift)
        return z_t

    @torch.no_grad()
    def _compute_step_shifts_stream(self, stream_stats):
        shift_sums = stream_stats.get('step_shift_sum', [])
        shift_counts = stream_stats.get('step_shift_count', [])
        if len(shift_sums) == 0:
            raise ValueError('HTG stream_stats enabled but step_shift_sum is empty.')

        z_t = []
        for shift_sum, count in zip(shift_sums, shift_counts):
            if shift_sum is None or count <= 0:
                raise ValueError('HTG stream_stats encountered empty timestep statistics.')
            z_t.append((shift_sum / float(count)).float())
        return z_t

    def _resolve_group_num(self, num_steps):
        if self.group_num is not None:
            return max(self.min_group_num, min(int(self.group_num), num_steps))
        return max(self.min_group_num, min(num_steps, max(1, num_steps // self.group_ratio)))

    def _build_group_boundaries_norm(self, groups: List[List[int]], num_steps: int):
        if len(groups) <= 1:
            return torch.empty(0, dtype=torch.float32)
        denom = max(num_steps - 1, 1)
        boundaries = [max(group) / float(denom) for group in groups[:-1]]
        return torch.tensor(boundaries, dtype=torch.float32)

    def _adjacent_cluster(self, vectors: List[torch.Tensor], group_num: int):
        groups = [[i] for i in range(len(vectors))]
        centroids = [v.clone() for v in vectors]

        while len(groups) > group_num:
            best_idx = None
            best_dist = None
            for i in range(len(groups) - 1):
                dist = torch.mean((centroids[i] - centroids[i + 1]) ** 2).item()
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = i

            left = best_idx
            right = best_idx + 1
            merged_group = groups[left] + groups[right]
            merged_centroid = (
                centroids[left] * len(groups[left]) + centroids[right] * len(groups[right])
            ) / len(merged_group)

            groups[left] = merged_group
            centroids[left] = merged_centroid
            del groups[right]
            del centroids[right]

        step_to_group = {}
        for g, group in enumerate(groups):
            for step_idx in group:
                step_to_group[step_idx] = g

        boundaries = [max(group) for group in groups[:-1]]
        return groups, step_to_group, boundaries

    @torch.no_grad()
    def _compute_group_shifts(self, z_t: List[torch.Tensor], groups: List[List[int]]):
        z_g = []
        for group in groups:
            group_shift = None
            for step_idx in group:
                group_shift = z_t[step_idx] if group_shift is None else (group_shift + z_t[step_idx])
            group_shift = group_shift / max(len(group), 1)
            z_g.append(group_shift)
        return z_g

    @torch.no_grad()
    def _compute_ema_scale(
        self,
        step_buckets: List[List[torch.Tensor]],
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
    ):
        x_abs_max_t = []
        for t, bucket in enumerate(step_buckets):
            shift = z_g[step_to_group[t]]
            cur_max = None
            for x in bucket:
                x = x.cuda()
                shape = [1] * (x.dim() - 1) + [-1]
                shifted = x - shift.view(*shape)
                step_max = self._channel_abs_max(shifted).float()
                cur_max = step_max if cur_max is None else (cur_max + step_max)
            cur_max = cur_max / max(len(bucket), 1)
            x_abs_max_t.append(cur_max.clamp(min=1e-5))

        ema = None
        for t in range(len(x_abs_max_t) - 1, -1, -1):
            cur = x_abs_max_t[t]
            ema = cur if ema is None else (self.alpha * ema + (1.0 - self.alpha) * cur)

        # HTG Eq.(8): one global channel-wise scaling vector from EMA statistics.
        ema = ema.clamp(min=1e-5)
        scale = ema / ema.max().clamp(min=1e-5)
        scale = scale.clamp(min=1e-5)
        return scale

    @torch.no_grad()
    def _compute_ema_scale_stream(
        self,
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
        input_name: str,
        num_steps: int,
    ):
        if self._current_block is None:
            raise ValueError('HTG stream_stats requires current block context.')

        x_abs_sum = [None for _ in range(num_steps)]
        x_abs_count = [0 for _ in range(num_steps)]
        stream_state = {'idx': 0}

        def collect_step_absmax(_, x, __):
            if len(x) == 0 or not torch.is_tensor(x[0]):
                return
            inp = x[0].detach()
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            step_idx = stream_state['idx'] % num_steps
            group_idx = step_to_group[step_idx]
            shift = z_g[group_idx].to(device=inp.device, dtype=inp.dtype)
            shape = [1] * (inp.dim() - 1) + [-1]
            shifted = inp - shift.view(*shape)
            step_max = shifted.abs().view(-1, shifted.shape[-1]).amax(dim=0).to(torch.float32).cpu()
            if x_abs_sum[step_idx] is None:
                x_abs_sum[step_idx] = step_max
            else:
                x_abs_sum[step_idx] = x_abs_sum[step_idx] + step_max
            x_abs_count[step_idx] += 1
            stream_state['idx'] += 1

        layer = dict(self._current_block.named_modules())[input_name]
        handle = layer.register_forward_hook(collect_step_absmax)
        try:
            self.block_forward(self._current_block)
        finally:
            handle.remove()

        x_abs_max_t = []
        for step_idx in range(num_steps):
            if x_abs_sum[step_idx] is None or x_abs_count[step_idx] == 0:
                raise ValueError('HTG stream_stats second pass encountered empty timestep data.')
            cur_max = x_abs_sum[step_idx] / float(x_abs_count[step_idx])
            x_abs_max_t.append(cur_max.clamp(min=1e-5))

        ema = None
        for t in range(len(x_abs_max_t) - 1, -1, -1):
            cur = x_abs_max_t[t]
            ema = cur if ema is None else (self.alpha * ema + (1.0 - self.alpha) * cur)

        ema = ema.clamp(min=1e-5)
        scale = ema / ema.max().clamp(min=1e-5)
        scale = scale.clamp(min=1e-5)
        return scale

    def _collapse_group_shifts(self, z_g: List[torch.Tensor], groups: List[List[int]]):
        # Phase-1 keeps runtime static, so collapse grouped shifts into one vector.
        total = 0
        merged = None
        for shift, group in zip(z_g, groups):
            weight = len(group)
            merged = shift * weight if merged is None else (merged + shift * weight)
            total += weight
        return merged / max(total, 1)

    @torch.no_grad()
    def _register_htg_runtime_buffers(
        self,
        prev_norm: nn.Module,
        layers: List[nn.Module],
        z_g: List[torch.Tensor],
        scale: torch.Tensor,
        boundaries_norm: torch.Tensor,
    ):
        # For Wan2.1 affine norms we expect both weight and bias.
        if not hasattr(prev_norm, 'weight') or not hasattr(prev_norm, 'bias'):
            return
        if prev_norm.bias is None:
            return

        # Use post-scale norm weight as runtime base; only bias is group-dependent.
        norm_weight = prev_norm.weight.detach().clone()
        base_bias = prev_norm.bias.detach().clone()
        scale = scale.to(device=base_bias.device, dtype=base_bias.dtype)

        group_weights = []
        group_biases = []
        for shift in z_g:
            shift = shift.to(device=base_bias.device, dtype=base_bias.dtype)
            group_weights.append(norm_weight)
            group_biases.append(base_bias - shift / scale)

        htg_norm_weight = torch.stack(group_weights, dim=0).detach()
        htg_norm_bias = torch.stack(group_biases, dim=0).detach()
        boundaries_norm = boundaries_norm.to(device=base_bias.device, dtype=base_bias.dtype)

        if hasattr(prev_norm, 'htg_norm_weight'):
            prev_norm.htg_norm_weight.data = htg_norm_weight
        else:
            prev_norm.register_buffer('htg_norm_weight', htg_norm_weight)
        if hasattr(prev_norm, 'htg_norm_bias'):
            prev_norm.htg_norm_bias.data = htg_norm_bias
        else:
            prev_norm.register_buffer('htg_norm_bias', htg_norm_bias)
        if hasattr(prev_norm, 'htg_group_boundaries'):
            prev_norm.htg_group_boundaries.data = boundaries_norm
        else:
            prev_norm.register_buffer('htg_group_boundaries', boundaries_norm)

        # Store group-specific linear bias compensation:
        # b_hat_g = b + z_g * W  == b + (z_g / s) * W_hat
        for layer in layers:
            if not hasattr(layer, 'weight'):
                continue

            layer_weight = layer.weight.detach()
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer_bias = layer.bias.detach()
            else:
                layer_bias = torch.zeros(layer_weight.shape[0], device=layer_weight.device, dtype=layer_weight.dtype)

            shift_scaled_list = []
            for shift in z_g:
                shift_scaled = (shift / scale).to(device=layer_bias.device, dtype=layer_bias.dtype)
                comp = torch.matmul(layer_weight, shift_scaled)
                shift_scaled_list.append(layer_bias + comp)
            htg_group_bias = torch.stack(shift_scaled_list, dim=0).detach()
            if hasattr(layer, 'htg_group_bias'):
                layer.htg_group_bias.data = htg_group_bias
            else:
                layer.register_buffer('htg_group_bias', htg_group_bias)

    @torch.no_grad()
    def cache_input_hook(self, m, x, y, name, feat_dict):
        if not self.stream_stats:
            return super().cache_input_hook(m, x, y, name, feat_dict)

        inputs = [i.detach() for i in x]
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            return super().cache_input_hook(m, x, y, name, feat_dict)

        inp = inputs[0]
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        flat = inp.view(-1, inp.shape[-1])
        if self.stream_chunk_size > 0 and flat.shape[0] > self.stream_chunk_size:
            xmax, xmin = None, None
            for start in range(0, flat.shape[0], self.stream_chunk_size):
                end = min(start + self.stream_chunk_size, flat.shape[0])
                cur_x = flat[start:end]
                cur_max = cur_x.max(dim=0)[0]
                cur_min = cur_x.min(dim=0)[0]
                xmax = cur_max if xmax is None else torch.maximum(xmax, cur_max)
                xmin = cur_min if xmin is None else torch.minimum(xmin, cur_min)
        else:
            xmax = flat.max(dim=0)[0]
            xmin = flat.min(dim=0)[0]
        shift = ((xmax + xmin) * 0.5).to(torch.float32).cpu()

        if not isinstance(feat_dict[name], dict):
            feat_dict[name] = {}

        num_steps = self._resolve_num_steps_stream()
        if 'step_shift_sum' not in feat_dict[name]:
            feat_dict[name]['step_shift_sum'] = [None for _ in range(num_steps)]
            feat_dict[name]['step_shift_count'] = [0 for _ in range(num_steps)]
            feat_dict[name]['step_cursor'] = 0

        step_idx = feat_dict[name]['step_cursor'] % num_steps
        if feat_dict[name]['step_shift_sum'][step_idx] is None:
            feat_dict[name]['step_shift_sum'][step_idx] = shift
        else:
            feat_dict[name]['step_shift_sum'][step_idx] = (
                feat_dict[name]['step_shift_sum'][step_idx] + shift
            )
        feat_dict[name]['step_shift_count'][step_idx] += 1
        feat_dict[name]['step_cursor'] += 1

    @torch.no_grad()
    def subset_transform(self, subset, input_feat, subset_kwargs):
        del subset_kwargs
        layers_dict = subset['layers']
        prev_op = subset['prev_op']
        input_name = subset['input'][0]

        if not self.filter_subset(prev_op):
            logger.info('HTG skip subset: prev_op is not layernorm-like.')
            return

        tensors = input_feat[input_name]
        if not self.stream_stats and len(tensors) == 0:
            logger.info('HTG skip subset: empty input features.')
            return

        layers = list(layers_dict.values())
        if self.stream_stats and isinstance(tensors, dict):
            num_steps = self._resolve_num_steps_stream()
            z_t = self._compute_step_shifts_stream(tensors)
            group_num = self._resolve_group_num(num_steps)
        else:
            num_steps = self._resolve_num_steps(tensors)
            group_num = self._resolve_group_num(num_steps)
            step_buckets = self._split_by_timestep(tensors, num_steps)
            z_t = self._compute_step_shifts(step_buckets)
        groups, step_to_group, boundaries = self._adjacent_cluster(z_t, group_num)
        boundaries_norm = self._build_group_boundaries_norm(groups, num_steps)
        z_g = self._compute_group_shifts(z_t, groups)

        if self.stream_stats and isinstance(tensors, dict):
            scale = self._compute_ema_scale_stream(z_g, step_to_group, input_name, num_steps)
        else:
            scale = self._compute_ema_scale(step_buckets, z_g, step_to_group)
        scale = scale.to(dtype=prev_op[0].weight.dtype, device=prev_op[0].weight.device)

        if self.enable_dynamic_runtime:
            # For runtime switching, keep base bias untouched and only absorb scaling.
            self.apply_scale(scale, prev_op, layers)
            self._register_htg_runtime_buffers(prev_op[0], layers, z_g, scale, boundaries_norm)
            shift = self._collapse_group_shifts(z_g, groups).to(dtype=scale.dtype, device=scale.device)
        else:
            # Static fallback follows Eq.(9) equivalent re-parameterization: shift first, then scale.
            shift = self._collapse_group_shifts(z_g, groups).to(dtype=scale.dtype, device=scale.device)
            self.apply_shift(shift, prev_op, layers)
            self.apply_scale(scale, prev_op, layers)

        if self.act_static:
            self.update_input_feat(scale, input_feat, layers_dict, False)

        # Save metadata for later runtime-phase dynamic grouping implementation.
        subset_name = list(layers_dict.keys())[0]
        key = f'{self.model.block_name_prefix}.{self.block_idx}.{subset_name}'
        self.htg_meta[key] = {
            'group_num': group_num,
            'boundaries': boundaries,
            'boundaries_norm': boundaries_norm.detach().cpu().tolist(),
            'groups': groups,
            'num_steps': num_steps,
        }
        if self.save_shift:
            self.act_shifts[key] = shift.detach().cpu()
        if self.save_scale:
            self.act_scales[key] = scale.detach().cpu()
