import gc
import os

import torch
from loguru import logger

from llmc.utils.visualizer import visualize_hiband_channel_histogram
from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_


@ALGO_REGISTRY
class SmoothQuant(BaseBlockwiseQuantization):
    def __init__(self, model, quant_config, input, padding_mask, config):
        super().__init__(model, quant_config, input, padding_mask, config)
        special_config = self.quant_config.get('special', {})
        self.alpha = special_config.get('alpha', 0.5)
        hiband_cfg = special_config.get('hiband', {}) if isinstance(special_config, dict) else {}
        self.hiband_enabled = bool(hiband_cfg.get('enabled', False))
        self.hiband_k_min = int(hiband_cfg.get('k_min', -5))
        self.hiband_k_max = int(hiband_cfg.get('k_max', 5))
        self.hiband_use_offset = bool(hiband_cfg.get('use_offset', False))
        self.hiband_eps = float(hiband_cfg.get('eps', 1e-12))
        self.hiband_use_true_qdq_mse = bool(hiband_cfg.get('use_true_qdq_mse', False))
        self.hiband_weight_scale_enabled = bool(
            hiband_cfg.get('weight_scale_enabled', True)
        )
        sample_ratio = hiband_cfg.get('sample_ratio', None)
        if sample_ratio is None:
            self.hiband_sample_ratio = None
        else:
            self.hiband_sample_ratio = float(sample_ratio)
            self.hiband_sample_ratio = max(0.0, min(1.0, self.hiband_sample_ratio))
        self.hiband_sample_min_tokens = max(int(hiband_cfg.get('sample_min_tokens', 1)), 0)
        self.hiband_sample_max_tokens = max(int(hiband_cfg.get('sample_max_tokens', 8192)), 0)
        self.hiband_visualize_hist = bool(hiband_cfg.get('visualize_hist', False))
        self.hiband_visualize_top_channels = max(
            int(hiband_cfg.get('visualize_top_channels', 3)), 0
        )
        self.hiband_visualize_dirname = str(
            hiband_cfg.get('visualize_dirname', 'hiband_histograms')
        )
        if self.hiband_enabled and not hasattr(self, 'hiband_act_scales'):
            self.hiband_act_scales = {}

    @torch.no_grad()
    def _hif8_qdq_like(self, tensor):
        x = tensor.float()
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)
        eps = 2 ** (-45)
        e = torch.floor(torch.log2(x_unsigned + eps))
        abse = e.abs()
        mant_bits = torch.zeros_like(abse)
        mant_bits[abse <= 15] = 1
        mant_bits[abse <= 7] = 2
        mant_bits[abse <= 3] = 3
        qdq = (
            torch.floor(x_unsigned * torch.exp2(-e + mant_bits) + 0.5)
            * torch.exp2(e - mant_bits)
            * sign
        )
        return qdq.to(tensor.dtype)

    @torch.no_grad()
    def _build_hiband_error_weight_table(self, min_e_prime, max_e_prime):
        num_entries = max_e_prime - min_e_prime + 1
        W = torch.zeros(num_entries, dtype=torch.float32)
        for idx in range(num_entries):
            e_prime = min_e_prime + idx
            abse = abs(e_prime)
            if abse <= 3:
                m = 3
            elif abse <= 7:
                m = 2
            elif abse <= 15:
                m = 1
            else:
                m = 0
            W[idx] = 2.0 ** (2 * (e_prime - m)) / 12.0
        self._hiband_w_min_e = min_e_prime
        return W

    @torch.no_grad()
    def _build_hiband_histogram(self, samples):
        samples = samples.to(torch.float32)
        abs_samples = samples.abs()
        if self.hiband_use_offset:
            channel_max = abs_samples.amax(dim=0)
            nonzero_channel = channel_max > self.hiband_eps
            offset = torch.where(
                nonzero_channel,
                torch.round(torch.log2(channel_max.clamp(min=self.hiband_eps))).to(torch.int32),
                torch.zeros_like(channel_max, dtype=torch.int32),
            )
        else:
            offset = torch.zeros(samples.shape[1], dtype=torch.int32, device=samples.device)

        nonzero_mask = abs_samples > self.hiband_eps
        zero_count = (~nonzero_mask).sum(dim=0).to(torch.float32)
        if not bool(nonzero_mask.any().item()):
            hist = torch.zeros((samples.shape[1], 1), dtype=torch.float32, device=samples.device)
            return offset, hist, zero_count, 0

        exponents = torch.floor(torch.log2(abs_samples.clamp(min=self.hiband_eps))).to(torch.int32)
        nonzero_exp = exponents[nonzero_mask]
        min_exp = int(nonzero_exp.min().item())
        max_exp = int(nonzero_exp.max().item())
        hist_bins = max(max_exp - min_exp + 1, 1)

        channel_idx = torch.arange(samples.shape[1], device=samples.device, dtype=torch.long)
        channel_idx = channel_idx.view(1, -1).expand_as(exponents)
        flat_indices = channel_idx[nonzero_mask] * hist_bins + (
            exponents[nonzero_mask] - min_exp
        ).to(torch.long)
        hist_flat = torch.zeros(
            samples.shape[1] * hist_bins, dtype=torch.float32, device=samples.device
        )
        hist_flat.scatter_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32)
        )
        hist = hist_flat.view(samples.shape[1], hist_bins)
        return offset, hist, zero_count, min_exp

    @torch.no_grad()
    def _build_uniform_budgets(self, bucket_count, total_budget):
        if bucket_count <= 0:
            return []
        if total_budget <= 0:
            return [0 for _ in range(bucket_count)]
        base = total_budget // bucket_count
        remain = total_budget % bucket_count
        budgets = [base for _ in range(bucket_count)]
        for idx in range(remain):
            budgets[idx] += 1
        return budgets

    @torch.no_grad()
    def _build_proportional_budgets(self, weights, total_budget):
        if len(weights) == 0:
            return []
        total_budget = int(total_budget)
        if total_budget <= 0:
            return [0 for _ in weights]

        normalized = [max(int(w), 0) for w in weights]
        total_weight = sum(normalized)
        if total_weight <= 0:
            return self._build_uniform_budgets(len(normalized), total_budget)

        raw = [total_budget * w / total_weight for w in normalized]
        budgets = [int(v) for v in raw]
        remain = total_budget - sum(budgets)
        if remain > 0:
            order = sorted(
                range(len(raw)),
                key=lambda idx: (raw[idx] - budgets[idx], normalized[idx], -idx),
                reverse=True,
            )
            for idx in order[:remain]:
                budgets[idx] += 1
        return budgets

    @torch.no_grad()
    def _resolve_hiband_token_budget(self, total_rows):
        total_rows = int(total_rows)
        if total_rows <= 0:
            return 0
        if self.hiband_sample_ratio is None:
            budget = total_rows
        else:
            budget = int(round(total_rows * self.hiband_sample_ratio))
            if self.hiband_sample_ratio > 0.0:
                budget = max(budget, self.hiband_sample_min_tokens)
        budget = min(max(budget, 0), total_rows)
        if self.hiband_sample_max_tokens > 0:
            budget = min(budget, self.hiband_sample_max_tokens)
        return budget

    @torch.no_grad()
    def _sample_rows_evenly(self, rows, sample_count):
        if rows is None:
            return None
        sample_count = int(sample_count)
        if sample_count <= 0:
            return rows[:0]
        if rows.shape[0] <= sample_count:
            return rows
        indices = (
            (torch.arange(sample_count, device=rows.device, dtype=torch.float32) + 0.5)
            * (rows.shape[0] / float(sample_count))
        ).floor().to(torch.long)
        indices = indices.clamp(max=rows.shape[0] - 1)
        return rows.index_select(0, indices)

    @torch.no_grad()
    def _maybe_visualize_hiband_histograms(
        self,
        hist,
        min_exp,
        offset,
        topk_candidates,
        best_k,
        side,
    ):
        if not self.hiband_visualize_hist or self.hiband_visualize_top_channels <= 0:
            return
        if hist is None or hist.numel() == 0:
            return

        model_path = getattr(self.model, 'model_path', None)
        if not model_path:
            return

        save_root = os.path.join(model_path, self.hiband_visualize_dirname)
        side_dir = os.path.join(save_root, f'block_{int(self.block_idx):02d}', side)
        channel_limit = min(self.hiband_visualize_top_channels, hist.shape[0])
        topk_cpu = topk_candidates.detach().cpu()
        best_k_cpu = best_k.detach().cpu()
        offset_cpu = offset.detach().cpu()
        hist_cpu = hist.detach().cpu()

        for channel_idx in range(channel_limit):
            channel_topk = topk_cpu[:, channel_idx].tolist()
            topk_str = '-'.join(str(int(value)) for value in channel_topk)
            file_name = (
                f'block_{int(self.block_idx):02d}_{side}_ch_{channel_idx:04d}'
                f'_topk_{topk_str}.png'
            )
            save_path = os.path.join(side_dir, file_name)
            visualize_hiband_channel_histogram(
                hist_counts=hist_cpu[channel_idx].numpy(),
                min_exp=min_exp,
                save_path=save_path,
                block_idx=int(self.block_idx),
                channel_idx=channel_idx,
                side=side,
                offset=int(offset_cpu[channel_idx].item()),
                best_k=int(best_k_cpu[channel_idx].item()),
                topk_values=channel_topk,
            )

    @torch.no_grad()
    def _resolve_hiband_num_steps(self, tensor_count=None):
        num_steps = int(getattr(self.model, 'sample_steps', 0) or 0)
        if num_steps <= 0:
            return 1
        if tensor_count is None:
            return num_steps
        if tensor_count < num_steps:
            return 1
        return num_steps

    @torch.no_grad()
    def _split_by_timestep(self, tensors, num_steps):
        if num_steps <= 1:
            return [list(tensors)]
        return [list(tensors[step_idx::num_steps]) for step_idx in range(num_steps)]

    @torch.no_grad()
    def _search_hiband_scale(self, samples, target_device, target_dtype, side):
        if samples is None or samples.numel() == 0:
            return None

        samples = samples.to(torch.float32)
        offset, hist, _zero_count, min_exp = self._build_hiband_histogram(samples)
        num_channels = hist.shape[0]
        hist_bins = hist.shape[1]
        N_c = samples.shape[0]
        candidate = torch.arange(
            self.hiband_k_min, self.hiband_k_max + 1, device=samples.device, dtype=torch.int32
        ).view(-1, 1) + offset.view(1, -1)
        candidate_count = candidate.shape[0]

        max_exp = min_exp + hist_bins - 1
        cand_min = int(candidate.min().item())
        cand_max = int(candidate.max().item())
        min_e_prime = min_exp - cand_max
        max_e_prime = max_exp - cand_min
        W = self._build_hiband_error_weight_table(min_e_prime, max_e_prime).to(device=samples.device)

        bin_exp = torch.arange(hist_bins, device=samples.device, dtype=torch.int32) + min_exp
        e_prime = bin_exp.view(1, 1, -1) - candidate.unsqueeze(2)
        w_idx = (e_prime - self._hiband_w_min_e).clamp(min=0, max=W.shape[0] - 1)
        weight_table = W[w_idx.long()]
        analytical_mse_scaled = torch.einsum('icb,cb->ic', weight_table, hist) / N_c
        scale_sq = torch.pow(2.0, 2.0 * candidate.to(torch.float32))
        analytical_mse = analytical_mse_scaled * scale_sq
        viz_topk = min(3, candidate_count)
        topk_for_viz = candidate.gather(
            0,
            analytical_mse.topk(viz_topk, dim=0, largest=False).indices,
        )
        k_anchor = torch.zeros(num_channels, device=samples.device, dtype=torch.int32)
        if self.hiband_use_true_qdq_mse:
            best_k = candidate[0].clone()
            best_mse = None
            best_analytical = analytical_mse[0].clone()
            for idx in range(candidate_count):
                k = candidate[idx].to(samples.dtype)
                scale = torch.pow(2.0, k).clamp(min=self.hiband_eps)
                qdq = self._hif8_qdq_like(samples / scale.view(1, -1)) * scale.view(1, -1)
                mse = (qdq - samples).pow(2).mean(dim=0)
                if best_mse is None:
                    best_mse = mse
                    best_k = candidate[idx].clone()
                    best_analytical = analytical_mse[idx].clone()
                    continue

                better = mse < best_mse
                tie = torch.isclose(mse, best_mse)
                better_tie = tie & (
                    (candidate[idx] - k_anchor).abs() < (best_k - k_anchor).abs()
                )
                better_tie = better_tie | (
                    tie
                    & ((candidate[idx] - k_anchor).abs() == (best_k - k_anchor).abs())
                    & (analytical_mse[idx] < best_analytical)
                )
                better = better | better_tie
                best_mse = torch.where(better, mse, best_mse)
                best_analytical = torch.where(better, analytical_mse[idx], best_analytical)
                best_k = torch.where(better, candidate[idx], best_k)
        else:
            best_idx = analytical_mse.argmin(dim=0)
            best_k = candidate.gather(0, best_idx.unsqueeze(0)).squeeze(0)
            best_analytical = analytical_mse.gather(0, best_idx.unsqueeze(0)).squeeze(0)
            close_mask = torch.isclose(analytical_mse, best_analytical.unsqueeze(0))
            tie_dist = (candidate - k_anchor.unsqueeze(0)).abs().to(torch.float32)
            tie_dist = torch.where(
                close_mask, tie_dist, torch.full_like(tie_dist, float('inf'))
            )
            tie_idx = tie_dist.argmin(dim=0)
            best_k = candidate.gather(0, tie_idx.unsqueeze(0)).squeeze(0)

        self._maybe_visualize_hiband_histograms(
            hist=hist,
            min_exp=min_exp,
            offset=offset,
            topk_candidates=topk_for_viz,
            best_k=best_k,
            side=side,
        )
        hiband_scale = torch.pow(2.0, best_k.to(torch.float32)).clamp(min=self.hiband_eps)
        return hiband_scale.to(device=target_device, dtype=target_dtype)

    @torch.no_grad()
    def _collect_hiband_act_samples(self, tensors, scale):
        samples = []
        num_steps = self._resolve_hiband_num_steps(len(tensors))
        step_buckets = self._split_by_timestep(tensors, num_steps)
        total_rows = 0
        for bucket in step_buckets:
            for tensor in bucket:
                if tensor.shape[-1] > 0:
                    total_rows += int(tensor.numel() // tensor.shape[-1])
        target_budget = self._resolve_hiband_token_budget(total_rows)
        step_budgets = self._build_uniform_budgets(len(step_buckets), target_budget)
        for bucket, step_budget in zip(step_buckets, step_budgets):
            if step_budget <= 0 or len(bucket) == 0:
                continue
            tensor_budgets = self._build_uniform_budgets(len(bucket), step_budget)
            for tensor, tensor_budget in zip(bucket, tensor_budgets):
                if tensor_budget <= 0:
                    continue
                shape = [1] * (tensor.dim() - 1) + [-1]
                post = tensor / scale.to(device=tensor.device, dtype=tensor.dtype).view(*shape)
                post = post.view(-1, post.shape[-1]).detach().to(torch.float32)
                post = self._sample_rows_evenly(post, min(post.shape[0], tensor_budget))
                if post.numel() > 0:
                    samples.append(post)
        if len(samples) == 0:
            return None
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def _collect_hiband_act_samples_stream(self, stream_stats, input_name, scale):
        if self._current_block is None:
            raise ValueError('SmoothQuant stream_stats HiBand requires current block context.')

        samples = []
        stream_state = {'idx': 0}
        scale_cpu = scale.detach().to(torch.float32).cpu()
        num_steps = self._resolve_hiband_num_steps()
        step_counts = stream_stats.get('step_shift_count', None)
        if not isinstance(step_counts, list) or len(step_counts) != num_steps:
            step_counts = [1 for _ in range(num_steps)]
        step_token_rows = stream_stats.get('step_token_rows', None)
        if not isinstance(step_token_rows, list) or len(step_token_rows) != num_steps:
            step_token_rows = [
                [1 for _ in range(max(int(step_counts[step_idx]), 1))]
                for step_idx in range(num_steps)
            ]
        normalized_step_rows = []
        for step_idx in range(num_steps):
            rows = step_token_rows[step_idx]
            if not isinstance(rows, list) or len(rows) != max(int(step_counts[step_idx]), 1):
                rows = [1 for _ in range(max(int(step_counts[step_idx]), 1))]
            normalized_step_rows.append([max(int(row), 0) for row in rows])
        step_total_rows = [sum(rows) for rows in normalized_step_rows]
        total_rows = sum(step_total_rows)
        target_budget = self._resolve_hiband_token_budget(total_rows)
        step_budgets = self._build_proportional_budgets(step_total_rows, target_budget)
        step_occurrence_budgets = [
            self._build_proportional_budgets(
                normalized_step_rows[step_idx], step_budgets[step_idx]
            )
            for step_idx in range(num_steps)
        ]
        step_occurrence_cursor = [0 for _ in range(num_steps)]

        def collect_hiband_post_scale(_, x, _output):
            step_idx = stream_state['idx'] % num_steps
            occurrence_idx = step_occurrence_cursor[step_idx]
            step_occurrence_cursor[step_idx] += 1
            stream_state['idx'] += 1
            if occurrence_idx >= len(step_occurrence_budgets[step_idx]):
                return
            occurrence_budget = step_occurrence_budgets[step_idx][occurrence_idx]
            if occurrence_budget <= 0:
                return
            if len(x) == 0 or not torch.is_tensor(x[0]):
                return

            inp = x[0].detach()
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            shape = [1] * (inp.dim() - 1) + [-1]
            post = inp / scale_cpu.to(device=inp.device, dtype=inp.dtype).view(*shape)
            post = post.view(-1, post.shape[-1]).to(torch.float32).cpu()
            post = self._sample_rows_evenly(post, min(post.shape[0], occurrence_budget))
            if post.numel() > 0:
                samples.append(post)

        layer = dict(self._current_block.named_modules())[input_name]
        handle = layer.register_forward_hook(collect_hiband_post_scale)
        try:
            self.block_forward(self._current_block, collect_output=False)
        finally:
            handle.remove()

        if len(samples) == 0:
            return None
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def _collect_hiband_weight_samples(self, layer, scale):
        if not hasattr(layer, 'weight'):
            return None
        weight = layer.weight.detach()
        if weight.shape[-1] != scale.numel():
            return None
        merged = weight.to(torch.float32) * scale.to(
            device=weight.device, dtype=torch.float32
        ).view(1, -1)
        merged = merged.view(-1, merged.shape[-1])
        budget = self._resolve_hiband_token_budget(int(merged.shape[0]))
        return self._sample_rows_evenly(merged, min(merged.shape[0], budget))

    @torch.no_grad()
    def _estimate_hiband_act_output_moments(self, act_samples, hiband_act_scale):
        if (
            act_samples is None
            or act_samples.numel() == 0
            or hiband_act_scale is None
            or hiband_act_scale.numel() != act_samples.shape[1]
        ):
            return None, None

        act_samples = act_samples.to(torch.float32)
        act_scale = hiband_act_scale.to(
            device=act_samples.device, dtype=torch.float32
        ).view(1, -1)
        x_hat = self._hif8_qdq_like(act_samples / act_scale) * act_scale
        alpha = x_hat.pow(2).mean(dim=0).clamp(min=self.hiband_eps)
        beta = (x_hat * act_samples).mean(dim=0)
        return alpha, beta

    @torch.no_grad()
    def _attach_hiband_act_scale(self, layers, hiband_act_scale):
        for layer in layers:
            if not hasattr(layer, 'weight') or layer.weight.shape[-1] != hiband_act_scale.numel():
                continue
            value = hiband_act_scale.to(device=layer.weight.device, dtype=torch.float32)
            if hasattr(layer, 'hiband_act_scale'):
                layer.hiband_act_scale.data = value
            else:
                layer.register_buffer('hiband_act_scale', value)

    @torch.no_grad()
    def _search_hiband_weight_scale(
        self,
        layer,
        base_scale,
        act_alpha,
        act_beta,
        target_device,
        target_dtype,
    ):
        if (
            not hasattr(layer, 'weight')
            or act_alpha is None
            or act_beta is None
            or layer.weight.shape[-1] != act_alpha.numel()
            or act_alpha.numel() != act_beta.numel()
        ):
            return None

        post_weight = layer.weight.detach().to(torch.float32) * base_scale.to(
            device=layer.weight.device, dtype=torch.float32
        ).view(1, -1)
        if self.hiband_use_offset:
            channel_max = post_weight.abs().amax(dim=0)
            nonzero_channel = channel_max > self.hiband_eps
            offset = torch.where(
                nonzero_channel,
                torch.round(torch.log2(channel_max.clamp(min=self.hiband_eps))).to(
                    torch.int32
                ),
                torch.zeros_like(channel_max, dtype=torch.int32),
            )
        else:
            offset = torch.zeros(
                post_weight.shape[1], dtype=torch.int32, device=post_weight.device
            )

        candidate = torch.arange(
            self.hiband_k_min,
            self.hiband_k_max + 1,
            device=post_weight.device,
            dtype=torch.int32,
        ).view(-1, 1) + offset.view(1, -1)
        candidate_count = candidate.shape[0]
        k_anchor = torch.zeros_like(offset)
        best_score = None
        best_k = candidate[0].clone()

        act_alpha = act_alpha.to(device=post_weight.device, dtype=torch.float32)
        act_beta = act_beta.to(device=post_weight.device, dtype=torch.float32)
        for idx in range(candidate_count):
            k = candidate[idx].to(torch.float32)
            cand_scale = torch.pow(2.0, k).clamp(min=self.hiband_eps).view(1, -1)
            w_hb = self._hif8_qdq_like(post_weight / cand_scale) * cand_scale
            cand_norm = w_hb.pow(2).sum(dim=0)
            cand_dot = (w_hb * post_weight).sum(dim=0)
            score = act_alpha * cand_norm - 2.0 * act_beta * cand_dot
            if best_score is None:
                best_score = score
                best_k = candidate[idx].clone()
                continue

            better = score < best_score
            tie = torch.isclose(score, best_score)
            better_tie = tie & (
                (candidate[idx] - k_anchor).abs() < (best_k - k_anchor).abs()
            )
            best_score = torch.where(better | better_tie, score, best_score)
            best_k = torch.where(better | better_tie, candidate[idx], best_k)

        hiband_scale = torch.pow(2.0, best_k.to(torch.float32)).clamp(min=self.hiband_eps)
        return hiband_scale.to(device=target_device, dtype=target_dtype)

    @torch.no_grad()
    def _search_hiband_weight_scales(
        self,
        layers,
        scale,
        act_samples,
        hiband_act_scale,
        target_device,
        target_dtype,
    ):
        act_alpha, act_beta = self._estimate_hiband_act_output_moments(
            act_samples, hiband_act_scale
        )
        if act_alpha is None or act_beta is None:
            return {}
        hiband_weight_scales = {}
        for layer in layers:
            layer_scale = self._search_hiband_weight_scale(
                layer, scale, act_alpha, act_beta, target_device, target_dtype
            )
            if layer_scale is not None:
                hiband_weight_scales[layer] = layer_scale
        return hiband_weight_scales

    @torch.no_grad()
    def _apply_hiband_weight_scale(self, layer, hiband_weight_scale):
        if not hasattr(layer, 'weight') or layer.weight.shape[-1] != hiband_weight_scale.numel():
            return
        scale = hiband_weight_scale.to(device=layer.weight.device, dtype=torch.float32).view(1, -1)
        weight = layer.weight.detach().to(torch.float32)
        qdq_weight = self._hif8_qdq_like(weight / scale) * scale
        layer.weight.data.copy_(qdq_weight.to(dtype=layer.weight.dtype))

    @torch.no_grad()
    def filter_subset(self, prev_op):
        if isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_)):
            return True
        else:
            return False

    @torch.no_grad()
    def get_weight_scale(self, layers):
        weights = self.collect_layers_weights(layers)
        scale = torch.cat(
            [fc.abs().max(dim=0, keepdim=True)[0] for fc in weights], dim=0
        )
        scale = scale.max(dim=0)[0].clamp(min=1e-5)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        return scale

    @torch.no_grad()
    def get_act_scale(self, tensors):
        scale_max = None
        for x in tensors:
            x = x.cuda()
            comming_max = self._channel_abs_max(x)
            if scale_max is not None:
                scale_max = torch.max(scale_max, comming_max)
            else:
                scale_max = comming_max
            x = x.cpu()
        return scale_max

    @torch.no_grad()
    def _channel_abs_max(self, x):
        x = x.view(-1, x.shape[-1])
        if not self.stream_stats or self.stream_chunk_size <= 0 or x.shape[0] <= self.stream_chunk_size:
            return x.abs().max(dim=0)[0]

        max_values = None
        for start in range(0, x.shape[0], self.stream_chunk_size):
            end = min(start + self.stream_chunk_size, x.shape[0])
            current = x[start:end].abs().max(dim=0)[0]
            max_values = current if max_values is None else torch.maximum(max_values, current)
        return max_values

    @torch.no_grad()
    def _get_stream_act_scale(self, stream_stats):
        scale_max = stream_stats.get('act_abs_max', None)
        if scale_max is None:
            raise ValueError('stream_stats enabled but act_abs_max is empty.')
        return scale_max

    @torch.no_grad()
    def search_scale_subset(self, layers, tensors):
        w_max = self.get_weight_scale(layers)
        if self.stream_stats and isinstance(tensors, dict):
            x_max = self._get_stream_act_scale(tensors)
        else:
            x_max = self.get_act_scale(tensors)
        x_max = x_max.to(dtype=w_max.dtype, device=w_max.device)
        scale = (x_max.pow(self.alpha) / w_max.pow(1 - self.alpha)).clamp(min=1e-5)
        return scale

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
        flat = inp.abs().view(-1, inp.shape[-1])
        token_row_count = flat.shape[0]
        inp = flat.amax(dim=0).to(torch.float32).cpu()

        if not isinstance(feat_dict[name], dict):
            feat_dict[name] = {}
        if 'act_abs_max' in feat_dict[name]:
            feat_dict[name]['act_abs_max'] = torch.maximum(
                feat_dict[name]['act_abs_max'], inp
            )
        else:
            feat_dict[name]['act_abs_max'] = inp

        num_steps = self._resolve_hiband_num_steps()
        if num_steps > 1:
            if 'step_shift_count' not in feat_dict[name]:
                feat_dict[name]['step_shift_count'] = [0 for _ in range(num_steps)]
                feat_dict[name]['step_token_rows'] = [[] for _ in range(num_steps)]
                feat_dict[name]['hiband_call_idx'] = 0
            step_idx = feat_dict[name]['hiband_call_idx'] % num_steps
            feat_dict[name]['step_shift_count'][step_idx] += 1
            feat_dict[name]['step_token_rows'][step_idx].append(int(token_row_count))
            feat_dict[name]['hiband_call_idx'] += 1

    @torch.no_grad()
    def subset_transform(
        self,
        subset,
        input_feat,
        subset_kwargs,
    ):
        del subset_kwargs
        layers_dict = subset['layers']
        prev_op = subset['prev_op']
        input_name = subset['input'][0]

        if not self.filter_subset(prev_op):
            logger.info('Do not transform this subset.')
            return
        layers = list(layers_dict.values())
        scale = self.search_scale_subset(layers, input_feat[input_name])
        hiband_act_scale = None
        hiband_weight_scales = {}
        if self.hiband_enabled:
            if self.stream_stats and isinstance(input_feat[input_name], dict):
                act_samples = self._collect_hiband_act_samples_stream(
                    input_feat[input_name], input_name, scale
                )
            else:
                act_samples = self._collect_hiband_act_samples(input_feat[input_name], scale)
            hiband_act_scale = self._search_hiband_scale(
                act_samples, scale.device, scale.dtype, side='act'
            )
            if self.hiband_weight_scale_enabled and hiband_act_scale is not None:
                hiband_weight_scales = self._search_hiband_weight_scales(
                    layers,
                    scale,
                    act_samples,
                    hiband_act_scale,
                    scale.device,
                    scale.dtype,
                )
            if hiband_act_scale is not None:
                self._attach_hiband_act_scale(layers, hiband_act_scale)
        self.apply_scale(scale, prev_op, layers)
        for layer, layer_scale in hiband_weight_scales.items():
            self._apply_hiband_weight_scale(layer, layer_scale)
        if self.act_static:
            self.update_input_feat(scale, input_feat, layers_dict, False)
        if self.hiband_enabled and hiband_act_scale is not None:
            key = ','.join(sorted(layers_dict.keys()))
            self.hiband_act_scales[key] = hiband_act_scale.detach().cpu()
