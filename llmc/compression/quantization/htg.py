import gc
import os
from typing import Dict, List

import torch
import torch.nn as nn
from loguru import logger

from llmc.utils.visualizer import visualize_hiband_channel_histogram
from llmc.utils.registry_factory import ALGO_REGISTRY

from .base_blockwise_quantization import BaseBlockwiseQuantization
from .module_utils import _LLMC_LN_TYPES_, _TRANSFORMERS_LN_TYPES_
from .quant import (
    HIF8_HIBAND_FIXED_BINS,
    HIF8_HIBAND_FIXED_MIN_EXP,
    HIF8_INF_TIE_RATIO,
    HIF8_MAX_NORMAL_EXP,
    hif8_bucket_mse_ta,
    hif8_qdq_ta,
)


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
        self.hiband_runtime_group_act_scale_enabled = bool(
            hiband_cfg.get('runtime_group_act_scale_enabled', False)
        )
        self.hiband_group_source = str(hiband_cfg.get('group_source', 'htg')).lower()
        self.hiband_export_global_act_scale = bool(
            hiband_cfg.get('export_global_act_scale', True)
        )
        self.hiband_visualize_hist = bool(hiband_cfg.get('visualize_hist', False))
        self.hiband_visualize_top_channels = max(
            int(hiband_cfg.get('visualize_top_channels', 3)), 0
        )
        self.hiband_visualize_dirname = str(
            hiband_cfg.get('visualize_dirname', 'hiband_histograms')
        )

        # Kept for optional export in later phases.
        self.htg_meta = {}
        if self.save_shift and not hasattr(self, 'act_shifts'):
            self.act_shifts = {}
        if self.hiband_enabled and not hasattr(self, 'hiband_act_scales'):
            self.hiband_act_scales = {}
        if self.hiband_enabled and not hasattr(self, 'hiband_group_act_scales'):
            self.hiband_group_act_scales = {}

    @torch.no_grad()
    def _hif8_qdq_like(self, tensor):
        return hif8_qdq_ta(tensor)

    @torch.no_grad()
    def _init_hiband_histogram_state(self, num_channels, device):
        hist = torch.zeros(
            (num_channels, HIF8_HIBAND_FIXED_BINS),
            dtype=torch.float32,
            device=device,
        )
        overflow_tail_hist = torch.zeros_like(hist)
        zero_count = torch.zeros(num_channels, dtype=torch.float32, device=device)
        channel_max = torch.zeros(num_channels, dtype=torch.float32, device=device)
        return {
            'hist': hist,
            'overflow_tail_hist': overflow_tail_hist,
            'zero_count': zero_count,
            'channel_max': channel_max,
            'N_c': 0,
            'num_channels': num_channels,
            'device': device,
        }

    @torch.no_grad()
    def _incremental_update_hiband_histogram(self, state, chunk):
        chunk = chunk.to(torch.float32)
        abs_chunk = chunk.abs()
        num_channels = state['num_channels']
        if abs_chunk.shape[-1] != num_channels:
            return

        state['channel_max'] = torch.maximum(
            state['channel_max'], abs_chunk.amax(dim=0)
        )

        nonzero_mask = abs_chunk > self.hiband_eps
        state['zero_count'] += (~nonzero_mask).sum(dim=0).to(torch.float32)
        state['N_c'] += chunk.shape[0]

        if not bool(nonzero_mask.any().item()):
            return

        exponents = torch.floor(
            torch.log2(abs_chunk.clamp(min=self.hiband_eps))
        ).to(torch.int32)
        bin_idx = (exponents - HIF8_HIBAND_FIXED_MIN_EXP).to(torch.long)
        clamped_bin_idx = bin_idx.clamp(min=0, max=HIF8_HIBAND_FIXED_BINS - 1)

        nz_row, nz_col = nonzero_mask.nonzero(as_tuple=True)
        if nz_row.numel() > 0:
            nz_bins = clamped_bin_idx[nz_row, nz_col]
            flat_idx = nz_col * HIF8_HIBAND_FIXED_BINS + nz_bins
            chunk_hist = torch.zeros(
                num_channels * HIF8_HIBAND_FIXED_BINS,
                dtype=torch.float32,
                device=state['device'],
            )
            chunk_hist.scatter_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
            )
            state['hist'] += chunk_hist.view(num_channels, HIF8_HIBAND_FIXED_BINS)

        bin_base = torch.exp2(exponents.to(torch.float32))
        overflow_tail_mask = nonzero_mask & (abs_chunk >= HIF8_INF_TIE_RATIO * bin_base)
        ot_nz_row, ot_nz_col = overflow_tail_mask.nonzero(as_tuple=True)
        if ot_nz_row.numel() > 0:
            ot_bins = clamped_bin_idx[ot_nz_row, ot_nz_col]
            ot_flat_idx = ot_nz_col * HIF8_HIBAND_FIXED_BINS + ot_bins
            ot_chunk_hist = torch.zeros(
                num_channels * HIF8_HIBAND_FIXED_BINS,
                dtype=torch.float32,
                device=state['device'],
            )
            ot_chunk_hist.scatter_add_(
                0, ot_flat_idx, torch.ones_like(ot_flat_idx, dtype=torch.float32)
            )
            state['overflow_tail_hist'] += ot_chunk_hist.view(
                num_channels, HIF8_HIBAND_FIXED_BINS
            )

    @torch.no_grad()
    def _finalize_hiband_histogram_state(self, state):
        if state['N_c'] == 0:
            num_channels = state['num_channels']
            device = state['device']
            offset = torch.zeros(num_channels, dtype=torch.int32, device=device)
            hist = torch.zeros((num_channels, 1), dtype=torch.float32, device=device)
            overflow_tail_hist = torch.zeros_like(hist)
            zero_count = torch.zeros(num_channels, dtype=torch.float32, device=device)
            return offset, hist, overflow_tail_hist, zero_count, 0, 0

        if self.hiband_use_offset:
            nonzero_channel = state['channel_max'] > self.hiband_eps
            offset = torch.where(
                nonzero_channel,
                torch.round(
                    torch.log2(state['channel_max'].clamp(min=self.hiband_eps))
                ).to(torch.int32),
                torch.zeros_like(state['channel_max'], dtype=torch.int32),
            )
        else:
            offset = torch.zeros(
                state['num_channels'], dtype=torch.int32, device=state['device']
            )

        return (
            offset,
            state['hist'],
            state['overflow_tail_hist'],
            state['zero_count'],
            HIF8_HIBAND_FIXED_MIN_EXP,
            state['N_c'],
        )

    @torch.no_grad()
    def _build_hiband_error_weight_table(self, min_e_prime, max_e_prime):
        num_entries = max_e_prime - min_e_prime + 1
        W = torch.zeros(num_entries, dtype=torch.float32)
        for idx in range(num_entries):
            e_prime = min_e_prime + idx
            if e_prime >= (HIF8_MAX_NORMAL_EXP + 1):
                W[idx] = 0.0
            elif e_prime == HIF8_MAX_NORMAL_EXP:
                W[idx] = 0.0
            else:
                W[idx] = hif8_bucket_mse_ta(e_prime, interval='full')
        self._hiband_w_min_e = min_e_prime
        self._hiband_exp15_safe_weight = hif8_bucket_mse_ta(
            HIF8_MAX_NORMAL_EXP, interval='safe_exp15'
        )
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
            overflow_tail_hist = torch.zeros_like(hist)
            return offset, hist, overflow_tail_hist, zero_count, 0

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
        bin_base = torch.exp2(exponents.to(torch.float32))
        overflow_tail_mask = nonzero_mask & (abs_samples >= HIF8_INF_TIE_RATIO * bin_base)
        overflow_tail_flat = torch.zeros_like(hist_flat)
        overflow_tail_indices = flat_indices[overflow_tail_mask[nonzero_mask]]
        overflow_tail_flat.scatter_add_(
            0, overflow_tail_indices, torch.ones_like(overflow_tail_indices, dtype=torch.float32)
        )
        overflow_tail_hist = overflow_tail_flat.view(samples.shape[1], hist_bins)
        return offset, hist, overflow_tail_hist, zero_count, min_exp

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
    def _search_hiband_scale(self, samples, target_device, target_dtype, side):
        if samples is None or samples.numel() == 0:
            return None

        samples = samples.to(torch.float32)
        offset, hist, overflow_tail_hist, _zero_count, min_exp = self._build_hiband_histogram(samples)
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
        exp15_mask = (e_prime == HIF8_MAX_NORMAL_EXP).to(torch.float32)
        safe_exp15_hist = (hist - overflow_tail_hist).clamp(min=0.0)
        safe_exp15_count = torch.einsum('icb,cb->ic', exp15_mask, safe_exp15_hist)
        analytical_mse_scaled = analytical_mse_scaled + (
            safe_exp15_count * self._hiband_exp15_safe_weight / N_c
        )
        overflow_full_mask = (e_prime >= (HIF8_MAX_NORMAL_EXP + 1)).to(torch.float32)
        overflow_full_count = torch.einsum('icb,cb->ic', overflow_full_mask, hist)
        overflow_tail_count = torch.einsum('icb,cb->ic', exp15_mask, overflow_tail_hist)
        overflow_present = (overflow_full_count + overflow_tail_count) > 0
        analytical_mse_scaled = torch.where(
            overflow_present,
            torch.full_like(analytical_mse_scaled, float('inf')),
            analytical_mse_scaled,
        )
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
    def _search_hiband_scale_from_histogram(
        self, offset, hist, overflow_tail_hist, zero_count, min_exp, N_c,
        target_device, target_dtype, side,
    ):
        if N_c == 0 or hist.numel() == 0:
            return None

        num_channels = hist.shape[0]
        hist_bins = hist.shape[1]
        candidate = torch.arange(
            self.hiband_k_min, self.hiband_k_max + 1, device=hist.device, dtype=torch.int32
        ).view(-1, 1) + offset.view(1, -1)
        candidate_count = candidate.shape[0]

        max_exp = min_exp + hist_bins - 1
        cand_min = int(candidate.min().item())
        cand_max = int(candidate.max().item())
        min_e_prime = min_exp - cand_max
        max_e_prime = max_exp - cand_min
        W = self._build_hiband_error_weight_table(min_e_prime, max_e_prime).to(device=hist.device)

        bin_exp = torch.arange(hist_bins, device=hist.device, dtype=torch.int32) + min_exp
        e_prime = bin_exp.view(1, 1, -1) - candidate.unsqueeze(2)
        w_idx = (e_prime - self._hiband_w_min_e).clamp(min=0, max=W.shape[0] - 1)
        weight_table = W[w_idx.long()]
        analytical_mse_scaled = torch.einsum('icb,cb->ic', weight_table, hist) / N_c
        exp15_mask = (e_prime == HIF8_MAX_NORMAL_EXP).to(torch.float32)
        safe_exp15_hist = (hist - overflow_tail_hist).clamp(min=0.0)
        safe_exp15_count = torch.einsum('icb,cb->ic', exp15_mask, safe_exp15_hist)
        analytical_mse_scaled = analytical_mse_scaled + (
            safe_exp15_count * self._hiband_exp15_safe_weight / N_c
        )
        overflow_full_mask = (e_prime >= (HIF8_MAX_NORMAL_EXP + 1)).to(torch.float32)
        overflow_full_count = torch.einsum('icb,cb->ic', overflow_full_mask, hist)
        overflow_tail_count = torch.einsum('icb,cb->ic', exp15_mask, overflow_tail_hist)
        overflow_present = (overflow_full_count + overflow_tail_count) > 0
        analytical_mse_scaled = torch.where(
            overflow_present,
            torch.full_like(analytical_mse_scaled, float('inf')),
            analytical_mse_scaled,
        )
        scale_sq = torch.pow(2.0, 2.0 * candidate.to(torch.float32))
        analytical_mse = analytical_mse_scaled * scale_sq
        viz_topk = min(3, candidate_count)
        topk_for_viz = candidate.gather(
            0,
            analytical_mse.topk(viz_topk, dim=0, largest=False).indices,
        )
        k_anchor = torch.zeros(num_channels, device=hist.device, dtype=torch.int32)
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
    def _collect_hiband_histogram(
        self, tensors, num_steps, step_to_group, z_g, scale
    ):
        step_buckets = self._split_by_timestep(tensors, num_steps)
        num_channels = None
        for bucket in step_buckets:
            for tensor in bucket:
                if tensor.shape[-1] > 0:
                    num_channels = tensor.shape[-1]
                    break
            if num_channels is not None:
                break
        if num_channels is None:
            return None

        state = self._init_hiband_histogram_state(num_channels, device='cpu')
        for step_idx, bucket in enumerate(step_buckets):
            if len(bucket) == 0:
                continue
            group_idx = step_to_group[step_idx]
            for tensor in bucket:
                if tensor.shape[-1] == 0:
                    continue
                shift = z_g[group_idx].to(device=tensor.device, dtype=tensor.dtype)
                shape = [1] * (tensor.dim() - 1) + [-1]
                post = tensor - shift.view(*shape)
                post = post / scale.to(device=tensor.device, dtype=tensor.dtype).view(*shape)
                post = post.view(-1, post.shape[-1]).detach().to(torch.float32).cpu()
                chunk_size = self.hiband_sample_max_tokens if self.hiband_sample_max_tokens > 0 else 8192
                for start in range(0, post.shape[0], chunk_size):
                    chunk = post[start:start + chunk_size]
                    self._incremental_update_hiband_histogram(state, chunk)
                del post

        result = self._finalize_hiband_histogram_state(state)
        if result[5] == 0:
            return None
        return result

    @torch.no_grad()
    def _collect_hiband_histogram_stream(
        self,
        _stream_stats,
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
        input_name: str,
        num_steps: int,
        scale: torch.Tensor,
    ):
        result, _ = self._collect_hiband_histograms_stream(
            z_g,
            step_to_group,
            input_name,
            num_steps,
            scale,
            collect_global=True,
            collect_grouped=False,
        )
        return result

    @torch.no_grad()
    def _collect_hiband_group_histograms(
        self, tensors, num_steps, step_to_group, z_g, scale, group_num
    ):
        step_buckets = self._split_by_timestep(tensors, num_steps)
        num_channels = None
        for bucket in step_buckets:
            for tensor in bucket:
                if tensor.shape[-1] > 0:
                    num_channels = tensor.shape[-1]
                    break
            if num_channels is not None:
                break
        if num_channels is None:
            return [None for _ in range(group_num)]

        states = [
            self._init_hiband_histogram_state(num_channels, device='cpu')
            for _ in range(group_num)
        ]
        for step_idx, bucket in enumerate(step_buckets):
            if len(bucket) == 0:
                continue
            group_idx = step_to_group[step_idx]
            for tensor in bucket:
                if tensor.shape[-1] == 0:
                    continue
                shift = z_g[group_idx].to(device=tensor.device, dtype=tensor.dtype)
                shape = [1] * (tensor.dim() - 1) + [-1]
                post = tensor - shift.view(*shape)
                post = post / scale.to(device=tensor.device, dtype=tensor.dtype).view(*shape)
                post = post.view(-1, post.shape[-1]).detach().to(torch.float32).cpu()
                chunk_size = self.hiband_sample_max_tokens if self.hiband_sample_max_tokens > 0 else 8192
                for start in range(0, post.shape[0], chunk_size):
                    chunk = post[start:start + chunk_size]
                    self._incremental_update_hiband_histogram(states[group_idx], chunk)
                del post

        results = []
        for state in states:
            result = self._finalize_hiband_histogram_state(state)
            if result[5] == 0:
                results.append(None)
            else:
                results.append(result)
        return results

    @torch.no_grad()
    def _collect_hiband_group_histograms_stream(
        self,
        _stream_stats,
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
        input_name: str,
        num_steps: int,
        scale: torch.Tensor,
        group_num: int,
    ):
        _, grouped_results = self._collect_hiband_histograms_stream(
            z_g,
            step_to_group,
            input_name,
            num_steps,
            scale,
            group_num=group_num,
            collect_global=False,
            collect_grouped=True,
        )
        return grouped_results

    @torch.no_grad()
    def _collect_hiband_act_samples(self, tensors, num_steps, step_to_group, z_g, scale):
        samples = []
        step_buckets = self._split_by_timestep(tensors, num_steps)
        total_rows = 0
        for bucket in step_buckets:
            for tensor in bucket:
                if tensor.shape[-1] > 0:
                    total_rows += int(tensor.numel() // tensor.shape[-1])
        target_budget = self._resolve_hiband_token_budget(total_rows)
        step_budgets = self._build_uniform_budgets(num_steps, target_budget)
        for step_idx, bucket in enumerate(step_buckets):
            step_budget = step_budgets[step_idx]
            if step_budget <= 0 or len(bucket) == 0:
                continue
            group_idx = step_to_group[step_idx]
            tensor_budgets = self._build_uniform_budgets(len(bucket), step_budget)
            for tensor, tensor_budget in zip(bucket, tensor_budgets):
                if tensor_budget <= 0:
                    continue
                shift = z_g[group_idx].to(device=tensor.device, dtype=tensor.dtype)
                shape = [1] * (tensor.dim() - 1) + [-1]
                shifted = tensor - shift.view(*shape)
                post = shifted / scale.to(device=tensor.device, dtype=tensor.dtype).view(*shape)
                post = post.view(-1, post.shape[-1]).detach().to(torch.float32)
                post = self._sample_rows_evenly(post, min(post.shape[0], tensor_budget))
                if post.numel() > 0:
                    samples.append(post)
        if len(samples) == 0:
            return None
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def _collect_hiband_group_act_samples(
        self, tensors, num_steps, step_to_group, z_g, scale, group_num
    ):
        grouped_samples = [[] for _ in range(group_num)]
        step_buckets = self._split_by_timestep(tensors, num_steps)
        total_rows = 0
        for bucket in step_buckets:
            for tensor in bucket:
                if tensor.shape[-1] > 0:
                    total_rows += int(tensor.numel() // tensor.shape[-1])
        target_budget = self._resolve_hiband_token_budget(total_rows)
        step_budgets = self._build_uniform_budgets(num_steps, target_budget)
        for step_idx, bucket in enumerate(step_buckets):
            step_budget = step_budgets[step_idx]
            if step_budget <= 0 or len(bucket) == 0:
                continue
            group_idx = step_to_group[step_idx]
            tensor_budgets = self._build_uniform_budgets(len(bucket), step_budget)
            for tensor, tensor_budget in zip(bucket, tensor_budgets):
                if tensor_budget <= 0:
                    continue
                shift = z_g[group_idx].to(device=tensor.device, dtype=tensor.dtype)
                shape = [1] * (tensor.dim() - 1) + [-1]
                shifted = tensor - shift.view(*shape)
                post = shifted / scale.to(device=tensor.device, dtype=tensor.dtype).view(*shape)
                post = post.view(-1, post.shape[-1]).detach().to(torch.float32)
                post = self._sample_rows_evenly(post, min(post.shape[0], tensor_budget))
                if post.numel() > 0:
                    grouped_samples[group_idx].append(post)

        merged = []
        for group_samples in grouped_samples:
            if len(group_samples) == 0:
                merged.append(None)
            else:
                merged.append(torch.cat(group_samples, dim=0))
        return merged

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
    def _attach_hiband_group_act_scales(self, layers, hiband_group_act_scales):
        for layer in layers:
            if (
                not hasattr(layer, 'weight')
                or hiband_group_act_scales.dim() != 2
                or layer.weight.shape[-1] != hiband_group_act_scales.shape[-1]
            ):
                continue
            value = hiband_group_act_scales.to(device=layer.weight.device, dtype=torch.float32)
            if hasattr(layer, 'hiband_group_act_scales'):
                layer.hiband_group_act_scales.data = value
            else:
                layer.register_buffer('hiband_group_act_scales', value)

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
        return isinstance(prev_op[0], tuple(_LLMC_LN_TYPES_ + _TRANSFORMERS_LN_TYPES_))

    @torch.no_grad()
    def _is_attn_o_subset(self, subset):
        return bool(subset.get('is_attn_o', False))

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
    def _channel_min_max(self, x):
        x = x.cuda()
        x = x.view(-1, x.shape[-1])
        if not self.stream_stats or self.stream_chunk_size <= 0 or x.shape[0] <= self.stream_chunk_size:
            return x.max(dim=0)[0], x.min(dim=0)[0]

        xmax, xmin = None, None
        for start in range(0, x.shape[0], self.stream_chunk_size):
            end = min(start + self.stream_chunk_size, x.shape[0])
            cur_x = x[start:end]
            cur_max = cur_x.max(dim=0)[0]
            cur_min = cur_x.min(dim=0)[0]
            xmax = cur_max if xmax is None else torch.maximum(xmax, cur_max)
            xmin = cur_min if xmin is None else torch.minimum(xmin, cur_min)
        return xmax, xmin

    @torch.no_grad()
    def _channel_shift(self, x):
        xmax, xmin = self._channel_min_max(x)
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
            bucket_xmax = None
            bucket_xmin = None
            for x in bucket:
                cur_xmax, cur_xmin = self._channel_min_max(x)
                cur_xmax = cur_xmax.float()
                cur_xmin = cur_xmin.float()
                bucket_xmax = cur_xmax if bucket_xmax is None else torch.maximum(bucket_xmax, cur_xmax)
                bucket_xmin = cur_xmin if bucket_xmin is None else torch.minimum(bucket_xmin, cur_xmin)
            if bucket_xmax is None or bucket_xmin is None:
                raise ValueError('HTG encountered empty timestep bucket when computing z_t.')
            z_t.append((bucket_xmax + bucket_xmin) * 0.5)
        return z_t

    @torch.no_grad()
    def _compute_step_shifts_stream(self, stream_stats):
        step_xmax = stream_stats.get('step_xmax', [])
        step_xmin = stream_stats.get('step_xmin', [])
        shift_counts = stream_stats.get('step_shift_count', [])
        if len(step_xmax) == 0 or len(step_xmin) == 0:
            raise ValueError('HTG stream_stats enabled but timestep extrema are empty.')

        z_t = []
        for xmax, xmin, count in zip(step_xmax, step_xmin, shift_counts):
            if xmax is None or xmin is None or count <= 0:
                raise ValueError('HTG stream_stats encountered empty timestep statistics.')
            z_t.append(((xmax + xmin) * 0.5).float())
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
                cur_max = step_max if cur_max is None else torch.maximum(cur_max, step_max)
            if cur_max is None:
                raise ValueError('HTG encountered empty timestep bucket when computing EMA scale.')
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
        stream_stats,
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
        num_steps: int,
    ):
        step_xmax = stream_stats.get('step_xmax', [])
        step_xmin = stream_stats.get('step_xmin', [])
        shift_counts = stream_stats.get('step_shift_count', [])
        if len(step_xmax) == 0 or len(step_xmin) == 0:
            raise ValueError('HTG stream_stats enabled but timestep extrema are empty.')
        x_abs_max_t = []
        for step_idx in range(num_steps):
            if step_idx >= len(step_xmax) or step_idx >= len(step_xmin):
                raise ValueError('HTG stream_stats second pass encountered empty timestep data.')
            xmax = step_xmax[step_idx]
            xmin = step_xmin[step_idx]
            count = shift_counts[step_idx] if step_idx < len(shift_counts) else 0
            if xmax is None or xmin is None or count <= 0:
                raise ValueError('HTG stream_stats encountered empty timestep statistics.')
            group_idx = step_to_group[step_idx]
            shift = z_g[group_idx].to(dtype=torch.float32, device=xmax.device)
            step_max = torch.maximum(
                (xmax.to(torch.float32) - shift).abs(),
                (xmin.to(torch.float32) - shift).abs(),
            )
            x_abs_max_t.append(step_max.clamp(min=1e-5))

        ema = None
        for t in range(len(x_abs_max_t) - 1, -1, -1):
            cur = x_abs_max_t[t]
            ema = cur if ema is None else (self.alpha * ema + (1.0 - self.alpha) * cur)

        ema = ema.clamp(min=1e-5)
        scale = ema / ema.max().clamp(min=1e-5)
        scale = scale.clamp(min=1e-5)
        return scale

    @torch.no_grad()
    def _collect_hiband_histograms_stream(
        self,
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
        input_name: str,
        num_steps: int,
        scale: torch.Tensor,
        group_num: int = 0,
        collect_global: bool = True,
        collect_grouped: bool = False,
    ):
        if self._current_block is None:
            raise ValueError('HTG stream_stats HiBand requires current block context.')

        stream_state = {'idx': 0}
        scale_cpu = scale.detach().to(torch.float32).cpu()
        global_hist_state = [None] if collect_global else None
        group_hist_states = [None for _ in range(group_num)] if collect_grouped else None
        chunk_size = self.hiband_sample_max_tokens if self.hiband_sample_max_tokens > 0 else 8192

        def collect_hiband_histogram_hook(_, x, _output):
            step_idx = stream_state['idx'] % num_steps
            stream_state['idx'] += 1
            if len(x) == 0 or not torch.is_tensor(x[0]):
                return

            inp = x[0].detach()
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)

            group_idx = step_to_group[step_idx]
            shift = z_g[group_idx].to(device=inp.device, dtype=inp.dtype)
            shape = [1] * (inp.dim() - 1) + [-1]
            post = (inp - shift.view(*shape)) / scale_cpu.to(
                device=inp.device, dtype=inp.dtype
            ).view(*shape)
            post = post.view(-1, post.shape[-1]).to(torch.float32).cpu()

            if collect_global and global_hist_state[0] is None:
                global_hist_state[0] = self._init_hiband_histogram_state(
                    post.shape[-1], device='cpu'
                )
            if collect_grouped and group_hist_states[group_idx] is None:
                group_hist_states[group_idx] = self._init_hiband_histogram_state(
                    post.shape[-1], device='cpu'
                )

            for start in range(0, post.shape[0], chunk_size):
                chunk = post[start:start + chunk_size]
                if collect_global:
                    self._incremental_update_hiband_histogram(global_hist_state[0], chunk)
                if collect_grouped:
                    self._incremental_update_hiband_histogram(group_hist_states[group_idx], chunk)
            del post

        layer = dict(self._current_block.named_modules())[input_name]
        handle = layer.register_forward_hook(collect_hiband_histogram_hook)
        try:
            self.block_forward(self._current_block, collect_output=False)
        finally:
            handle.remove()

        global_result = None
        if collect_global and global_hist_state[0] is not None:
            result = self._finalize_hiband_histogram_state(global_hist_state[0])
            if result[5] != 0:
                global_result = result

        grouped_results = None
        if collect_grouped:
            grouped_results = []
            for state in group_hist_states:
                if state is None:
                    grouped_results.append(None)
                    continue
                result = self._finalize_hiband_histogram_state(state)
                if result[5] == 0:
                    grouped_results.append(None)
                else:
                    grouped_results.append(result)

        return global_result, grouped_results

    @torch.no_grad()
    def _collect_hiband_act_samples_stream(
        self,
        stream_stats,
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
        input_name: str,
        num_steps: int,
        scale: torch.Tensor,
    ):
        if self._current_block is None:
            raise ValueError('HTG stream_stats HiBand requires current block context.')

        samples = []
        stream_state = {'idx': 0}
        scale_cpu = scale.detach().to(torch.float32).cpu()
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

            group_idx = step_to_group[step_idx]
            shift = z_g[group_idx].to(device=inp.device, dtype=inp.dtype)
            shape = [1] * (inp.dim() - 1) + [-1]
            post = (inp - shift.view(*shape)) / scale_cpu.to(device=inp.device, dtype=inp.dtype).view(*shape)
            post = post.view(-1, post.shape[-1]).to(torch.float32).cpu()
            post = self._sample_rows_evenly(post, min(post.shape[0], occurrence_budget))
            if post.numel() > 0:
                samples.append(post)

        layer = dict(self._current_block.named_modules())[input_name]
        handle = layer.register_forward_hook(collect_hiband_post_scale)
        try:
            # Reuse stream second pass and keep collect_output=False.
            self.block_forward(self._current_block, collect_output=False)
        finally:
            handle.remove()

        if len(samples) == 0:
            return None
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def _collect_hiband_group_act_samples_stream(
        self,
        stream_stats,
        z_g: List[torch.Tensor],
        step_to_group: Dict[int, int],
        input_name: str,
        num_steps: int,
        scale: torch.Tensor,
        group_num: int,
    ):
        if self._current_block is None:
            raise ValueError('HTG stream_stats HiBand requires current block context.')

        grouped_samples = [[] for _ in range(group_num)]
        stream_state = {'idx': 0}
        scale_cpu = scale.detach().to(torch.float32).cpu()
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

            group_idx = step_to_group[step_idx]
            shift = z_g[group_idx].to(device=inp.device, dtype=inp.dtype)
            shape = [1] * (inp.dim() - 1) + [-1]
            post = (inp - shift.view(*shape)) / scale_cpu.to(device=inp.device, dtype=inp.dtype).view(*shape)
            post = post.view(-1, post.shape[-1]).to(torch.float32).cpu()
            post = self._sample_rows_evenly(post, min(post.shape[0], occurrence_budget))
            if post.numel() > 0:
                grouped_samples[group_idx].append(post)

        layer = dict(self._current_block.named_modules())[input_name]
        handle = layer.register_forward_hook(collect_hiband_post_scale)
        try:
            self.block_forward(self._current_block, collect_output=False)
        finally:
            handle.remove()

        merged = []
        for group_samples in grouped_samples:
            if len(group_samples) == 0:
                merged.append(None)
            else:
                merged.append(torch.cat(group_samples, dim=0))
        return merged

    @torch.no_grad()
    def _search_hiband_group_act_scales(
        self, grouped_samples, fallback_scale, target_device, target_dtype
    ):
        if grouped_samples is None or len(grouped_samples) == 0:
            return None

        fallback_value = None
        if fallback_scale is not None:
            fallback_value = fallback_scale.to(device=target_device, dtype=target_dtype)

        group_scales = []
        for samples in grouped_samples:
            cur_scale = self._search_hiband_scale(samples, target_device, target_dtype, side='act')
            if cur_scale is None:
                if fallback_value is None:
                    return None
                cur_scale = fallback_value
            group_scales.append(cur_scale)

        return torch.stack(group_scales, dim=0)

    @torch.no_grad()
    def _search_hiband_group_act_scales_from_histograms(
        self, grouped_histograms, fallback_scale, target_device, target_dtype
    ):
        if grouped_histograms is None or len(grouped_histograms) == 0:
            return None

        fallback_value = None
        if fallback_scale is not None:
            fallback_value = fallback_scale.to(device=target_device, dtype=target_dtype)

        group_scales = []
        for hist_result in grouped_histograms:
            if hist_result is None:
                if fallback_value is None:
                    return None
                group_scales.append(fallback_value)
                continue
            offset, hist, overflow_tail_hist, zero_count, min_exp, N_c = hist_result
            cur_scale = self._search_hiband_scale_from_histogram(
                offset, hist, overflow_tail_hist, zero_count, min_exp, N_c,
                target_device, target_dtype, side='act',
            )
            if cur_scale is None:
                if fallback_value is None:
                    return None
                cur_scale = fallback_value
            group_scales.append(cur_scale)

        return torch.stack(group_scales, dim=0)

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
                shift = shift.to(scale.device)
                shift_scaled = (shift / scale).to(device=layer_bias.device, dtype=layer_bias.dtype)
                comp = torch.matmul(layer_weight, shift_scaled)
                shift_scaled_list.append(layer_bias + comp)
            htg_group_bias = torch.stack(shift_scaled_list, dim=0).detach()
            if hasattr(layer, 'htg_group_bias'):
                layer.htg_group_bias.data = htg_group_bias
            else:
                layer.register_buffer('htg_group_bias', htg_group_bias)

    @torch.no_grad()
    def _register_htg_attn_o_buffers(
        self,
        layers: List[nn.Module],
        z_g: List[torch.Tensor],
        scale: torch.Tensor,
        boundaries_norm: torch.Tensor,
    ):
        htg_input_shift = torch.stack(z_g, dim=0).detach()
        scale = scale.detach()

        if hasattr(layers[0], 'htg_input_shift'):
            layers[0].htg_input_shift.data = htg_input_shift.to(
                device=layers[0].weight.device, dtype=layers[0].weight.dtype
            )
        else:
            layers[0].register_buffer(
                'htg_input_shift',
                htg_input_shift.to(device=layers[0].weight.device, dtype=layers[0].weight.dtype),
            )

        if hasattr(layers[0], 'htg_input_scale'):
            layers[0].htg_input_scale.data = scale.to(
                device=layers[0].weight.device, dtype=layers[0].weight.dtype
            )
        else:
            layers[0].register_buffer(
                'htg_input_scale',
                scale.to(device=layers[0].weight.device, dtype=layers[0].weight.dtype),
            )

        if hasattr(layers[0], 'htg_group_boundaries'):
            layers[0].htg_group_boundaries.data = boundaries_norm.to(
                device=layers[0].weight.device, dtype=layers[0].weight.dtype
            )
        else:
            layers[0].register_buffer(
                'htg_group_boundaries',
                boundaries_norm.to(device=layers[0].weight.device, dtype=layers[0].weight.dtype),
            )

        for layer in layers:
            if not hasattr(layer, 'weight'):
                continue
            layer_weight = layer.weight.detach()
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer_bias = layer.bias.detach()
            else:
                layer_bias = torch.zeros(
                    layer_weight.shape[0],
                    device=layer_weight.device,
                    dtype=layer_weight.dtype,
                )

            shift_scaled_list = []
            for shift in z_g:
                shift = shift.to(scale.device)
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
        if not isinstance(feat_dict[name], dict):
            feat_dict[name] = {}

        num_steps = self._resolve_num_steps_stream()
        if 'step_xmax' not in feat_dict[name]:
            feat_dict[name]['step_xmax'] = [None for _ in range(num_steps)]
            feat_dict[name]['step_xmin'] = [None for _ in range(num_steps)]
            feat_dict[name]['step_shift_count'] = [0 for _ in range(num_steps)]
            feat_dict[name]['step_token_rows'] = [[] for _ in range(num_steps)]
            feat_dict[name]['step_cursor'] = 0

        step_idx = feat_dict[name]['step_cursor'] % num_steps
        xmax = xmax.to(torch.float32).cpu()
        xmin = xmin.to(torch.float32).cpu()
        if feat_dict[name]['step_xmax'][step_idx] is None:
            feat_dict[name]['step_xmax'][step_idx] = xmax
            feat_dict[name]['step_xmin'][step_idx] = xmin
        else:
            feat_dict[name]['step_xmax'][step_idx] = torch.maximum(
                feat_dict[name]['step_xmax'][step_idx], xmax
            )
            feat_dict[name]['step_xmin'][step_idx] = torch.minimum(
                feat_dict[name]['step_xmin'][step_idx], xmin
            )
        feat_dict[name]['step_shift_count'][step_idx] += 1
        feat_dict[name]['step_token_rows'][step_idx].append(int(flat.shape[0]))
        feat_dict[name]['step_cursor'] += 1

    @torch.no_grad()
    def subset_transform(self, subset, input_feat, subset_kwargs):
        del subset_kwargs
        layers_dict = subset['layers']
        prev_op = subset['prev_op']
        input_name = subset['input'][0]
        is_attn_o = self._is_attn_o_subset(subset)

        if not is_attn_o and not self.filter_subset(prev_op):
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
            scale = self._compute_ema_scale_stream(tensors, z_g, step_to_group, num_steps)
        else:
            scale = self._compute_ema_scale(step_buckets, z_g, step_to_group)

        if is_attn_o:
            scale = scale.to(dtype=layers[0].weight.dtype, device=layers[0].weight.device)
            hiband_act_scale = None
            hiband_group_act_scales = None
            if self.hiband_enabled:
                need_group_histograms = (
                    self.hiband_runtime_group_act_scale_enabled
                    and self.hiband_group_source == 'htg'
                )
                if self.hiband_use_true_qdq_mse:
                    if self.stream_stats and isinstance(tensors, dict):
                        act_samples = self._collect_hiband_act_samples_stream(
                            tensors, z_g, step_to_group, input_name, num_steps, scale
                        )
                    else:
                        act_samples = self._collect_hiband_act_samples(
                            tensors, num_steps, step_to_group, z_g, scale
                        )
                    hiband_act_scale = self._search_hiband_scale(
                        act_samples, scale.device, scale.dtype, side='act'
                    )
                else:
                    grouped_histograms = None
                    if self.stream_stats and isinstance(tensors, dict):
                        hist_result, grouped_histograms = self._collect_hiband_histograms_stream(
                            z_g,
                            step_to_group,
                            input_name,
                            num_steps,
                            scale,
                            group_num=group_num,
                            collect_global=True,
                            collect_grouped=need_group_histograms,
                        )
                    else:
                        hist_result = self._collect_hiband_histogram(
                            tensors, num_steps, step_to_group, z_g, scale
                        )
                    if hist_result is not None:
                        offset, hist, overflow_tail_hist, zero_count, min_exp, N_c = hist_result
                        hiband_act_scale = self._search_hiband_scale_from_histogram(
                            offset, hist, overflow_tail_hist, zero_count, min_exp, N_c,
                            scale.device, scale.dtype, side='act',
                        )
                if hiband_act_scale is not None:
                    self._attach_hiband_act_scale(layers, hiband_act_scale)
                if (
                    need_group_histograms
                    and hiband_act_scale is not None
                ):
                    if self.hiband_use_true_qdq_mse:
                        if self.stream_stats and isinstance(tensors, dict):
                            grouped_act_samples = self._collect_hiband_group_act_samples_stream(
                                tensors, z_g, step_to_group, input_name, num_steps, scale, group_num
                            )
                        else:
                            grouped_act_samples = self._collect_hiband_group_act_samples(
                                tensors, num_steps, step_to_group, z_g, scale, group_num
                            )
                        hiband_group_act_scales = self._search_hiband_group_act_scales(
                            grouped_act_samples, hiband_act_scale, scale.device, scale.dtype
                        )
                    else:
                        if not (self.stream_stats and isinstance(tensors, dict)):
                            grouped_histograms = self._collect_hiband_group_histograms(
                                tensors, num_steps, step_to_group, z_g, scale, group_num
                            )
                        hiband_group_act_scales = self._search_hiband_group_act_scales_from_histograms(
                            grouped_histograms, hiband_act_scale, scale.device, scale.dtype
                        )
                    if hiband_group_act_scales is not None:
                        self._attach_hiband_group_act_scales(layers, hiband_group_act_scales)
            if self.enable_dynamic_runtime:
                for layer in layers:
                    layer.weight.mul_(scale.view(1, -1))
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.data = layer.bias.data.clone()
                self._register_htg_attn_o_buffers(layers, z_g, scale, boundaries_norm)
                shift = self._collapse_group_shifts(z_g, groups).to(dtype=scale.dtype, device=scale.device)
            else:
                shift = self._collapse_group_shifts(z_g, groups).to(dtype=scale.dtype, device=scale.device)
                for layer in layers:
                    layer.weight.mul_(scale.view(1, -1))
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.data = layer.bias.data.clone()
                self._register_htg_attn_o_buffers(layers, z_g, scale, boundaries_norm)

            if self.act_static:
                self.update_input_feat(scale, input_feat, layers_dict, False)

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
            if self.hiband_enabled and hiband_act_scale is not None:
                self.hiband_act_scales[key] = hiband_act_scale.detach().cpu()
            if self.hiband_enabled and hiband_group_act_scales is not None:
                self.hiband_group_act_scales[key] = (
                    hiband_group_act_scales.detach().cpu()
                )
            return

        scale = scale.to(dtype=prev_op[0].weight.dtype, device=prev_op[0].weight.device)
        hiband_act_scale = None
        hiband_group_act_scales = None
        hiband_weight_scales = {}
        if self.hiband_enabled:
            need_group_histograms = (
                self.hiband_runtime_group_act_scale_enabled
                and self.hiband_group_source == 'htg'
            )
            needs_act_samples = (
                self.hiband_use_true_qdq_mse
                or self.hiband_weight_scale_enabled
            )
            if needs_act_samples:
                if self.stream_stats and isinstance(tensors, dict):
                    act_samples = self._collect_hiband_act_samples_stream(
                        tensors, z_g, step_to_group, input_name, num_steps, scale
                    )
                else:
                    act_samples = self._collect_hiband_act_samples(
                        tensors, num_steps, step_to_group, z_g, scale
                    )
                hiband_act_scale = self._search_hiband_scale(
                    act_samples, scale.device, scale.dtype, side='act'
                )
            else:
                act_samples = None
                grouped_histograms = None
                if self.stream_stats and isinstance(tensors, dict):
                    hist_result, grouped_histograms = self._collect_hiband_histograms_stream(
                        z_g,
                        step_to_group,
                        input_name,
                        num_steps,
                        scale,
                        group_num=group_num,
                        collect_global=True,
                        collect_grouped=need_group_histograms,
                    )
                else:
                    hist_result = self._collect_hiband_histogram(
                        tensors, num_steps, step_to_group, z_g, scale
                    )
                if hist_result is not None:
                    offset, hist, overflow_tail_hist, zero_count, min_exp, N_c = hist_result
                    hiband_act_scale = self._search_hiband_scale_from_histogram(
                        offset, hist, overflow_tail_hist, zero_count, min_exp, N_c,
                        scale.device, scale.dtype, side='act',
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
            if (
                need_group_histograms
            ):
                if self.hiband_use_true_qdq_mse:
                    if self.stream_stats and isinstance(tensors, dict):
                        grouped_act_samples = self._collect_hiband_group_act_samples_stream(
                            tensors, z_g, step_to_group, input_name, num_steps, scale, group_num
                        )
                    else:
                        grouped_act_samples = self._collect_hiband_group_act_samples(
                            tensors, num_steps, step_to_group, z_g, scale, group_num
                        )
                    hiband_group_act_scales = self._search_hiband_group_act_scales(
                        grouped_act_samples, hiband_act_scale, scale.device, scale.dtype
                    )
                else:
                    if not (self.stream_stats and isinstance(tensors, dict)):
                        grouped_histograms = self._collect_hiband_group_histograms(
                            tensors, num_steps, step_to_group, z_g, scale, group_num
                        )
                    hiband_group_act_scales = self._search_hiband_group_act_scales_from_histograms(
                        grouped_histograms, hiband_act_scale, scale.device, scale.dtype
                    )
                if hiband_group_act_scales is not None:
                    self._attach_hiband_group_act_scales(layers, hiband_group_act_scales)

        if self.enable_dynamic_runtime:
            # For runtime switching, keep base bias untouched and only absorb scaling.
            self.apply_scale(scale, prev_op, layers)
            for layer, layer_scale in hiband_weight_scales.items():
                self._apply_hiband_weight_scale(layer, layer_scale)
            # Runtime bias compensation must see the final exported weights.
            self._register_htg_runtime_buffers(prev_op[0], layers, z_g, scale, boundaries_norm)
            shift = self._collapse_group_shifts(z_g, groups).to(dtype=scale.dtype, device=scale.device)
        else:
            # Static fallback follows Eq.(9) equivalent re-parameterization: shift first, then scale.
            shift = self._collapse_group_shifts(z_g, groups).to(dtype=scale.dtype, device=scale.device)
            self.apply_shift(shift, prev_op, layers)
            self.apply_scale(scale, prev_op, layers)
            for layer, layer_scale in hiband_weight_scales.items():
                self._apply_hiband_weight_scale(layer, layer_scale)

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
        if self.hiband_enabled and hiband_act_scale is not None:
            self.hiband_act_scales[key] = hiband_act_scale.detach().cpu()
        if self.hiband_enabled and hiband_group_act_scales is not None:
            self.hiband_group_act_scales[key] = hiband_group_act_scales.detach().cpu()
