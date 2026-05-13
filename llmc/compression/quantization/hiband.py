import torch
from loguru import logger

from llmc.utils.registry_factory import ALGO_REGISTRY

from .smoothquant import SmoothQuant


def _ensure_hiband_special_config(quant_config, model_type):
    special_config = quant_config.get('special', {})
    if not isinstance(special_config, dict):
        special_config = dict(special_config)

    hiband_cfg = special_config.get('hiband', {})
    if not isinstance(hiband_cfg, dict):
        hiband_cfg = dict(hiband_cfg)

    hiband_cfg.setdefault('enabled', True)
    hiband_cfg.setdefault('act_scale_enabled', True)
    # Standalone HiBand defaults to base fake-quant weights unless explicitly enabled.
    hiband_cfg.setdefault('weight_scale_enabled', False)

    special_config['hiband'] = hiband_cfg
    quant_config['special'] = special_config
    return quant_config


@ALGO_REGISTRY
class HiBand(SmoothQuant):
    def __init__(self, model, quant_config, input, padding_mask, config):
        quant_config = _ensure_hiband_special_config(quant_config, config.model.type)
        super().__init__(model, quant_config, input, padding_mask, config)

    @torch.no_grad()
    def block_opt(self, block, *opt_kwargs):
        if self.input is None and not self.act_static:
            if self.block_idx == 0:
                logger.warning(
                    'Standalone HiBand requires calibration inputs to search '
                    'hiband_act_scale. No calib data detected, fall back to base fake-quant.'
                )
            if self.quant_kvcache:
                self.register_kv_cache(block)
            return
        super().block_opt(block, *opt_kwargs)

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
        is_attn_o = self._is_attn_o_subset(subset)
        hiband_act_only = self._is_hiband_act_only_subset(subset)
        supports_hiband = is_attn_o or hiband_act_only or self.filter_subset(prev_op)
        if not supports_hiband:
            logger.info('Do not transform this subset.')
            return

        layers = list(layers_dict.values())
        input_channels = self._get_hiband_input_channels(layers[0])
        if input_channels is None:
            logger.info('Skip HiBand for subset without weight-bearing layer.')
            return
        base_scale = torch.ones(
            input_channels,
            device=layers[0].weight.device,
            dtype=layers[0].weight.dtype,
        )
        channel_axis = self._get_channel_axis(layers[0])

        hiband_act_scale = None
        hiband_weight_scales = {}
        skip_weight_hiband = is_attn_o or hiband_act_only

        if self.hiband_enabled:
            needs_act_samples = (
                self.hiband_act_scale_enabled
                and (
                    self.hiband_use_true_qdq_mse
                    or (not skip_weight_hiband and self.hiband_weight_scale_enabled)
                )
            )
            if needs_act_samples:
                if self.stream_stats and isinstance(input_feat[input_name], dict):
                    act_samples = self._collect_hiband_act_samples_stream(
                        input_feat[input_name],
                        input_name,
                        base_scale,
                        channel_axis=channel_axis,
                    )
                else:
                    act_samples = self._collect_hiband_act_samples(
                        input_feat[input_name],
                        base_scale,
                        channel_axis=channel_axis,
                    )
                hiband_act_scale = self._search_hiband_scale(
                    act_samples, base_scale.device, base_scale.dtype, side='act'
                )
            elif self.hiband_act_scale_enabled:
                act_samples = None
                if self.stream_stats and isinstance(input_feat[input_name], dict):
                    hist_result = self._collect_hiband_histogram_stream(
                        input_feat[input_name],
                        input_name,
                        base_scale,
                        channel_axis=channel_axis,
                    )
                else:
                    hist_result = self._collect_hiband_histogram(
                        input_feat[input_name],
                        base_scale,
                        channel_axis=channel_axis,
                    )
                if hist_result is not None:
                    offset, hist, overflow_tail_hist, zero_count, min_exp, N_c = hist_result
                    hiband_act_scale = self._search_hiband_scale_from_histogram(
                        offset,
                        hist,
                        overflow_tail_hist,
                        zero_count,
                        min_exp,
                        N_c,
                        base_scale.device,
                        base_scale.dtype,
                        side='act',
                    )
            else:
                act_samples = None

            if (
                not skip_weight_hiband
                and self.hiband_weight_scale_enabled
                and hiband_act_scale is not None
            ):
                hiband_weight_scales = self._search_hiband_weight_scales(
                    layers,
                    base_scale,
                    act_samples,
                    hiband_act_scale,
                    base_scale.device,
                    base_scale.dtype,
                )
            if self.hiband_act_scale_enabled and hiband_act_scale is not None:
                self._attach_hiband_act_scale(layers, hiband_act_scale)
                key = ','.join(sorted(layers_dict.keys()))
                self.hiband_act_scales[key] = hiband_act_scale.detach().cpu()

        for layer, layer_scale in hiband_weight_scales.items():
            self._apply_hiband_weight_scale(layer, layer_scale)
