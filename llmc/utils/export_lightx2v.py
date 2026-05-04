import json


def _has_hif8_quant(config):
    if config is None or 'quant' not in config:
        return False

    for _, modality_config in config.quant.items():
        if not isinstance(modality_config, dict) or not modality_config.get('weight'):
            continue
        weight_quant_type = modality_config.weight.get('quant_type', 'int-quant')
        if weight_quant_type in ['hif8-quant', 'hif8']:
            return True
    return False


def update_lightx2v_quant_config(save_quant_path, config=None, hif8_export_mode='native_u8'):
    def _cfg_get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _get_hiband_special_cfg(cfg):
        quant_cfg = _cfg_get(cfg, 'quant', None)
        video_cfg = _cfg_get(quant_cfg, 'video_gen', None)
        special_cfg = _cfg_get(video_cfg, 'special', {}) or {}
        if not isinstance(special_cfg, dict):
            special_cfg = {}
        hiband_cfg = special_cfg.get('hiband', {}) or {}
        return hiband_cfg if isinstance(hiband_cfg, dict) else {}

    config_file = save_quant_path + '/config.json'
    with open(config_file, 'r') as file:
        config_lightx2v = json.load(file)
    config_lightx2v['quant_method'] = 'advanced_ptq'
    if config is not None and 'quant' in config and 'video_gen' in config.quant:
        method = config.quant.video_gen.get('method', '')
        config_lightx2v['htg_enabled'] = str(method).lower() == 'htg'
    if _has_hif8_quant(config):
        hiband_cfg = _get_hiband_special_cfg(config)
        hiband_enabled = bool(hiband_cfg.get('enabled', False))
        config_lightx2v['dit_quantized'] = True
        use_fake_bf16 = str(hif8_export_mode).lower() == 'fake_bf16'
        config_lightx2v['dit_quant_scheme'] = (
            'hif8-fake-bf16' if use_fake_bf16 else 'hif8-cuda'
        )
        config_lightx2v['hif8_format_version'] = 2
        # Keep runtime metadata explicit so LightX2V can select the proper
        # weight container path without affecting existing native-u8 exports.
        config_lightx2v['hif8_storage_dtype'] = (
            'bf16_hif8_fake' if use_fake_bf16 else 'uint8_hif8'
        )
        config_lightx2v['hif8_weight_scale_granularity'] = 'none'
        hif8_runtime = {
            'enable_input_qdq': True,
            'enable_output_requant': False,
            'compute_dtype': 'bf16',
        }
        if hiband_enabled:
            hif8_runtime['hiband_enabled'] = True
            hif8_runtime['hiband_apply_mode'] = 'pre_qdq'
            hif8_runtime['hiband_runtime_group_act_scale_enabled'] = bool(
                hiband_cfg.get('runtime_group_act_scale_enabled', False)
            )
            hif8_runtime['hiband_group_source'] = str(
                hiband_cfg.get('group_source', 'htg')
            ).lower()
            hif8_runtime['hiband_fallback_global_act_scale'] = bool(
                hiband_cfg.get('export_global_act_scale', True)
            )
        config_lightx2v['hif8_runtime'] = hif8_runtime
    with open(config_file, 'w') as file:
        json.dump(config_lightx2v, file, indent=4)
