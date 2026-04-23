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


def update_lightx2v_quant_config(save_quant_path, config=None):

    config_file = save_quant_path + '/config.json'
    with open(config_file, 'r') as file:
        config_lightx2v = json.load(file)
    config_lightx2v['quant_method'] = 'advanced_ptq'
    if config is not None and 'quant' in config and 'video_gen' in config.quant:
        method = config.quant.video_gen.get('method', '')
        config_lightx2v['htg_enabled'] = str(method).lower() == 'htg'
    if _has_hif8_quant(config):
        config_lightx2v['dit_quantized'] = True
        config_lightx2v['dit_quant_scheme'] = 'hif8-cuda'
        config_lightx2v['hif8_format_version'] = 2
        # Phase-3 interface metadata for native HiF8 bit packing.
        config_lightx2v['hif8_storage_dtype'] = 'uint8_hif8'
        config_lightx2v['hif8_weight_scale_granularity'] = 'none'
        config_lightx2v['hif8_runtime'] = {
            'enable_input_qdq': True,
            'enable_output_requant': False,
            'compute_dtype': 'bf16'
        }
    with open(config_file, 'w') as file:
        json.dump(config_lightx2v, file, indent=4)
