from PIL import Image
from transformers import (AutoConfig, AutoImageProcessor,
                          AutoModelForImageClassification)

from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class ResNet(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageClassification.from_pretrained(
            self.model_path,
            config=self.model_config,
            device_map=self.device_map,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        )

    def find_blocks(self, modality='vision'):
        del modality
        blocks = [self.model.resnet.embedder]
        for stage in self.model.resnet.encoder.stages:
            blocks.extend(list(stage.layers))
        self.blocks = blocks

    def replace_first_block_with_catcher(self, catcher_module):
        original_module = self.model.resnet.embedder
        self.model.resnet.embedder = catcher_module
        self.blocks[0] = catcher_module
        return original_module

    def restore_first_block_from_catcher(self, original_module):
        self.model.resnet.embedder = original_module
        self.blocks[0] = original_module

    def find_embed_layers(self):
        self.embed_tokens = []

    def get_embed_layers(self):
        return self.embed_tokens

    def get_head_layers(self):
        return [self.model.classifier[1]]

    def get_pre_head_layernorm_layers(self):
        return []

    def get_layers_except_blocks(self):
        return [self.model.resnet.pooler, self.model.classifier]

    def skip_layer_name(self):
        return ['classifier']

    def has_bias(self):
        return True

    def get_layernorms_in_block(self, block, modality='vision'):
        del modality
        if hasattr(block, 'embedder'):
            return {'embedder.normalization': block.embedder.normalization}

        lns = {}
        for idx, layer in enumerate(block.layer):
            lns[f'layer.{idx}.normalization'] = layer.normalization
        if hasattr(block, 'shortcut') and hasattr(block.shortcut, 'normalization'):
            lns['shortcut.normalization'] = block.shortcut.normalization
        return lns

    def batch_process(
        self, imgs, calib_or_eval='eval', apply_chat_template=False, return_inputs=True
    ):
        del return_inputs
        assert calib_or_eval == 'calib' or calib_or_eval == 'eval'
        assert not apply_chat_template
        img_data_list = []
        for img in imgs:
            path = img['image']
            img_data = Image.open(path).convert('RGB')
            img_data_list.append(img_data)
        inputs = self.processor(images=img_data_list, return_tensors='pt')
        return inputs

    def get_subsets_in_block(self, block):
        if hasattr(block, 'embedder'):
            return []

        subsets = []
        layer_count = len(block.layer)
        for idx in range(1, layer_count):
            prev_norm = block.layer[idx - 1].normalization
            conv = block.layer[idx].convolution
            subsets.append(
                {
                    'layers': {f'layer.{idx}.convolution': conv},
                    'prev_op': [prev_norm],
                    'input': [f'layer.{idx}.convolution'],
                    'inspect': conv,
                    'has_kwargs': False,
                }
            )
        return subsets
