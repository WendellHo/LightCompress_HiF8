import inspect
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanPipeline
from loguru import logger
from PIL import Image

from llmc.compression.quantization.module_utils import LlmcWanTransformerBlock
from llmc.utils import seed_all
from llmc.utils.registry_factory import MODEL_REGISTRY

from .base_model import BaseModel


@MODEL_REGISTRY
class WanT2V(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        if 'calib' in config:
            self.calib_bs = config.calib.bs
            self.sample_steps = config.calib.sample_steps
            self.calib_inference_per_block = config.calib.get('inference_per_block', False)
            self.calib_distributed_merge = config.calib.get('distributed_merge', False)
            self.calib_seed = int(config.calib.get('seed', config.base.seed))
            self.target_height = config.calib.get('target_height', 480)
            self.target_width = config.calib.get('target_width', 832)
            self.num_frames = config.calib.get('num_frames', 81)
            self.guidance_scale = config.calib.get('guidance_scale', 5.0)
        else:
            self.sample_steps = None
            self.calib_inference_per_block = False
            self.calib_distributed_merge = False
            self.calib_seed = int(config.base.seed)
        self._pipeline_supports_generator = None

    def build_model(self):
        vae = AutoencoderKLWan.from_pretrained(
            self.model_path, subfolder='vae', torch_dtype=torch.float32
        )
        self.Pipeline = WanPipeline.from_pretrained(
            self.model_path, vae=vae, torch_dtype=torch.bfloat16
        )
        self.find_llmc_model()
        self.find_blocks()
        for block_idx, block in enumerate(self.blocks):
            new_block = LlmcWanTransformerBlock.new(block)
            new_block.block_idx = block_idx  # Keep track of index if needed
            new_block._llmc_transformer_ref = [self.model]
            self.Pipeline.transformer.blocks[block_idx] = new_block
        logger.info(f'self.model : {self.model}')
        
        # Monkey patch transformer to track timestep progress for HTG evaluation
        original_forward = self.model.forward
        self.model._llmc_step_index = 0
        self.model._llmc_infer_steps = getattr(self, 'sample_steps', None)
        if self.model._llmc_infer_steps is None:
            self.model._llmc_infer_steps = 50

        
        def patched_forward(*args, **kwargs):
            # Hack to get the current progress from scheduler if available
            if hasattr(self.Pipeline, 'scheduler'):
                if hasattr(self.Pipeline.scheduler, 'step_index') and self.Pipeline.scheduler.step_index is not None:
                    self.model._llmc_step_index = self.Pipeline.scheduler.step_index
                elif hasattr(self.Pipeline.scheduler, '_step_index') and self.Pipeline.scheduler._step_index is not None:
                    self.model._llmc_step_index = self.Pipeline.scheduler._step_index
                
                # Try to get infer_steps, it's usually set in pipeline during __call__
                # But pipeline sets `num_inference_steps` locally.
                if hasattr(self.Pipeline, '_num_inference_steps') and self.Pipeline._num_inference_steps is not None:
                    self.model._llmc_infer_steps = self.Pipeline._num_inference_steps
            
            res = original_forward(*args, **kwargs)
            
            # If scheduler doesn't expose step_index, fallback to manual increment
            if not hasattr(self.Pipeline, 'scheduler') or (not hasattr(self.Pipeline.scheduler, 'step_index') and not hasattr(self.Pipeline.scheduler, '_step_index')):
                if self.model._llmc_step_index is None:
                    self.model._llmc_step_index = 0
                self.model._llmc_step_index += 1
                if self.model._llmc_infer_steps is not None and self.model._llmc_step_index >= self.model._llmc_infer_steps:
                    self.model._llmc_step_index = 0  # reset for next video
            
            return res
            
        self.model.forward = patched_forward

    def find_llmc_model(self):
        self.model = self.Pipeline.transformer

    def find_blocks(self):
        self.blocks = self.model.blocks

    def get_catcher(self, first_block_input):
        sample_steps = self.sample_steps
        to_cpu_serializable = self._to_cpu_serializable

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)
                self.step = 0

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                # Offload captured calib inputs to CPU immediately to avoid
                # accumulating per-step activations on GPU across prompts.
                first_block_input['data'].append(to_cpu_serializable(args[0]))
                first_block_input['kwargs'].append(to_cpu_serializable(kwargs))
                self.step += 1
                if self.step == sample_steps:
                    raise ValueError
                else:
                    return self.module(*args)

        return Catcher

    @torch.no_grad()
    def _calib_forward_pre_hook(self, m, x):
        m.cuda()

    @torch.no_grad()
    def _calib_forward_hook(self, m, x, y):
        m.cpu()

    @torch.no_grad()
    def _register_calib_video_hooks(self):
        handles = []
        for layer in self.blocks:
            handles.append(layer.register_forward_pre_hook(self._calib_forward_pre_hook))
        for layer in self.blocks:
            handles.append(layer.register_forward_hook(self._calib_forward_hook))
        return handles

    @torch.no_grad()
    def _move_video_blocks(self, device):
        for layer in self.blocks:
            layer.to(device)

    @torch.no_grad()
    def _move_pipeline_non_block_modules(self, device):
        # Avoid transient full-model peak by only moving non-transformer modules.
        pipeline_components = getattr(self.Pipeline, 'components', {})
        if isinstance(pipeline_components, dict):
            for name, module in pipeline_components.items():
                if not isinstance(module, nn.Module):
                    continue
                if name == 'transformer':
                    continue
                module.to(device)
        # Keep transformer blocks offloaded; only keep required non-block parts on device.
        for name, module in self.model.named_children():
            if name == 'blocks':
                continue
            module.to(device)

    def _is_dist_collect_enabled(self):
        if not self.calib_distributed_merge:
            return False
        if not dist.is_available() or not dist.is_initialized():
            return False
        return dist.get_world_size() > 1

    def _get_global_sample_index(self, local_idx):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return dist.get_rank() + local_idx * dist.get_world_size()
        return local_idx

    def _supports_pipeline_generator(self):
        if self._pipeline_supports_generator is None:
            self._pipeline_supports_generator = (
                'generator' in inspect.signature(self.Pipeline.__call__).parameters
            )
        return self._pipeline_supports_generator

    def _to_cpu_serializable(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._to_cpu_serializable(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self._to_cpu_serializable(v) for v in obj)
        if isinstance(obj, list):
            return [self._to_cpu_serializable(v) for v in obj]
        return obj

    def _merge_distributed_catcher_output(self, first_block_input, per_prompt_capture_counts):
        world_size = dist.get_world_size()
        payload = {
            'data': [x.detach().cpu() for x in first_block_input['data']],
            'kwargs': [self._to_cpu_serializable(k) for k in first_block_input['kwargs']],
            'counts': per_prompt_capture_counts,
        }
        gathered_payloads = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_payloads, payload)

        per_rank_chunks = []
        for rank_payload in gathered_payloads:
            rank_data = rank_payload['data']
            rank_kwargs = rank_payload['kwargs']
            rank_counts = rank_payload['counts']
            chunks = []
            cursor = 0
            for cnt in rank_counts:
                next_cursor = cursor + cnt
                chunks.append(
                    (
                        rank_data[cursor:next_cursor],
                        rank_kwargs[cursor:next_cursor],
                    )
                )
                cursor = next_cursor
            per_rank_chunks.append(chunks)

        merged = defaultdict(list)
        max_prompt_num = max(len(chunks) for chunks in per_rank_chunks) if per_rank_chunks else 0
        # BaseDataset shards prompts by rank stride: samples[rank::world_size].
        # Interleaving prompt chunks restores the same global prompt order as single-GPU.
        for prompt_idx in range(max_prompt_num):
            for rank in range(world_size):
                chunks = per_rank_chunks[rank]
                if prompt_idx >= len(chunks):
                    continue
                data_chunk, kwargs_chunk = chunks[prompt_idx]
                merged['data'].extend(data_chunk)
                merged['kwargs'].extend(kwargs_chunk)
        return merged

    @torch.no_grad()
    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = defaultdict(list)
        per_prompt_capture_counts = []
        Catcher = self.get_catcher(first_block_input)
        self.blocks[0] = Catcher(self.blocks[0])
        handles = []
        if self.calib_inference_per_block:
            self.Pipeline.to('cpu')
            self._move_pipeline_non_block_modules('cuda')
            handles = self._register_calib_video_hooks()
            self._move_video_blocks('cpu')
            logger.info('video_gen calib: enabled inference_per_block for transformer blocks.')
        else:
            self.Pipeline.to('cuda')
        for local_idx, data in enumerate(calib_data):
            self.blocks[0].step = 0
            before_num = len(first_block_input['data'])
            try:
                pipe_kw = {
                    'prompt': data['prompt'],
                    'negative_prompt': data['negative_prompt'],
                    'height': self.target_height,
                    'width': self.target_width,
                    'num_frames': self.num_frames,
                    'guidance_scale': self.guidance_scale,
                }
                if self._is_dist_collect_enabled() and self._supports_pipeline_generator():
                    global_idx = self._get_global_sample_index(local_idx)
                    generator = torch.Generator(device='cuda')
                    generator.manual_seed(self.calib_seed + int(global_idx))
                    pipe_kw['generator'] = generator
                self.Pipeline(
                    **pipe_kw
                )
            except ValueError:
                pass
            after_num = len(first_block_input['data'])
            per_prompt_capture_counts.append(after_num - before_num)
        for h in handles:
            h.remove()

        if self._is_dist_collect_enabled():
            first_block_input = self._merge_distributed_catcher_output(
                first_block_input, per_prompt_capture_counts
            )
            logger.info(
                'video_gen calib: enabled distributed_merge, merged calibration catcher inputs across ranks.'
            )

        self.first_block_input = first_block_input
        assert len(self.first_block_input['data']) > 0, 'Catch input data failed.'
        self.n_samples = len(self.first_block_input['data'])
        logger.info(f'Retrieved {self.n_samples} calibration samples for T2V.')
        self.blocks[0] = self.blocks[0].module
        self.Pipeline.to('cpu')

    def get_padding_mask(self):
        return None

    def has_bias(self):
        return True

    def __str__(self):
        return f'\nModel: \n{str(self.model)}'

    def get_layernorms_in_block(self, block):
        return {
            'affine_norm1': block.affine_norm1,
            'norm2': block.norm2,
            'affine_norm3': block.affine_norm3,
        }

    def get_subsets_in_block(self, block):
        return [
            {
                'layers': {
                    'attn1.to_q': block.attn1.to_q,
                    'attn1.to_k': block.attn1.to_k,
                    'attn1.to_v': block.attn1.to_v,
                },
                'prev_op': [block.affine_norm1],
                'input': ['attn1.to_q'],
                'inspect': block.attn1,
                'has_kwargs': True,
                'sub_keys': {'rotary_emb': 'rotary_emb'},
            },
            {
                'layers': {
                    'attn2.to_q': block.attn2.to_q,
                },
                'prev_op': [block.norm2],
                'input': ['attn2.to_q'],
                'inspect': block.attn2,
                'has_kwargs': True,
                'sub_keys': {'encoder_hidden_states': 'encoder_hidden_states'},
            },
            {
                'layers': {
                    'ffn.net.0.proj': block.ffn.net[0].proj,
                },
                'prev_op': [block.affine_norm3],
                'input': ['ffn.net.0.proj'],
                'inspect': block.ffn,
                'has_kwargs': True,
            },
        ]

    def find_block_name(self):
        # Wan transformer blocks live under `self.model.blocks`.
        self.block_name_prefix = 'blocks'

    def find_embed_layers(self):
        pass

    def get_embed_layers(self):
        pass

    def get_layers_except_blocks(self):
        pass

    def skip_layer_name(self):
        pass
