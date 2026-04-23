import gc
import inspect
import os

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import export_to_video, load_image
from loguru import logger

from llmc.utils import seed_all
from llmc.utils.registry_factory import MODEL_REGISTRY

from .eval_base import BaseEval


class VideoGenerateEval(BaseEval):

    def __init__(self, model, config):
        super().__init__(model, config)
        self.output_video_path = self.eval_cfg.get('output_video_path', None)
        assert self.output_video_path is not None
        os.makedirs(self.output_video_path, exist_ok=True)
        self.target_height = self.eval_cfg.get('target_height', 480)
        self.target_width = self.eval_cfg.get('target_width', 832)
        self.num_frames = self.eval_cfg.get('num_frames', 81)
        self.guidance_scale = self.eval_cfg.get('guidance_scale', 5.0)
        self.guidance_scale_2 = self.eval_cfg.get('guidance_scale_2', None)
        self.fps = self.eval_cfg.get('fps', 15)
        self.distributed = self.eval_cfg.get('distributed', False)
        self._pipeline_supports_generator = None

    def _is_dist_enabled(self):
        if not self.distributed:
            return False
        if not dist.is_available() or not dist.is_initialized():
            return False
        return dist.get_world_size() > 1

    def _iter_with_global_index(self, data):
        if not self._is_dist_enabled():
            for idx, sample in enumerate(data):
                yield idx, sample
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for local_idx, sample in enumerate(data[rank::world_size]):
            yield rank + local_idx * world_size, sample

    def _supports_pipeline_generator(self, model):
        if self._pipeline_supports_generator is None:
            self._pipeline_supports_generator = (
                'generator' in inspect.signature(model.Pipeline.__call__).parameters
            )
        return self._pipeline_supports_generator

    @torch.no_grad()
    def forward_pre_hook(self, m, x):
        m.cuda()

    @torch.no_grad()
    def forward_hook(self, m, x, y):
        # Use synchronous offload to avoid temporary multi-block GPU residency.
        m.cpu()

    @torch.no_grad()
    def move_pipeline_non_block_modules(self, model, device):
        # Avoid transient full-model peak: do not call Pipeline.to('cuda') in blockwise mode.
        pipeline_components = getattr(model.Pipeline, 'components', {})
        if isinstance(pipeline_components, dict):
            for name, module in pipeline_components.items():
                if not isinstance(module, torch.nn.Module):
                    continue
                if name == 'transformer':
                    continue
                module.to(device)
        for name, module in model.model.named_children():
            if name == 'blocks':
                continue
            module.to(device)

    @torch.no_grad()
    def eval(self, model_llmc, eval_pos):
        if self._is_dist_enabled():
            seed_all(int(self.config.base.seed))
        else:
            seed_all(self.config.base.seed + int(os.environ['RANK']))
        handles = []
        if self.inference_per_block:
            # Keep non-block modules on GPU for pipeline execution, while transformer
            # blocks are loaded on-demand via hooks.
            model_llmc.Pipeline.to('cpu')
            self.move_pipeline_non_block_modules(model_llmc, 'cuda')
            handles = self.register_video_hooks(model_llmc)
            self.move_video_blocks(model_llmc, 'cpu')
            logger.info('video_gen eval: enabled inference_per_block for transformer blocks.')
        else:
            model_llmc.Pipeline.to('cuda')

        eval_res = self.eval_func(model_llmc, self.testenc, self.eval_dataset_bs, eval_pos)

        if self.inference_per_block:
            for h in handles:
                h.remove()
        gc.collect()
        model_llmc.Pipeline.to('cpu')
        torch.cuda.empty_cache()
        return eval_res

    def register_video_hooks(self, model):
        handles = []
        for layer in model.get_blocks():
            handles.append(layer.register_forward_pre_hook(self.forward_pre_hook))
        for layer in model.get_blocks():
            handles.append(layer.register_forward_hook(self.forward_hook))
        return handles

    @torch.no_grad()
    def move_video_blocks(self, model, device):
        for layer in model.get_blocks():
            layer.to(device)

    def pre_process(self, model, image_path):
        image = load_image(image_path)
        max_area = self.target_height * self.target_width
        aspect_ratio = image.height / image.width
        mod_value = model.Pipeline.vae_scale_factor_spatial * model.model.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        return image, width, height

    @torch.no_grad()
    def t2v_eval(self, model, testenc, bs, eval_pos):
        assert bs == 1, 'Only support eval bs=1'

        if self._is_dist_enabled() and int(os.environ.get('RANK', 0)) == 0:
            logger.info('video_gen eval: enabled distributed sample sharding across ranks.')

        for i, data in self._iter_with_global_index(testenc):
            pipe_kw = {
                'prompt': data['prompt'],
                'negative_prompt': data['negative_prompt'],
                'height': self.target_height,
                'width': self.target_width,
                'num_frames': self.num_frames,
                'guidance_scale': self.guidance_scale,
            }
            if self.guidance_scale_2 is not None:
                pipe_kw['guidance_scale_2'] = self.guidance_scale_2
            if self._is_dist_enabled() and self._supports_pipeline_generator(model):
                generator = torch.Generator(device='cuda')
                generator.manual_seed(int(self.config.base.seed) + int(i))
                pipe_kw['generator'] = generator
            output = model.Pipeline(**pipe_kw).frames[0]
            export_to_video(
                output,
                os.path.join(self.output_video_path, f'{eval_pos}_output_{i}.mp4'),
                fps=self.fps,
            )

        return None

    @torch.no_grad()
    def i2v_eval(self, model, testenc, bs, eval_pos):
        if self._is_dist_enabled() and int(os.environ.get('RANK', 0)) == 0:
            logger.info('video_gen eval: enabled distributed sample sharding across ranks.')

        for i, data in self._iter_with_global_index(testenc):
            image, width, height = self.pre_process(model, data['image'])

            pipe_kw = {
                'image': image,
                'prompt': data['prompt'],
                'negative_prompt': data['negative_prompt'],
                'height': height,
                'width': width,
                'num_frames': self.num_frames,
                'guidance_scale': self.guidance_scale,
            }
            if self.guidance_scale_2 is not None:
                pipe_kw['guidance_scale_2'] = self.guidance_scale_2
            if self._is_dist_enabled() and self._supports_pipeline_generator(model):
                generator = torch.Generator(device='cuda')
                generator.manual_seed(int(self.config.base.seed) + int(i))
                pipe_kw['generator'] = generator
            output = model.Pipeline(**pipe_kw).frames[0]

            export_to_video(
                output,
                os.path.join(self.output_video_path, f'{eval_pos}_output_{i}.mp4'),
                fps=self.fps,
            )

        return None

    @torch.no_grad()
    def eval_func(self, model, testenc, bs, eval_pos):
        assert bs == 1, 'Evaluation only supports batch size = 1.'
        assert self.model_type in ['WanT2V', 'WanI2V', 'Wan2T2V'], (
            f"Unsupported model type '{self.model_type}'.\n"
            'Only Wan video generation models (WanT2V, WanI2V, Wan2T2V) are supported.'
        )
        if self.eval_dataset_name == 't2v':
            return self.t2v_eval(model, testenc, bs, eval_pos)
        elif self.eval_dataset_name == 'i2v':
            return self.i2v_eval(model, testenc, bs, eval_pos)
        else:
            raise Exception(f'Unsupported eval dataset: {self.eval_dataset_name}')
