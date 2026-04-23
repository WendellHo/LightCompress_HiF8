from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import traceback

model_path = '/home/hezhiye/hif8_quant/quant_model/qwen_awq_w8_20260412_223244/vllm_quant_model'
try:
    model = LLM(model_path)
except Exception as e:
    traceback.print_exc()