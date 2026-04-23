from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_path = '/home/hezhiye/hif8_quant/quant_model/qwen_awq_w8_20260412_223244/vllm_quant_model'
model = LLM(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompts = [
    'Hello, my name is',
    'The president of the United States is',
    'The capital of France is',
    'The future of AI is',
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = model.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')