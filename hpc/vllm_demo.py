"""
Minimal vLLM inference demo.

Runs on any NVIDIA GPU node (your university HPC, or Colab).
Called by run_vllm.slurm, or run directly:  python vllm_demo.py
"""

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------
# Model choice: start small, then scale up once it works.
#   ~1B   : "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#   ~1.5B : "Qwen/Qwen2.5-1.5B-Instruct"
#   ~3.8B : "microsoft/Phi-3-mini-4k-instruct"
#   ~7-8B : "meta-llama/Meta-Llama-3.1-8B-Instruct"  (needs a 24GB+ GPU)
# ---------------------------------------------------------------------
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# dtype note:
#   "auto"  -> bfloat16 on A100 / H100 (compute capability >= 8.0)
#   "half"  -> float16, REQUIRED on older GPUs like T4 / V100 (no bfloat16)
# Your USD cluster has V100 (Volta), so keep this as "half".
DTYPE = "half"

llm = LLM(
    model=MODEL,
    dtype=DTYPE,
    gpu_memory_utilization=0.90,   # fraction of VRAM vLLM may use
    max_model_len=2048,            # lower this if you hit out-of-memory
)

prompts = [
    "The future of artificial intelligence is",
    "In a distant galaxy, a lone explorer discovered",
    "Explain self-attention in transformers in one sentence:",
]

params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

outputs = llm.generate(prompts, params)

for out in outputs:
    print("=" * 70)
    print("PROMPT :", out.prompt)
    print("OUTPUT :", out.outputs[0].text.strip())
