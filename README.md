# Local OpenAI-Compatible API for Qwen3.5-9B

This project runs a local API that is compatible with OpenAI Chat Completions:

- `POST /v1/chat/completions`
- `GET /v1/models`

Model served: `Qwen/Qwen3.5-9B`

## 1. Environment setup (Conda)

```powershell
conda remove -n modelEnv --all -y
conda create -n modelEnv python=3.11 -y
conda activate modelEnv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## 2. Run the API server

```powershell
conda activate modelEnv
python app.py
```

Server starts at:

- `http://localhost:8000/v1`

## 3. Cache behavior

- First server start downloads model files into `./model_cache`.
- Next starts reuse cached files from `./model_cache` (no full re-download).
- `model_cache/` is ignored in git.

## 4. Call it like OpenAI Chat API

### cURL

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "Qwen/Qwen3.5-9B",
    "messages": [
      {"role": "user", "content": "Type \"I love Qwen3.5\" backwards"}
    ],
    "max_tokens": 512,
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5,
    "extra_body": {
      "top_k": 20
    }
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

## Notes for your hardware (8GB VRAM / 32GB RAM)

- The server uses 4-bit loading on CUDA (`bitsandbytes`) to fit better on 8GB GPU.
- `max_tokens` is safety-capped to reduce OOM risk (`2048` on GPU).
- This implementation is text-first for chat completions and OpenAI-compatible request format.

## Troubleshooting

If you get:

- `KeyError: 'qwen3_5'`
- `Transformers does not recognize this architecture`

then your environment is too old (usually Python 3.9 + old transformers).

Run:

```powershell
conda remove -n modelEnv --all -y
conda create -n modelEnv python=3.11 -y
conda activate modelEnv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install --upgrade git+https://github.com/huggingface/transformers.git@main
```
