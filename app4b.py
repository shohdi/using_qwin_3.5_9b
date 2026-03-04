import inspect
import argparse
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES


MODEL_ID = "Qwen/Qwen3.5-4B"
CACHE_DIR = Path("./model_cache").resolve()

# Force local cache directory so first run downloads and next runs reuse cached files.
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_tokenizer = None
_model = None


def _assert_runtime_support():
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "Qwen3.5 requires Python >= 3.10 in this setup. "
            "Your current modelEnv is Python 3.9. "
            "Recreate environment: conda remove -n modelEnv --all -y && "
            "conda create -n modelEnv python=3.11 -y && conda activate modelEnv && "
            "pip install -r requirements.txt"
        )
    if "qwen3_5" not in CONFIG_MAPPING_NAMES:
        raise RuntimeError(
            "Installed transformers does not support model_type 'qwen3_5'. "
            "Install latest from source in modelEnv: "
            "pip install --upgrade git+https://github.com/huggingface/transformers.git@main"
        )


class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_ID)
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=512, ge=1, le=81920)
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 1.5
    extra_body: Optional[Dict[str, Any]] = None


def _extract_text_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    if isinstance(content, str):
        return content
    text_parts: List[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text_parts.append(part.get("text", ""))
    return "\n".join(p for p in text_parts if p).strip()


def _normalize_messages(messages: List[Message]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for m in messages:
        normalized.append({"role": m.role, "content": _extract_text_content(m.content)})
    return normalized


def _load_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    model_source: Union[str, Path]
    try:
        model_source = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(CACHE_DIR),
            local_files_only=True,
        )
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            f"Model '{MODEL_ID}' is not available in local cache '{CACHE_DIR}'. "
            "Connect to the internet once to warm the cache, then retry offline."
        ) from exc

    _tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )

    if torch.cuda.is_available():
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_source,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
            quantization_config=quant_cfg,
            device_map={"": 0},
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        _model = AutoModelForCausalLM.from_pretrained(
            model_source,
            cache_dir=str(CACHE_DIR),
            trust_remote_code=True,
            device_map="cpu",
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    _model.eval()
    return _tokenizer, _model


def _build_prompt(tokenizer, messages: List[Dict[str, str]], extra_body: Optional[Dict[str, Any]]) -> str:
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if extra_body and isinstance(extra_body, dict):
        chat_template_kwargs = extra_body.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            kwargs.update(chat_template_kwargs)

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Older template signatures may not accept chat-specific kwargs.
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        return tokenizer.apply_chat_template(messages, **kwargs)


def _filter_generate_args(model, args: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model.generate)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return {k: v for k, v in args.items() if v is not None}
    return {k: v for k, v in args.items() if k in sig.parameters and v is not None}


def _generate_from_messages(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    extra_body: Optional[Dict[str, Any]] = None,
):
    tokenizer, model = _load_model()
    prompt = _build_prompt(tokenizer, messages, extra_body)

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    top_k = 20
    min_p = 0.0
    repetition_penalty = 1.0
    if extra_body and isinstance(extra_body, dict):
        top_k = extra_body.get("top_k", top_k)
        min_p = extra_body.get("min_p", min_p)
        repetition_penalty = extra_body.get("repetition_penalty", repetition_penalty)

    do_sample = bool((temperature or 0.0) > 0.0)
    safe_max_tokens = min(max_tokens, 2048) if torch.cuda.is_available() else min(max_tokens, 512)

    generate_args = _filter_generate_args(
        model,
        {
            "max_new_tokens": safe_max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        },
    )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_args)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return output_text, int(prompt_len), int(generated_ids.shape[0])


@asynccontextmanager
async def lifespan(_: FastAPI):
    _assert_runtime_support()
    _load_model()
    yield


app = FastAPI(title="OpenAI-compatible Qwen API", version="1.0.0", lifespan=lifespan)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.get("/health")
def health():
    how_are_you, _, _ = _generate_from_messages(
        [{"role": "user", "content": "How are you?"}],
        max_tokens=128,
        temperature=0.7,
        top_p=0.8,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    csharp_hello, _, _ = _generate_from_messages(
        [{"role": "user", "content": "Write a hello world C# program."}],
        max_tokens=256,
        temperature=0.7,
        top_p=0.8,
        extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    )
    return {
        "status": "ok",
        "model": MODEL_ID,
        "checks": {
            "how_are_you_prompt": "How are you?",
            "how_are_you_response": how_are_you,
            "csharp_prompt": "Write a hello world C# program.",
            "csharp_response": csharp_hello,
        },
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not implemented in this local server")

    if req.model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Only model '{MODEL_ID}' is served by this API")

    messages = _normalize_messages(req.messages)
    output_text, prompt_tokens, completion_tokens = _generate_from_messages(
        messages=messages,
        max_tokens=req.max_tokens or 512,
        temperature=req.temperature or 1.0,
        top_p=req.top_p or 0.95,
        extra_body=req.extra_body,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run local OpenAI-compatible Qwen API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind. Default: 8001.")
    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
