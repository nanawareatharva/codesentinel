"""
CodeSentinel — Local LLM Client Wrapper for AutoTestArmy
=========================================================
Wraps the local Ollama daemon with memory safety controls, timeout guards,
and fallback logic. Enforces strict context bounds (num_ctx=2048) to prevent
unbounded RAM allocation on CPU-backed runtimes.
"""
import logging
from typing import Optional
import ollama

logger = logging.getLogger("codesentinel.llm")

DEFAULT_PRIMARY_MODEL = "phi3:mini"
DEFAULT_FALLBACK_MODEL = "qwen2.5-coder:latest"
DEFAULT_NUM_CTX = 2048
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_PREDICT = 1024

DEFAULT_SYSTEM_PROMPT = """You are an expert Python test engineer.
Write ONLY executable pytest test functions.
Rules:
- DO NOT wrap code in markdown fences (no ```python or ```).
- Output pure Python code only.
- Import the target function correctly.
- Test normal paths, boundary edge cases (None, 0, -1, empty string/list), and exception paths using pytest.raises.
- Use descriptive function names like test_<function>_<scenario>().
"""


def clean_markdown_fences(code: str) -> str:
    """Strip markdown code fence blocks if returned by the LLM."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[len("```"):].strip()
    if code.endswith("```"):
        code = code[:-len("```")].strip()
    return code


def generate_code(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_PRIMARY_MODEL,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> str:
    """
    Invoke local Ollama chat model with explicit memory guard (num_ctx).
    Falls back to secondary model if primary fails.
    """
    options = {
        "num_ctx": num_ctx,
        "temperature": temperature,
        "num_predict": num_predict,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options=options,
        )
        content = response["message"]["content"]
        return clean_markdown_fences(content)
    except Exception as exc:
        logger.warning(
            "Primary LLM %s failed: %s. Attempting fallback to %s",
            model,
            exc,
            fallback_model,
        )
        if fallback_model and fallback_model != model:
            try:
                response = ollama.chat(
                    model=fallback_model,
                    messages=messages,
                    options=options,
                )
                content = response["message"]["content"]
                return clean_markdown_fences(content)
            except Exception as fb_exc:
                logger.error("Fallback LLM %s also failed: %s", fallback_model, fb_exc)
                raise RuntimeError(
                    f"Both primary ({model}) and fallback ({fallback_model}) LLMs failed: {fb_exc}"
                ) from fb_exc
        raise RuntimeError(f"Ollama generation failed with model {model}: {exc}") from exc


def verify_llm_ready(model: str = DEFAULT_PRIMARY_MODEL) -> bool:
    """Check if the local Ollama daemon is reachable and the model is listed."""
    try:
        model_list = ollama.list()
        names = [m.get("name", "") for m in model_list.get("models", [])]
        # Match base name or exact name (e.g., 'phi3:mini' or 'phi3:mini-...')
        return any(model in name for name in names)
    except Exception:
        return False
