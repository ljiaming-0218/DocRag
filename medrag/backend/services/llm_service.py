import logging
from time import perf_counter

import openai
from openai import OpenAI

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL


logger = logging.getLogger("uvicorn.error")

class LLMServiceError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
    
def log_llm_failure(code: str, operation: str, started_at: float, error: Exception,) -> None:
    elapsed = perf_counter() - started_at

    logger.warning(
        "llm_call_failed "
        "code=%s "
        "operation=%s "
        "model=%s "
        "elapsed_seconds=%.2f "
        "status_code=%s "
        "request_id=%s "
        "error_type=%s",
        code,
        operation,
        OPENROUTER_MODEL,
        elapsed,
        getattr(error, "status_code", None),
        getattr(error, "request_id", None),
        type(error).__name__,
    )

def create_llm_client():
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        timeout=120,
        max_retries=1,
    )


def generate_answer(prompt: str, operation: str = "answer") -> str:

    client = create_llm_client()
    started_at = perf_counter()
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": prompt,
                }
            ],
            temperature=0.1,
        )
    except openai.APITimeoutError as exc:
        log_llm_failure("LLM_TIMEOUT", operation, started_at, exc)

        raise LLMServiceError(
            "LLM_TIMEOUT",
            "大模型请求超时",
            504,
        ) from exc
    except openai.RateLimitError as exc:
        log_llm_failure("LLM_RATE_LIMITED", operation, started_at, exc)

        raise LLMServiceError(
            "LLM_RATE_LIMITED",
            "大模型服务繁忙，请稍后重试",
            503,
        ) from exc
    except openai.APIConnectionError as exc:
        log_llm_failure("LLM_CONNECTION_FAILED", operation, started_at, exc)

        raise LLMServiceError(
            "LLM_CONNECTION_FAILED",
            "无法连接大模型服务",
            503,
        ) from exc
    except openai.APIStatusError as exc:
        log_llm_failure("LLM_API_ERROR", operation, started_at, exc)

        raise LLMServiceError(
            "LLM_API_ERROR",
            "大模型服务返回异常",
            502,
        ) from exc

    if not response.choices:
        error = LLMServiceError("LLM_EMPTY_RESPONSE","大模型没有返回候选答案",502,)
        log_llm_failure(error.code, operation, started_at, error,)
        raise error

    content = response.choices[0].message.content

    if not content or not content.strip():
        error = LLMServiceError("LLM_EMPTY_CONTENT", "大模型返回了空回答", 502)
        log_llm_failure(error.code, operation, started_at, error)
        raise error

    elapsed = perf_counter() - started_at

    logger.info(
        "llm_call_succeeded operation=%s model=%s elapsed_seconds=%.2f",
        operation,
        OPENROUTER_MODEL,
        elapsed,
    )
        
    return content.strip()
    