import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import Lock
from time import perf_counter

import openai
from openai import OpenAI

from config import (
    ANSWER_LLM_API_KEY,
    ANSWER_LLM_BASE_URL,
    ANSWER_LLM_MAX_RETRIES,
    ANSWER_LLM_MODEL,
    ANSWER_LLM_PROVIDER,
    ANSWER_LLM_TIMEOUT_SECONDS,
    REWRITE_LLM_API_KEY,
    REWRITE_LLM_BASE_URL,
    REWRITE_LLM_MAX_RETRIES,
    REWRITE_LLM_MODEL,
    REWRITE_LLM_PROVIDER,
    REWRITE_LLM_TIMEOUT_SECONDS,
)


logger = logging.getLogger("uvicorn.error")
_clients: dict[str, OpenAI] = {}
_client_lock = Lock()
REWRITE_OPERATIONS = {"task_route", "query_rewrite", "summary_query"}
_usage_records: ContextVar[list[dict] | None] = ContextVar(
    "llm_usage_records",
    default=None,
)


@dataclass(frozen=True)
class LLMRoleConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int
    max_retries: int


class LLMServiceError(RuntimeError):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def begin_llm_usage_tracking() -> Token:
    return _usage_records.set([])


def finish_llm_usage_tracking(token: Token) -> dict:
    records = list(_usage_records.get() or [])
    _usage_records.reset(token)
    return {
        "prompt_tokens": sum(item["prompt_tokens"] for item in records),
        "completion_tokens": sum(
            item["completion_tokens"] for item in records
        ),
        "total_tokens": sum(item["total_tokens"] for item in records),
        "calls": records,
    }


def record_llm_usage(response, operation: str, role: str) -> None:
    records = _usage_records.get()
    usage = getattr(response, "usage", None)
    if records is None or usage is None:
        return
    records.append({
        "operation": operation,
        "role": role,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(
            getattr(usage, "completion_tokens", 0) or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    })


def resolve_llm_role(operation: str) -> str:
    if operation.strip().lower() in REWRITE_OPERATIONS:
        return "rewrite"
    return "answer"


def get_llm_config(role: str) -> LLMRoleConfig:
    if role == "rewrite":
        return LLMRoleConfig(
            provider=REWRITE_LLM_PROVIDER,
            api_key=REWRITE_LLM_API_KEY,
            base_url=REWRITE_LLM_BASE_URL,
            model=REWRITE_LLM_MODEL,
            timeout_seconds=REWRITE_LLM_TIMEOUT_SECONDS,
            max_retries=REWRITE_LLM_MAX_RETRIES,
        )
    if role == "answer":
        return LLMRoleConfig(
            provider=ANSWER_LLM_PROVIDER,
            api_key=ANSWER_LLM_API_KEY,
            base_url=ANSWER_LLM_BASE_URL,
            model=ANSWER_LLM_MODEL,
            timeout_seconds=ANSWER_LLM_TIMEOUT_SECONDS,
            max_retries=ANSWER_LLM_MAX_RETRIES,
        )
    raise ValueError(f"未知的 LLM 角色: {role}")


def log_llm_failure(
    code: str,
    operation: str,
    role: str,
    config: LLMRoleConfig,
    started_at: float,
    error: Exception,
) -> None:
    elapsed = perf_counter() - started_at
    logger.warning(
        "llm_call_failed code=%s operation=%s role=%s provider=%s "
        "model=%s elapsed_seconds=%.2f status_code=%s request_id=%s "
        "error_type=%s",
        code,
        operation,
        role,
        config.provider,
        config.model,
        elapsed,
        getattr(error, "status_code", None),
        getattr(error, "request_id", None),
        type(error).__name__,
    )


def create_llm_client(role: str) -> OpenAI:
    if role in _clients:
        return _clients[role]

    config = get_llm_config(role)
    missing_fields = [
        name
        for name, value in (
            (f"{role.upper()}_LLM_API_KEY", config.api_key),
            (f"{role.upper()}_LLM_BASE_URL", config.base_url),
            (f"{role.upper()}_LLM_MODEL", config.model),
        )
        if not isinstance(value, str) or not value.strip()
    ]
    if missing_fields:
        raise LLMServiceError(
            "LLM_NOT_CONFIGURED",
            f"大模型配置缺失: {', '.join(missing_fields)}",
            503,
        )

    with _client_lock:
        if role in _clients:
            return _clients[role]

        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        _clients[role] = client
        logger.info(
            "llm_client_initialized role=%s provider=%s model=%s",
            role,
            config.provider,
            config.model,
        )
        return client


def generate_answer(prompt: str, operation: str = "answer") -> str:
    started_at = perf_counter()
    role = resolve_llm_role(operation)
    config = get_llm_config(role)

    try:
        client = create_llm_client(role)
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
    except LLMServiceError as exc:
        log_llm_failure(exc.code, operation, role, config, started_at, exc)
        raise
    except openai.APITimeoutError as exc:
        log_llm_failure("LLM_TIMEOUT", operation, role, config, started_at, exc)
        raise LLMServiceError("LLM_TIMEOUT", "大模型请求超时", 504) from exc
    except openai.RateLimitError as exc:
        log_llm_failure(
            "LLM_RATE_LIMITED", operation, role, config, started_at, exc
        )
        raise LLMServiceError(
            "LLM_RATE_LIMITED", "大模型服务繁忙，请稍后重试", 503
        ) from exc
    except openai.APIConnectionError as exc:
        log_llm_failure(
            "LLM_CONNECTION_FAILED", operation, role, config, started_at, exc
        )
        raise LLMServiceError(
            "LLM_CONNECTION_FAILED", "无法连接大模型服务", 503
        ) from exc
    except openai.APIStatusError as exc:
        log_llm_failure("LLM_API_ERROR", operation, role, config, started_at, exc)
        raise LLMServiceError(
            "LLM_API_ERROR", "大模型服务返回异常", 502
        ) from exc

    if not response.choices:
        error = LLMServiceError(
            "LLM_EMPTY_RESPONSE", "大模型没有返回候选答案", 502
        )
        log_llm_failure(error.code, operation, role, config, started_at, error)
        raise error

    record_llm_usage(response, operation, role)

    content = response.choices[0].message.content
    if not content or not content.strip():
        error = LLMServiceError(
            "LLM_EMPTY_CONTENT", "大模型返回了空回答", 502
        )
        log_llm_failure(error.code, operation, role, config, started_at, error)
        raise error

    elapsed = perf_counter() - started_at
    logger.info(
        "llm_call_succeeded operation=%s role=%s provider=%s model=%s "
        "elapsed_seconds=%.2f",
        operation,
        role,
        config.provider,
        config.model,
        elapsed,
    )
    return content.strip()
