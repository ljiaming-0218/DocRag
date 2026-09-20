from pathlib import Path
from os import environ
from dotenv import load_dotenv


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

APP_VERSION = environ.get("APP_VERSION", "dev").strip() or "dev"
BUILD_COMMIT = (
    environ.get("BUILD_COMMIT", "").strip()
    or "unknown"
)

EMBEDDING_MODEL = (
    environ.get("EMBEDDING_MODEL", "BAAI/bge-m3").strip()
    or "BAAI/bge-m3"
)
EMBEDDING_PROVIDER = (
    environ.get("EMBEDDING_PROVIDER", "siliconflow").strip().lower()
    or "siliconflow"
)
EMBEDDING_API_KEY = environ.get("EMBEDDING_API_KEY", "").strip()
EMBEDDING_API_BASE = (
    environ.get(
        "EMBEDDING_API_BASE",
        "https://api.siliconflow.cn/v1",
    ).strip().rstrip("/")
)
CHROMA_DIR = Path(environ["CHROMA_DIR"])
UPLOAD_DIR = Path(environ["UPLOAD_DIR"])
MONGODB_URI = environ.get("MONGODB_URI")
MONGODB_DB_NAME = environ.get("MONGODB_DB_NAME")
REWRITE_LLM_PROVIDER = (
    environ.get("REWRITE_LLM_PROVIDER", "modelscope").strip().lower()
    or "modelscope"
)
REWRITE_LLM_API_KEY = environ.get("REWRITE_LLM_API_KEY", "").strip()
REWRITE_LLM_BASE_URL = (
    environ.get("REWRITE_LLM_BASE_URL", "").strip().rstrip("/")
)
REWRITE_LLM_MODEL = environ.get("REWRITE_LLM_MODEL", "").strip()

ANSWER_LLM_PROVIDER = (
    environ.get("ANSWER_LLM_PROVIDER", "modelscope").strip().lower()
    or "modelscope"
)
ANSWER_LLM_API_KEY = environ.get("ANSWER_LLM_API_KEY", "").strip()
ANSWER_LLM_BASE_URL = (
    environ.get("ANSWER_LLM_BASE_URL", "").strip().rstrip("/")
)
ANSWER_LLM_MODEL = environ.get("ANSWER_LLM_MODEL", "").strip()
RETRIEVAL_MODE = environ.get("RETRIEVAL_MODE", "dense").strip().lower()


def _get_optional_float(name: str) -> float | None:
    value = environ.get(name, "").strip()
    return float(value) if value else None


def _get_positive_int(name: str, default: int) -> int:
    value = int(environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _get_non_negative_int(name: str, default: int) -> int:
    value = int(environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} 不能小于 0")
    return value


def _get_optional_positive_int(name: str) -> int | None:
    raw_value = environ.get(name, "").strip()
    if not raw_value:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


EVIDENCE_RERANK_MIN_SCORE = _get_optional_float(
    "EVIDENCE_RERANK_MIN_SCORE"
)
EMBEDDING_BATCH_SIZE = _get_positive_int("EMBEDDING_BATCH_SIZE", 32)
EMBEDDING_DIMENSION = _get_optional_positive_int("EMBEDDING_DIMENSION")
EMBEDDING_TIMEOUT_SECONDS = _get_positive_int(
    "EMBEDDING_TIMEOUT_SECONDS",
    60,
)
EMBEDDING_MAX_RETRIES = _get_non_negative_int(
    "EMBEDDING_MAX_RETRIES",
    2,
)
VECTOR_WRITE_BATCH_SIZE = _get_positive_int(
    "VECTOR_WRITE_BATCH_SIZE",
    100,
)
REWRITE_LLM_TIMEOUT_SECONDS = _get_positive_int(
    "REWRITE_LLM_TIMEOUT_SECONDS",
    20,
)
REWRITE_LLM_MAX_RETRIES = _get_non_negative_int(
    "REWRITE_LLM_MAX_RETRIES",
    0,
)
ANSWER_LLM_TIMEOUT_SECONDS = _get_positive_int(
    "ANSWER_LLM_TIMEOUT_SECONDS",
    120,
)
ANSWER_LLM_MAX_RETRIES = _get_non_negative_int(
    "ANSWER_LLM_MAX_RETRIES",
    1,
)
SUMMARY_SEED_K = _get_positive_int("SUMMARY_SEED_K", 5)
SUMMARY_SUBQUERY_RETRIEVE_K = _get_positive_int(
    "SUMMARY_SUBQUERY_RETRIEVE_K",
    10,
)
SUMMARY_SUBQUERY_KEEP_K = _get_positive_int(
    "SUMMARY_SUBQUERY_KEEP_K",
    3,
)
SUMMARY_CONTEXT_K = _get_positive_int("SUMMARY_CONTEXT_K", 10)
SUMMARY_MAX_SUBQUERIES = _get_positive_int(
    "SUMMARY_MAX_SUBQUERIES",
    6,
)

if SUMMARY_SUBQUERY_KEEP_K > SUMMARY_SUBQUERY_RETRIEVE_K:
    raise ValueError(
        "SUMMARY_SUBQUERY_KEEP_K 不能大于 "
        "SUMMARY_SUBQUERY_RETRIEVE_K"
    )

OCR_ENABLED = environ.get("OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
OCR_LANGUAGES = environ.get("OCR_LANGUAGES", "eng+chi_sim")
OCR_DPI = int(environ.get("OCR_DPI", "300"))
OCR_MIN_TEXT_CHARS = int(environ.get("OCR_MIN_TEXT_CHARS", "20"))
OCR_TESSDATA_DIR = environ.get("TESSDATA_PREFIX")
MAX_UPLOAD_SIZE_MB = int(
    environ.get("MAX_UPLOAD_SIZE_MB", "20")
)

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)

JWT_SECRET_KEY = environ.get("JWT_SECRET_KEY", "").strip()
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = _get_positive_int(
    "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
    60,
)
