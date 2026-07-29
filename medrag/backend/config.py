from pathlib import Path
from os import environ
from dotenv import load_dotenv


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


CHROMA_DIR = Path(environ["CHROMA_DIR"])
UPLOAD_DIR = Path(environ["UPLOAD_DIR"])
MONGODB_URI = environ.get("MONGODB_URI")
MONGODB_DB_NAME = environ.get("MONGODB_DB_NAME")
OPENROUTER_API_KEY = environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = environ.get("OPENROUTER_BASE_URL")
OPENROUTER_MODEL = environ.get("OPENROUTER_MODEL")

OCR_ENABLED = environ.get("OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
OCR_LANGUAGES = environ.get("OCR_LANGUAGES", "eng+chi_sim")
OCR_DPI = int(environ.get("OCR_DPI", "300"))
OCR_MIN_TEXT_CHARS = int(environ.get("OCR_MIN_TEXT_CHARS", "20"))
OCR_TESSDATA_DIR = environ.get("TESSDATA_PREFIX")
