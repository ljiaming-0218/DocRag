import os
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "medrag" / "backend"

sys.path.insert(0, str(BACKEND_DIR))

TEST_RUNTIME_DIR = Path(tempfile.gettempdir()) / "docrag-tests"
os.environ.setdefault("MONGODB_URI", "mongodb://127.0.0.1:27017")
os.environ.setdefault("MONGODB_DB_NAME", "docrag_test")
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-secret-key-that-is-long-enough-for-tests",
)
os.environ.setdefault(
    "CHROMA_DIR",
    str(TEST_RUNTIME_DIR / "chroma_db"),
)
os.environ.setdefault(
    "UPLOAD_DIR",
    str(TEST_RUNTIME_DIR / "uploads"),
)
