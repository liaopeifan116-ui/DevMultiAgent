"""Central configuration. Reads DEEPSEEK_API_KEY from .env file or env var."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RULES_DIR = PROJECT_ROOT / "devma" / "rules"
EVIDENCE_DIR = PROJECT_ROOT / "devma" / "evidence"

# Load .env file if exists
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip().strip('"').strip("'"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Intent config
INTENTS = ["after_sales", "tech_support", "logistics"]
MAX_SOLVER_RETRY = 2

# SQLite
ORDER_DB_PATH = str(DATA_DIR / "orders.db")
