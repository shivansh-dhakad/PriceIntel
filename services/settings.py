from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_ANON_KEY", "")).strip()
USE_SUPABASE = os.getenv("USE_SUPABASE", "1").strip().lower() not in {"0", "false", "no"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

SCRAPER_USER_AGENT = os.getenv(
    "SCRAPER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
)
SCRAPER_TIMEOUT = float(os.getenv("SCRAPER_TIMEOUT", "12"))

TECHSPECS_API_KEY = os.getenv("TECHSPECS_API_KEY", "").strip()
TECHSPECS_API_ID  = os.getenv("TECHSPECS_API_ID",  "").strip()
TECHSPECS_BASE_URL = "https://api.techspecs.io/v5"
