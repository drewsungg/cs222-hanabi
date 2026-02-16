"""Configuration loaded from environment / .env file."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "anthropic"
LLM_MODEL = os.getenv("LLM_MODEL", "")  # blank = use default for provider
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
