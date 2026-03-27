"""
Centralized environment loader.

Import this module FIRST in any entry point (main.py, build_graph.py, etc.)
to ensure .env is loaded before any service reads os.getenv().

Python's import cache guarantees load_dotenv() runs exactly once per process.
"""
from dotenv import load_dotenv

load_dotenv()
