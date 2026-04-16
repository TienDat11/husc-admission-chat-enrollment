"""Apply fixes to eval_core.py"""
import re

file = 'packages/rag-chatbot-husc/src/notebooks/eval_core.py'
with open(file, 'utf8') as f:
    content = f.read()

# 1. Add PipelineError class after imports
old_imports = """from typing import Any


def load_test_questions"""

new_imports = """from typing import Any


class PipelineError(Exception):
    """Raised when call_pipeline fails due to connection or response errors."""


def load_test_questions"""

content = content.replace(old_imports, new_imports)

# 2. Update call_pipeline to handle errors
old_code = """    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
"""

new_code = """    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as e:
        raise PipelineError(
            f"Failed to connect to pipeline at {url}: {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise PipelineError(
            f"Pipeline at {url} returned non-JSON response: {e}"
        ) from e
"""

content = content.replace(old_code, new_code)

with open(file, 'utf8') as f:
    f.write(content)

print("Done")
