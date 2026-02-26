"""
pytest configuration for the project

this ensures that the project root is on sys.path so tests can import
top-level modules like `main` and `model` without import errors
"""

import sys  
from pathlib import Path 


# compute project root directory.
project_root = Path(__file__).resolve().parents[1]


# ensure project root is present on sys.path.
if str(project_root) not in sys.path:  

    # prepend project root for module resolution.
    sys.path.insert(0, str(project_root))  

